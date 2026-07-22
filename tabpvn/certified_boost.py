"""Proof-carrying ADDITIVE certified predictor — the architectural lift that raises accuracy WITHOUT giving
up the proof.

Key idea: a single certified region tree is low-capacity (loses to GBDT). But an ADDITIVE ensemble of
certified region trees is still sound — because a sum of certified terms is certified. Each stage m routes a
row to a leaf region (a threshold conjunction the FOLKernel verifies by closure); the prediction is
    f(x) = base + Σ_m lr · value_m(leaf_m(x))
a fixed linear combination of KERNEL-VERIFIED region memberships. So the FOLKernel reproduces f(x) exactly
(certify = 1.0), and the PROOF of a prediction is the list of fired regions across stages, each with its
membership proof, plus the arithmetic sum. Longer proof, same soundness — and full ensemble accuracy.

This lifts every category on the same principle:
  regression     -> additive region VALUES               (below; beats GBDT on Sendy)
  classification -> additive per-class region LOGITS      (argmax; each region kernel-verified)
  relational     -> weighted vote over τ-gated rules      (each rule kernel-verified)

"""

from __future__ import annotations

import numpy as np

from core.kernel_fol import FOLKernel
from tabpvn.region_algebra import canonicalize_conjunction
from tabpvn.trees import (  # OUR OWN (no sklearn)
    _HAS_NUMBA,
    _affine_tree_pred,
    _cat_member,
    _forest_predict_adaptive_binary_mirrored_nb,
    _forest_predict_adaptive_flat_nb,
    _forest_predict_adaptive_multiclass_shared_nb,
    _forest_scores_affine_nb,
    _forest_scores_binary_mirrored_nb,
    _forest_scores_flat_nb,
    _forest_scores_multiclass_shared_nb,
    _numba_threading_is_threadsafe,
    _pack_affine_forest,
    _pack_binary_mirrored_forest,
    _pack_flat_forest,
    _pack_multiclass_shared_forest,
    _prior_preserving_subset_weight,
    _route_flat_nb,
    _tree_interval_flat_nb,
    _tree_pred,
    boost_predict,
    reason_boost_2nd,
    reason_boost_softmax,
)

# At/above this many rows the softmax booster grows its K per-class trees concurrently (see
# reason_boost_softmax `parallel_k`). The legacy large-table path remains enabled regardless of refit mode.
# A smaller, final-fit-only path below also uses this for compact constant-leaf classifiers, while search and
# OOF fits stay serial so their fold/config worker pools never nest with a per-class pool.
_PARALLEL_K_MIN_ROWS = 50_000
_PARALLEL_K_FINAL_MIN_ROWS = 400

# Min-fill is a useful deterministic separator heuristic on compact tree
# feature graphs.  A very wide one-hot table needs a dedicated sparse ordering
# implementation before it is worth compiling; fail closed for now so the
# verifier index never adds unbounded fit cost to the zero-knob path.
_COMPILED_REGION_MAX_FEATURES = 128
# At the 2M production cap this retains 500K rows of broad history while
# concentrating 1.5M rows on the deployment regime. Controlled stable/drift
# sweeps favored this over 50/50 and 2/3 allocations without adding a fit.
_TEMPORAL_RECENT_SHARE = 0.75
_TEMPORAL_HISTORY_BANDS = 6

# Check once per eight semantic boosting rounds. The generic scalar layout has
# one routed tree per class and therefore uses a proportionally wider stride.
# Short forests retain the branch-free fused scorer because they cannot
# amortize an intermediate certificate check.
_ADAPTIVE_DEPTH_ROUNDS_PER_CHECK = 8
_ADAPTIVE_DEPTH_AUDIT_ROWS = 512
_ADAPTIVE_DEPTH_MIN_STAGE_REDUCTION = 0.25
_ADAPTIVE_DEPTH_SHALLOW_MIN_STAGE_REDUCTION = 0.30
_ADAPTIVE_DEPTH_SHARED_MIN_STAGE_REDUCTION = 0.35


def _fit_sample(y, cap, seed, stratified, min_class_rows=0):
    """Deterministic bounded sample plus inverse-inclusion weights."""
    y = np.asarray(y)
    n = len(y)
    if cap is None or n <= int(cap):
        return None, None
    cap = int(cap)
    if cap < 1:
        raise ValueError("fit_cap must be a positive integer or None")
    rng = np.random.default_rng(seed)
    if not stratified:
        return rng.choice(n, cap, replace=False).astype(np.int64, copy=False), None
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if cap < len(classes):
        raise ValueError("fit_cap must be at least the number of target classes")

    ideal = counts.astype(float) * (cap / n)
    quotas = np.floor(ideal).astype(np.int64)
    quotas = np.minimum(counts, np.maximum(quotas, 1))
    minimum = np.ones(len(classes), dtype=np.int64)
    if min_class_rows:
        minimum = np.minimum(counts, int(min_class_rows))
        if int(minimum.sum()) > cap:
            raise ValueError("fit_cap is too small for the requested per-class evidence floor")
        quotas = np.maximum(quotas, minimum)
    fractions = ideal - np.floor(ideal)
    remaining = cap - int(quotas.sum())
    while remaining > 0:
        available = np.flatnonzero(quotas < counts)
        order = available[np.lexsort((available, -counts[available], -fractions[available]))]
        take = min(remaining, len(order))
        quotas[order[:take]] += 1
        remaining -= take
    while remaining < 0:
        available = np.flatnonzero(quotas > minimum)
        order = available[np.lexsort((available, -counts[available], fractions[available]))]
        take = min(-remaining, len(order))
        quotas[order[:take]] -= 1
        remaining += take

    sampled, sample_weights = [], []
    for class_idx, quota in enumerate(quotas):
        rows = np.flatnonzero(inverse == class_idx)
        part = rows if quota == len(rows) else rng.choice(rows, int(quota), replace=False)
        sampled.append(part)
        if min_class_rows:
            inclusion_weight = (counts[class_idx] / float(quota)) * (cap / float(n))
            sample_weights.append(np.full(len(part), inclusion_weight, dtype=float))
    out = np.concatenate(sampled).astype(np.int64, copy=False)
    weights = np.concatenate(sample_weights) if sample_weights else None
    order = rng.permutation(len(out))
    return out[order], (None if weights is None else weights[order])


def _fit_rows(y, cap, seed, stratified):
    """Compatibility wrapper returning only deterministic bounded rows."""
    return _fit_sample(y, cap, seed, stratified)[0]


def _temporal_history_rows(n_rows, recent_rows):
    """Rows outside the dense recent reservoir, with a cheap contiguous-tail path."""
    recent_rows = np.asarray(recent_rows, dtype=np.int64)
    if len(recent_rows) and np.array_equal(
        recent_rows,
        np.arange(int(recent_rows[0]), n_rows, dtype=np.int64),
    ):
        return np.arange(int(recent_rows[0]), dtype=np.int64)
    retained = np.zeros(n_rows, dtype=bool)
    retained[recent_rows] = True
    return np.flatnonzero(~retained).astype(np.int64, copy=False)


def _multiscale_band_quotas(sizes, budget):
    """Water-fill an exact budget across newest-to-oldest age bands."""
    sizes = np.asarray(sizes, dtype=np.int64)
    quotas = np.zeros(len(sizes), dtype=np.int64)
    active = list(range(len(sizes)))
    remaining = int(budget)
    while remaining and active:
        share, extra = divmod(remaining, len(active))
        filled = []
        for rank, index in enumerate(active):
            capacity = int(sizes[index] - quotas[index])
            take = min(capacity, share + (rank < extra))
            quotas[index] += take
            remaining -= take
            if quotas[index] == sizes[index]:
                filled.append(index)
        active = [index for index in active if index not in filled]
    if remaining:
        raise RuntimeError("unable to allocate temporal history budget")
    return quotas


def _multiscale_history_sample(history_rows, groups, budget, seed):
    """Sample logarithmic age bands so recurring old regimes remain visible."""
    history_rows = np.asarray(history_rows, dtype=np.int64)
    if budget <= 0:
        return np.empty(0, dtype=np.int64), (), ()
    chronological = bool(np.all(groups[1:] >= groups[:-1]))
    ordered = (
        history_rows
        if chronological and np.all(history_rows[1:] >= history_rows[:-1])
        else history_rows[np.lexsort((history_rows, groups[history_rows]))]
    )
    band_count = min(_TEMPORAL_HISTORY_BANDS, int(budget), len(ordered))
    base_width = max(1, int(budget) // band_count)
    bands = []
    end = len(ordered)
    for scale in range(band_count - 1):
        width = base_width * (2**scale)
        start = max(0, end - width)
        bands.append(ordered[start:end])
        end = start
        if end == 0:
            break
    if end:
        bands.append(ordered[:end])
    sizes = np.asarray([len(band) for band in bands], dtype=np.int64)
    quotas = _multiscale_band_quotas(sizes, budget)
    sampled = []
    for index, (band, quota) in enumerate(zip(bands, quotas, strict=True)):
        if quota == len(band):
            sampled.append(band)
        elif quota:
            rng = np.random.default_rng(seed + 104_729 * (index + 1))
            sampled.append(rng.choice(band, int(quota), replace=False))
    return (
        np.concatenate(sampled).astype(np.int64, copy=False),
        tuple(int(size) for size in sizes),
        tuple(int(quota) for quota in quotas),
    )


def _ensure_temporal_class_floor(selected, y, groups, min_class_rows):
    """Replace oldest surplus rows until every source class meets its evidence floor."""
    selected = np.asarray(selected, dtype=np.int64)
    classes, inverse, source_counts = np.unique(y, return_inverse=True, return_counts=True)
    if len(selected) < len(classes):
        raise ValueError("fit_cap must be at least the number of target classes")
    minimum = np.ones(len(classes), dtype=np.int64)
    if min_class_rows:
        minimum = np.minimum(source_counts, int(min_class_rows))
    if int(minimum.sum()) > len(selected):
        raise ValueError("fit_cap is too small for the requested per-class evidence floor")

    selected_mask = np.zeros(len(y), dtype=bool)
    selected_mask[selected] = True
    selected_counts = np.bincount(inverse[selected], minlength=len(classes)).astype(np.int64)
    for class_index in np.flatnonzero(selected_counts < minimum):
        deficit = int(minimum[class_index] - selected_counts[class_index])
        candidates = np.flatnonzero((inverse == class_index) & ~selected_mask)
        candidate_order = np.lexsort((candidates, groups[candidates]))
        additions = candidates[candidate_order[-deficit:]]
        selected_counts[class_index] += len(additions)

        donor_positions = []
        for position in np.lexsort((selected, groups[selected])):
            donor_class = int(inverse[selected[position]])
            if selected_counts[donor_class] <= minimum[donor_class]:
                continue
            selected_counts[donor_class] -= 1
            donor_positions.append(int(position))
            if len(donor_positions) == len(additions):
                break
        if len(donor_positions) != len(additions):
            raise RuntimeError("unable to satisfy temporal class evidence floor")
        keep = np.ones(len(selected), dtype=bool)
        keep[donor_positions] = False
        selected = np.concatenate([selected[keep], additions]).astype(np.int64, copy=False)
        selected_mask.fill(False)
        selected_mask[selected] = True
    return selected, inverse, source_counts


def _temporal_fit_sample(y, validation_groups, cap, seed, stratified, min_class_rows=0):
    """Hybrid recent/history reservoir for event fitting under a hard row cap.

    A dense recent 75% carries current regimes while logarithmic age bands in
    the remaining budget retain recurring and long-range patterns.
    Classification floors are enforced after all strata are joined; rare-floor
    weighting preserves the natural pre-floor temporal prior.
    """
    from tabpvn.validation import FutureValidation

    y = np.asarray(y)
    groups = np.asarray(validation_groups, dtype=np.int64)
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError("validation_groups must have one value per fitted row")
    n_rows = len(y)
    if cap is None or n_rows <= int(cap):
        return (
            None,
            None,
            {
                "mode": "temporal_full",
                "source_rows": int(n_rows),
                "sample_rows": int(n_rows),
            },
        )
    cap = int(cap)
    if cap < 2:
        raise ValueError("temporal fit_cap must be at least two rows")

    validation = FutureValidation(groups)
    if cap == 2:
        recent_rows = validation.bounded_rows(cap)
    else:
        recent_budget = max(2, min(cap - 1, int(round(cap * _TEMPORAL_RECENT_SHARE))))
        recent_rows = validation.bounded_rows(recent_budget)
    history_rows = _temporal_history_rows(n_rows, recent_rows)
    history_budget = cap - len(recent_rows)
    if history_budget:
        history_sample, history_band_rows, history_band_samples = _multiscale_history_sample(
            history_rows,
            groups,
            history_budget,
            seed,
        )
        selected = np.concatenate([history_sample, recent_rows])
    else:
        history_band_rows, history_band_samples = (), ()
        selected = recent_rows.copy()

    inclusion_weight = None
    if stratified:
        classes, initial_inverse, _source_counts = np.unique(
            y,
            return_inverse=True,
            return_counts=True,
        )
        baseline_counts = np.bincount(initial_inverse[selected], minlength=len(classes)).astype(float)
        selected, inverse, source_counts = _ensure_temporal_class_floor(
            selected,
            y,
            groups,
            min_class_rows,
        )
        if min_class_rows:
            sample_counts = np.bincount(inverse[selected], minlength=len(source_counts))
            target_counts = baseline_counts.copy()
            for class_index in np.flatnonzero((target_counts == 0) & (source_counts > 0)):
                donor = int(np.argmax(target_counts))
                if target_counts[donor] <= 1.0:
                    raise RuntimeError("unable to assign temporal class-prior weight")
                target_counts[donor] -= 1.0
                target_counts[class_index] = 1.0
            correction = target_counts / sample_counts
            inclusion_weight = correction[inverse[selected]].astype(float, copy=False)

    chronological = bool(np.all(groups[1:] >= groups[:-1]))
    order = np.argsort(selected, kind="stable") if chronological else np.lexsort((selected, groups[selected]))
    selected = selected[order].astype(np.int64, copy=False)
    if inclusion_weight is not None:
        inclusion_weight = inclusion_weight[order]
    recent_mask = np.zeros(n_rows, dtype=bool)
    recent_mask[recent_rows] = True
    selected_recent = int(recent_mask[selected].sum())
    return (
        selected,
        inclusion_weight,
        {
            "mode": "temporal_multiscale_reservoir",
            "source_rows": int(n_rows),
            "sample_rows": int(len(selected)),
            "recent_rows": selected_recent,
            "history_rows": int(len(selected) - selected_recent),
            "recent_share": float(selected_recent / len(selected)),
            "class_floor": int(min_class_rows) if stratified else 0,
            "prior_reference": "temporal_reservoir" if min_class_rows else None,
            "history_band_rows": history_band_rows,
            "history_band_samples": history_band_samples,
        },
    )


def _region_rule(rid, preds):
    """Compile a leaf path into a self-contained numeric/category Horn clause.

    A categorical node consumes one base fact ``cat(row, onehot_block,
    level)`` and a finite trusted-kernel membership comparison. That is a real
    partition in the proof language, not a target statistic or an opaque
    encoded feature.
    """
    original = tuple(preds)
    canonical = canonicalize_conjunction(original)
    if canonical is None:
        # Keep the head range-restricted while eliminating an impossible path.
        first = original[0]
        if first[0] == "cat":
            cols = tuple(int(j) for j in first[1])
            return (
                (f"reg{rid}", "R"),
                [("cat", "R", cols, "V0"), ("cmp", "==", 0, 1)],
                {"features": [], "categories": (cols,)},
            )
        feature = int(first[0])
        return (
            (f"reg{rid}", "R"),
            [("feat", "R", feature, "V0"), ("cmp", "==", 0, 1)],
            {"features": [feature], "categories": ()},
        )

    body, feats, cats, variables = [], set(), [], {}
    for pred in canonical:
        if pred[0] == "cat":
            _, cols, op, levels = pred
            cols = tuple(int(j) for j in cols)
            key = ("cat", cols)
            if key not in variables:
                variables[key] = f"V{len(variables)}"
                cats.append(cols)
                body.append(("cat", "R", cols, variables[key]))
            body.append(("cmp", op, variables[key], tuple(int(level) for level in levels)))
        else:
            j, op, thr = pred
            key = ("feat", int(j))
            if key not in variables:
                variables[key] = f"V{len(variables)}"
                body.append(("feat", "R", int(j), variables[key]))
            feats.add(j)
            body.append(("cmp", op, variables[key], float(thr)))
    return ((f"reg{rid}", "R"), body, {"features": sorted(feats), "categories": tuple(cats)})


def _category_level(X, row, cols):
    block = np.asarray(X)[row, cols]
    level = int(block.argmax())
    return level if block[level] > 0.5 else -1


def _region_facts(X, rows, inputs):
    """The base facts required by a numeric/category region clause."""
    facts = []
    for row in rows:
        facts += [("feat", int(row), j, float(X[row, j])) for j in inputs["features"]]
        facts += [("cat", int(row), cols, _category_level(X, row, cols)) for cols in inputs["categories"]]
    return facts


def _leaf_regions(tree, path=()):
    """Walk numeric/category tuple trees into finite leaf-region predicates."""
    if tree[0] == "leaf":
        return [(list(path), tree[1])]
    if tree[0] == "cat":
        _, cols, levels, lt, rt = tree
        return _leaf_regions(lt, path + (("cat", cols, "in", levels),)) + _leaf_regions(
            rt, path + (("cat", cols, "not in", levels),)
        )
    _, j, t, lt, rt = tree
    return _leaf_regions(lt, path + ((j, "<=", t),)) + _leaf_regions(rt, path + ((j, ">", t),))


def _flatten_tree(tree):
    """Compile a tuple-tree into flat arrays (feat, thr, left, right, val) for vectorized batch prediction.
    Leaves have feat = -1 and hold their value in `val`; internal nodes point to child node indices."""
    feat, thr, left, right, val = [], [], [], [], []

    def rec(t):
        idx = len(feat)
        if t[0] == "leaf":
            feat.append(-1)
            thr.append(0.0)
            left.append(-1)
            right.append(-1)
            val.append(float(t[1]))
            return idx
        if t[0] == "cat":
            return None
        feat.append(t[1])
        thr.append(float(t[2]))
        left.append(-1)
        right.append(-1)
        val.append(0.0)
        li = rec(t[3])
        ri = rec(t[4])
        if li is None or ri is None:
            return None
        left[idx] = li
        right[idx] = ri
        return idx

    if rec(tree) is None:
        return None
    return (
        np.array(feat, np.int64),
        np.array(thr, float),
        np.array(left, np.int64),
        np.array(right, np.int64),
        np.array(val, float),
    )


def _flat_pred(flat, X):
    """Leaf value for every row of X through one flattened tree. Routes in numba (per-row, no per-level array
    allocations) when available — the hot path for prediction and the certificate grid evaluations — else the
    vectorized numpy fallback. Bit-identical routing either way."""
    feat, thr, left, right, val = flat
    if _HAS_NUMBA:
        return _route_flat_nb(
            np.ascontiguousarray(feat, np.int64),
            np.ascontiguousarray(thr, np.float64),
            np.ascontiguousarray(left, np.int64),
            np.ascontiguousarray(right, np.int64),
            np.ascontiguousarray(val, np.float64),
            np.ascontiguousarray(X, np.float64),
        )
    node = np.zeros(len(X), np.int64)
    while True:
        f = feat[node]
        internal = f >= 0
        if not internal.any():
            break
        rows = np.nonzero(internal)[0]
        nd = node[rows]
        go_left = X[rows, f[rows]] <= thr[nd]
        node[rows] = np.where(go_left, left[nd], right[nd])
    return val[node]


def _adaptive_checkpoint_stride(kind, stage_class, n_classes):
    """Return a fixed semantic-round checkpoint stride for one packed layout."""
    if kind in {"binary_mirrored", "multiclass_shared"}:
        return _ADAPTIVE_DEPTH_ROUNDS_PER_CHECK
    expected = np.arange(n_classes, dtype=np.int64)
    stage_class = np.asarray(stage_class, dtype=np.int64)
    if (
        n_classes > 1
        and len(stage_class) >= n_classes
        and len(stage_class) % n_classes == 0
        and np.array_equal(
            stage_class.reshape(-1, n_classes),
            np.tile(expected, (len(stage_class) // n_classes, 1)),
        )
    ):
        return _ADAPTIVE_DEPTH_ROUNDS_PER_CHECK * n_classes
    return max(_ADAPTIVE_DEPTH_ROUNDS_PER_CHECK, min(32, _ADAPTIVE_DEPTH_ROUNDS_PER_CHECK * n_classes))


def _adaptive_suffix_bounds(kind, packed, lr, n_classes):
    """Compile global remaining-leaf bounds for exact adaptive prediction.

    Each packed stage contributes to one class, a mirrored binary pair, or all
    classes through a shared vector leaf. Independent leaf extrema are loose
    but sound and need no query-time tree traversal.
    """
    feat, _thr, _left, _right, values, starts = packed[:6]
    stages = len(starts)
    stage_min = np.zeros((stages, n_classes), dtype=np.float64)
    stage_max = np.zeros_like(stage_min)
    stage_abs = np.zeros_like(stage_min)
    stage_count = np.zeros((stages, n_classes), dtype=np.int64)
    stage_class = None if kind != "flat" else np.asarray(packed[6], dtype=np.int64)

    for stage, start_value in enumerate(starts):
        start = int(start_value)
        stop = int(starts[stage + 1]) if stage + 1 < stages else len(feat)
        leaves = np.flatnonzero(np.asarray(feat[start:stop]) < 0) + start
        if not len(leaves):
            return None
        if kind == "flat":
            cls = int(stage_class[stage])
            contribution = float(lr) * np.asarray(values[leaves], dtype=np.float64)
            stage_min[stage, cls] = float(contribution.min())
            stage_max[stage, cls] = float(contribution.max())
            stage_abs[stage, cls] = float(np.abs(contribution).max())
            stage_count[stage, cls] = 1
        elif kind == "binary_mirrored":
            contribution = float(lr) * np.asarray(values[leaves], dtype=np.float64)
            low, high = float(contribution.min()), float(contribution.max())
            stage_min[stage] = (low, -high)
            stage_max[stage] = (high, -low)
            bound = float(np.abs(contribution).max())
            stage_abs[stage] = (bound, bound)
            stage_count[stage] = 1
        elif kind == "multiclass_shared":
            contribution = float(lr) * np.asarray(values[leaves], dtype=np.float64)
            stage_min[stage] = contribution.min(axis=0)
            stage_max[stage] = contribution.max(axis=0)
            stage_abs[stage] = np.abs(contribution).max(axis=0)
            stage_count[stage] = 1
        else:
            return None

    arrays = (stage_min, stage_max, stage_abs)
    if any(not np.isfinite(array).all() for array in arrays):
        return None
    suffix_min = np.zeros((stages + 1, n_classes), dtype=np.float64)
    suffix_max = np.zeros_like(suffix_min)
    suffix_abs = np.zeros_like(suffix_min)
    suffix_count = np.zeros((stages + 1, n_classes), dtype=np.int64)
    for stage in range(stages - 1, -1, -1):
        suffix_min[stage] = suffix_min[stage + 1] + stage_min[stage]
        suffix_max[stage] = suffix_max[stage + 1] + stage_max[stage]
        suffix_abs[stage] = suffix_abs[stage + 1] + stage_abs[stage]
        suffix_count[stage] = suffix_count[stage + 1] + stage_count[stage]
    return tuple(np.ascontiguousarray(array) for array in (suffix_min, suffix_max, suffix_abs, suffix_count))


def _tree_features(tree):
    """Set of feature indices a tree splits on (to skip trees that are constant over a box)."""
    fs, stack = set(), [tree]
    while stack:
        n = stack.pop()
        if n[0] == "node":
            fs.add(n[1])
            stack += [n[3], n[4]]
        elif n[0] == "cat":
            fs.update(n[1])
            stack += [n[3], n[4]]
    return fs


def _row_path(tree, x):
    """The threshold clauses fired routing x to its leaf in one tree: list of (feature, op, thr)."""
    path = []
    while tree[0] != "leaf":
        if tree[0] == "cat":
            _, cols, levels, L, R = tree
            if bool(_cat_member(np.asarray(x)[None, :], cols, levels)[0]):
                path.append(("cat", cols, "in", levels))
                tree = L
            else:
                path.append(("cat", cols, "not in", levels))
                tree = R
            continue
        _, j, thr, L, R = tree
        if x[j] <= thr:
            path.append((j, "<=", thr))
            tree = L
        else:
            path.append((j, ">", thr))
            tree = R
    return path


class _CompiledRegionGraph:
    """Static, exact factor graph for cap-bounded certified margin queries.

    A boosted tree is a small factor over the features on which it splits.  The
    fitted ensemble therefore forms a sparse feature co-occurrence graph.  We
    compile that topology once, choose a deterministic min-fill separator
    order, then contract factors at query time under the input box.  Each
    contracted table is an exact shortcut for a class-pair margin; if a table
    would exceed the cap, callers fall back to the existing interval verifier.

    This deliberately does *not* replace flat-tree prediction or the FOL
    proofs.  The original trees and clauses stay authoritative; the graph only
    removes redundant work from robustness and sufficient-reason queries.
    """

    def __init__(self, trees, flats, lr, base, n_features):
        self.lr = float(lr)
        self.base = np.asarray(base, float).copy()
        self.n_features = int(n_features)
        self.complete = all(flat is not None for flat in flats)
        self.factors = []
        predicates = {}
        adjacency = {}

        # Read scopes and atomic predicates directly from the flat trees.  This
        # preserves lazy tuple-tree materialization for ordinary prediction.
        for stage, ((cls, _tree), flat) in enumerate(zip(trees, flats, strict=False)):
            if flat is None:
                continue
            feat, thr, _left, _right, _val = flat
            internal = np.flatnonzero(feat >= 0)
            scope = tuple(int(v) for v in np.unique(feat[internal]))
            predicate_ids = []
            for node in internal:
                key = (int(feat[node]), float(thr[node]))
                predicate_ids.append(predicates.setdefault(key, len(predicates)))
            self.factors.append((int(cls), flat, scope, tuple(predicate_ids), int(stage)))
            for j in scope:
                adjacency.setdefault(j, set())
            for pos, left_feature in enumerate(scope):
                adjacency[left_feature].update(scope[:pos])
                adjacency[left_feature].update(scope[pos + 1 :])

        self.predicate_count = len(predicates)
        self.factor_count = len(self.factors)
        self.thresholds = {}
        for j, threshold in predicates:
            self.thresholds.setdefault(j, []).append(threshold)
        self.thresholds = {j: np.array(sorted(set(values)), float) for j, values in self.thresholds.items()}
        if not self.complete:
            self.active = False
            self.skip_reason = "categorical_or_nonflat_tree"
            self.elimination_order = ()
        elif len(adjacency) > _COMPILED_REGION_MAX_FEATURES:
            self.active = False
            self.skip_reason = "feature_budget"
            self.elimination_order = ()
        else:
            self.active = True
            self.skip_reason = None
            self.elimination_order = self._min_fill_order(adjacency)

    @staticmethod
    def _min_fill_order(adjacency):
        """Deterministic separator order for factor contraction.

        Nested dissection is ideal for a large road graph.  Our graphs are
        small feature hypergraphs, where deterministic min-fill gives the same
        useful property: contract low-connectivity features first and leave
        dense separators until late.
        """
        graph = {int(node): set(neighbors) for node, neighbors in adjacency.items()}
        order = []
        while graph:

            def key(node):
                neighbors = sorted(graph[node])
                fill = sum(
                    right not in graph[left]
                    for pos, left in enumerate(neighbors)
                    for right in neighbors[pos + 1 :]
                )
                return (fill, len(neighbors), node)

            node = min(graph, key=key)
            neighbors = sorted(graph[node])
            for pos, left in enumerate(neighbors):
                for right in neighbors[pos + 1 :]:
                    graph[left].add(right)
                    graph[right].add(left)
            for neighbor in neighbors:
                graph[neighbor].discard(node)
            del graph[node]
            order.append(node)
        return tuple(order)

    @staticmethod
    def _base_point(lo, hi):
        point = np.empty(len(lo), float)
        for j in range(len(point)):
            if np.isfinite(lo[j]) and np.isfinite(hi[j]):
                point[j] = (lo[j] + hi[j]) / 2.0
            elif np.isfinite(hi[j]):
                point[j] = hi[j] - 1.0
            elif np.isfinite(lo[j]):
                point[j] = lo[j] + 1.0
            else:
                point[j] = 0.0
        return point

    @staticmethod
    def _state_count(scope, domains):
        count = 1
        for feature in scope:
            count *= len(domains[feature])
        return int(count)

    def _query_domains(self, lo, hi, features):
        point = self._base_point(lo, hi)
        domains = {}
        for j in sorted(features):
            thresholds = self.thresholds[j]
            inside = thresholds[(thresholds > lo[j]) & (thresholds < hi[j])]
            if len(inside):
                lower = float(lo[j]) if np.isfinite(lo[j]) else float(inside[0] - 1.0)
                upper = float(hi[j]) if np.isfinite(hi[j]) else float(inside[-1] + 1.0)
                bounds = np.concatenate(([lower], inside, [upper]))
                reps = (bounds[:-1] + bounds[1:]) / 2.0
            else:
                reps = np.array([point[j]], float)
            domains[j] = reps
            point[j] = reps[0]
        return point, domains

    def _factor_message(self, factor, sign, point, domains, cap):
        _cls, flat, scope, _predicate_ids, _stage = factor
        states = self._state_count(scope, domains)
        if states > cap:
            return None
        if not scope:
            value = float(sign * self.lr * _flat_pred(flat, point[None, :])[0])
            return (), np.asarray(value)

        shape = tuple(len(domains[j]) for j in scope)
        grid = np.tile(point, (states, 1))
        for axis, feature in enumerate(scope):
            repeat = int(np.prod(shape[axis + 1 :], dtype=int)) if axis + 1 < len(shape) else 1
            tile = int(np.prod(shape[:axis], dtype=int)) if axis else 1
            grid[:, feature] = np.tile(np.repeat(domains[feature], repeat), tile)
        values = sign * self.lr * _flat_pred(flat, grid).reshape(shape)
        return scope, values

    def _contract_min(self, messages, domains, cap):
        """Eliminate feature factors with min-sum dynamic programming."""
        constant = 0.0
        work = []
        for scope, values in messages:
            if scope:
                work.append((scope, values))
            else:
                constant += float(values)

        for feature in self.elimination_order:
            touching_idx = [idx for idx, (scope, _values) in enumerate(work) if feature in scope]
            if not touching_idx:
                continue
            touching = [work[idx] for idx in touching_idx]
            union = tuple(sorted({item for scope, _values in touching for item in scope}))
            if self._state_count(union, domains) > cap:
                return None
            shape = tuple(len(domains[item]) for item in union)
            total = np.zeros(shape, float)
            positions = {item: axis for axis, item in enumerate(union)}
            for scope, values in touching:
                reshape = [1] * len(union)
                for axis, item in enumerate(scope):
                    reshape[positions[item]] = values.shape[axis]
                total += values.reshape(reshape)
            axis = positions[feature]
            shortcut = total.min(axis=axis)
            remaining_scope = tuple(item for item in union if item != feature)
            touched_set = set(touching_idx)
            work = [item for idx, item in enumerate(work) if idx not in touched_set]
            if remaining_scope:
                work.append((remaining_scope, shortcut))
            else:
                constant += float(shortcut)

        # The order is compiled from every numeric factor.  This guard keeps a
        # future partial compiler conservative rather than silently ignoring a
        # variable it did not know how to contract.
        if any(scope for scope, _values in work):
            return None
        return constant + sum(float(values) for _scope, values in work)

    def margin_lower(self, lo, hi, winner, challenger, cap=4096):
        """Exact lower bound of ``score[winner] - score[challenger]`` or None.

        A missing result means the graph would exceed its fixed work cap or a
        categorical/affine tree is present.  The caller must then use the
        established sound interval verifier.
        """
        if not self.active:
            return None
        winner, challenger = int(winner), int(challenger)
        selected = [
            (factor, 1.0 if cls == winner else -1.0)
            for factor in self.factors
            for cls in [factor[0]]
            if cls == winner or cls == challenger
        ]
        features = {feature for factor, _sign in selected for feature in factor[2]}
        point, domains = self._query_domains(np.asarray(lo, float), np.asarray(hi, float), features)
        messages = []
        for factor, sign in selected:
            message = self._factor_message(factor, sign, point, domains, cap)
            if message is None:
                return None
            messages.append(message)
        contracted = self._contract_min(messages, domains, cap)
        if contracted is None:
            return None
        # A one-ulp downward nudge makes the optimized answer conservative at
        # the floating-point boundary used by the existing certificate code.
        value = float(self.base[winner] - self.base[challenger] + contracted)
        return float(np.nextafter(value, -np.inf))

    def report(self):
        return {
            "active": bool(self.active),
            "factors": int(self.factor_count),
            "atomic_predicates": int(self.predicate_count),
            "separator_features": int(len(self.elimination_order)),
            "skip_reason": self.skip_reason,
        }


def _leaf_affine_interval(node, lo, hi):
    """EXACT [min,max] of a leaf value over the box {lo <= x <= hi}. Constant leaf ('leaf', b) -> (b, b).
    Linear leaf ('leaf', b, coef, clamp_lo, clamp_hi) -> b + Σ_j coef_j·clip(x_j, clamp_lo_j, clamp_hi_j);
    since clip is monotone the per-coordinate range is [clip(lo_j), clip(hi_j)] (ordered by sign of coef_j),
    and clip pins it to the FINITE leaf clamp range so the interval is bounded even when a box bound is ±inf.
    coef is nonzero only on path-constrained features (cert-safe)."""
    if len(node) == 2:
        return (node[1], node[1])
    _, b, coef, clo, chi = node
    nz = coef != 0
    if not nz.any():
        return (b, b)
    c = coef[nz]
    lo_c = np.minimum(np.maximum(lo[nz], clo[nz]), chi[nz])  # clip(box_lo) into the finite leaf clamp range
    hi_c = np.minimum(np.maximum(hi[nz], clo[nz]), chi[nz])
    pos = c >= 0
    cmin = float(np.where(pos, c * lo_c, c * hi_c).sum())
    cmax = float(np.where(pos, c * hi_c, c * lo_c).sum())
    return (b + cmin, b + cmax)


def _tree_interval(tree, lo, hi):
    """EXACT [min, max] leaf value reachable in one tree over the axis-aligned box {lo[j] < x_j <= hi[j]}.
    At a split (j, thr): if the whole box is one side of thr, recurse that child; if it straddles thr, BOTH
    children are reachable, so union their ranges. Unconstrained features (lo=-inf, hi=+inf) always straddle.
    Leaves may be constant or affine (linear leaves) — `_leaf_affine_interval` bounds either. Summed over the
    additive stages this bounds the prediction over the box — the primitive for certified intervals/reasons."""
    if tree[0] == "leaf":
        return _leaf_affine_interval(tree, lo, hi)
    if tree[0] == "cat":
        # An axis-aligned numeric box cannot assert an atomic one-hot level;
        # both finite category branches remain reachable. The union is sound.
        _, _, _, lt, rt = tree
        la, lb = _tree_interval(lt, lo, hi)
        ra, rb = _tree_interval(rt, lo, hi)
        return (min(la, ra), max(lb, rb))
    _, j, thr, lt, rt = tree
    if hi[j] <= thr:  # box entirely on the '<=' side
        return _tree_interval(lt, lo, hi)
    if lo[j] >= thr:  # box entirely on the '>' side
        return _tree_interval(rt, lo, hi)
    la, lb = _tree_interval(lt, lo, hi)
    ra, rb = _tree_interval(rt, lo, hi)
    return (min(la, ra), max(lb, rb))


def _recourse_search(pred_fn, goal, x, thr, scale, meta, max_options):
    """Group-aware minimal-change search shared by classification & regression recourse. Numeric features are
    moved across tree thresholds (both sides); a CATEGORICAL (one-hot group in `meta`) is switched to each
    other LEVEL atomically (chosen dummy=1, siblings=0) so the candidate is always a VALID input — never a
    fractional or multi-level dummy vector. Returns options sorted by cost (kernel-verification is the caller's
    job). `meta` None => every column its own numeric group (raw ndarray)."""
    groups = (
        meta
        if meta is not None
        else [{"kind": "num", "label": int(j), "cols": [int(j)]} for j in sorted(thr)]
    )
    opts = []
    for g in groups:
        cols, kind = g["cols"], g.get("kind", "num")
        if kind in ("isna", "encoded_category", "derived", "byte_evidence"):
            continue
        if kind == "onehot":
            cur = next((c for c in cols if x[c] > 0.5), None)
            for ci, lvl in zip(cols, g["levels"], strict=False):
                if ci == cur:
                    continue
                xp = x.copy()
                xp[cols] = 0.0
                xp[ci] = 1.0
                out = pred_fn(xp[None, :])[0]
                if goal(out):
                    frm = g["levels"][cols.index(cur)] if cur is not None else None
                    opts.append(
                        {
                            "kind": "categorical",
                            "label": g["label"],
                            "from_level": frm,
                            "to_level": lvl,
                            "cost": 1.0,
                            "outcome": out,
                            "cols": list(cols),
                            "to_col": ci,
                        }
                    )
                    break  # one valid switch per categorical is enough
        else:  # numeric / frequency-encoded single column
            j = cols[0]
            ths = thr.get(j)
            if ths is None or len(ths) == 0:
                continue
            cands, tags = [], []
            for th in ths:
                cands += [float(th), float(np.nextafter(th, th + 1.0))]
                tags += [(float(th), "≤"), (float(th), ">")]
            cands = np.array(cands)
            Xc = np.tile(x, (len(cands), 1))
            Xc[:, j] = cands
            outs = pred_fn(Xc)
            ok = np.array([goal(o) for o in outs])
            if not ok.any():
                continue
            ch = np.abs(cands - x[j])
            ch[~ok] = np.inf
            kk = int(ch.argmin())
            th_k, op = tags[kk]
            opts.append(
                {
                    "kind": "numeric",
                    "label": g["label"],
                    "col": j,
                    "from": round(float(x[j]), 4),
                    "op": op,
                    "threshold": round(th_k, 4),
                    "to": float(cands[kk]),
                    "cost": round(float(ch[kk]) / scale[j], 3),
                    "outcome": outs[kk],
                }
            )
    opts.sort(key=lambda o: o["cost"])
    return opts[:max_options]


def _group_universe(groups, universe):
    """Partition the split-feature `universe` into deletion groups. `groups` (list of column-index lists, e.g.
    the one-hot dummies of one categorical) are kept intact — restricted to columns actually in `universe`;
    any split feature not covered by a group becomes its own singleton. None => every feature its own group."""
    uni = set(universe)
    if groups is None:
        return [[j] for j in universe]
    gs, covered = [], set()
    for g in groups:
        cols = [j for j in g if j in uni]
        if cols:
            gs.append(cols)
            covered.update(cols)
    gs += [[j] for j in universe if j not in covered]
    return gs


def _smallest_reason_box(initial_box, groups, valid, quality, feature_use):
    """Keep the smallest box found by bounded deterministic deletion orders.

    A single greedy order is subset-minimal but can retain many weak features
    simply because a different feature was tested first. Wide tables receive
    two additional orders based on certified-tree feature usage. The incumbent
    order is always a candidate, so this optimization cannot enlarge a reason.
    """
    groups = [tuple(group) for group in groups]
    orders = [groups]
    if 16 <= len(groups) <= 128:
        usage_order = sorted(
            groups,
            key=lambda group: (
                sum(feature_use.get(column, 0) for column in group),
                len(group),
                group,
            ),
        )
        orders.extend((usage_order, list(reversed(usage_order))))

    candidates = []
    seen = set()
    for order in orders:
        signature = tuple(order)
        if signature in seen:
            continue
        seen.add(signature)
        box = dict(initial_box)
        for group in order:
            saved = {column: box.pop(column) for column in group if column in box}
            if saved and not valid(quality(box)):
                box.update(saved)
        retained_groups = sum(any(column in box for column in group) for group in groups)
        candidates.append(
            (
                retained_groups,
                len(box),
                -float(quality(box)),
                tuple(sorted(box)),
                box,
            )
        )
    return min(candidates, key=lambda candidate: candidate[:-1])[-1]


class AdditiveCertifiedRegressor:
    """Additive certified regressor built ONLY on our own primitives — `reason_boost_2nd` (our 2nd-order
    residual tree-boosting, `trees.py`) as the region proposer, the FOLKernel as verifier. No external ML
    library: split search, boosting, and prediction are all ours. Each leaf is a threshold region the kernel
    verifies; prediction = base + Σ lr·leaf_value, so the whole additive prediction is a certified sum."""

    def __init__(
        self,
        rounds=2000,
        lr=0.03,
        depth=6,
        leaf=30,
        subsample=0.8,
        colsample=0.6,
        lam=1.0,
        nbins=128,
        huber=0.95,
        mono=None,
        allowed=None,
        holdout=0.15,
        patience=60,
        refit=True,
        seed=0,
        linear_leaf=False,
        lam_lin=10.0,
        fit_cap=None,
    ):
        # 2nd-order (Newton) residual boosting: lam = L2 leaf regularization; colsample->mf, subsample->sub;
        # a held-out slice early-stops to pick tree count, then refit on 100% of the data (rounds is a max).
        # huber=q uses a robust objective (gradient clipped at the q-quantile of |residual| each round) — a
        # principled, task-agnostic replacement for winsorizing the target; None = plain squared loss.
        # refit=False skips the refit-on-full pass (~2x faster) — used for INTERNAL decision fits (tuning,
        # invariant-verification, calibration) that only read held-out RMSE; the final model keeps refit=True.
        self.rounds, self.lr, self.depth, self.leaf, self.subsample, self.colsample = (
            rounds,
            lr,
            depth,
            leaf,
            subsample,
            colsample,
        )
        self.lam, self.nbins, self.huber, self.mono, self.allowed = lam, nbins, huber, mono, allowed
        self.holdout, self.patience, self.refit, self.seed = holdout, patience, refit, seed
        self.linear_leaf, self.lam_lin = linear_leaf, lam_lin
        self.fit_cap = fit_cap

    def fit(self, X, y, validation_groups=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        groups = None if validation_groups is None else np.asarray(validation_groups, dtype=np.int64)
        if groups is not None and (groups.ndim != 1 or len(groups) != len(y)):
            raise ValueError("validation_groups must have one value per regression row")
        if groups is None:
            fit_rows = _fit_rows(y, self.fit_cap, self.seed * 7 + 3, stratified=False)
            self.fit_sampling_ = {
                "mode": "exchangeable_full" if fit_rows is None else "exchangeable_reservoir",
                "source_rows": int(len(y)),
                "sample_rows": int(len(y) if fit_rows is None else len(fit_rows)),
            }
        else:
            fit_rows, _weight, self.fit_sampling_ = _temporal_fit_sample(
                y,
                groups,
                self.fit_cap,
                self.seed * 7 + 3,
                stratified=False,
            )
        self.fit_rows_ = fit_rows
        if fit_rows is not None:
            X = X[fit_rows]
            y = y[fit_rows]
            if groups is not None:
                groups = groups[fit_rows]
        # per-feature robust scale for input-noise bands; constant features -> unit scale
        quartiles = np.nanpercentile(X, (25, 75), axis=0)
        iqr = quartiles[1] - quartiles[0]
        self.scale_ = np.where(iqr > 0, iqr, 1.0)
        mono = None
        if self.mono is not None:  # {col: +1/-1} or array -> per-feature vector
            mono = np.zeros(X.shape[1], int)
            for jj, d in self.mono.items() if isinstance(self.mono, dict) else enumerate(self.mono):
                mono[jj] = d
        self.model_ = reason_boost_2nd(
            X,
            y,
            holdout=self.holdout,
            lr=self.lr,
            n_terms=self.rounds,
            depth=self.depth,
            min_leaf=self.leaf,
            sub=self.subsample,
            mf=self.colsample,
            lam=self.lam,
            nbins=self.nbins,
            huber=self.huber,
            mono=mono,
            allowed=self.allowed,
            patience=self.patience,
            refit=self.refit,
            seed=self.seed,
            linear_leaf=self.linear_leaf,
            lam_lin=self.lam_lin,
            validation_groups=groups,
        )
        self.base_, self.lr_, self.trees_, self._flat_cache, self.linear_ = self.model_
        self._serving_forest_cache = None
        self._serving_forest_ready = False
        return self

    def _serving_forest(self):
        """Pack numeric proof trees into one fused prediction kernel lazily."""
        if not self._serving_forest_ready:
            stages = np.zeros(len(self.trees_), dtype=np.int64)
            if self.linear_:
                packed = _pack_affine_forest(self.trees_, stages)
                self._serving_forest_cache = None if packed is None else ("affine", packed)
            else:
                packed = _pack_flat_forest(self._flat_cache, stages)
                self._serving_forest_cache = None if packed is None else ("flat", packed)
            self._serving_forest_ready = True
        return self._serving_forest_cache

    def predict(self, X):
        X = np.asarray(X, float)
        if _HAS_NUMBA:
            forest = self._serving_forest()
            if forest is not None:
                kind, packed = forest
                Xc = np.ascontiguousarray(X, np.float64)
                base = np.ascontiguousarray(np.array([self.base_]), np.float64)
                if kind == "affine":
                    feat, thr, left, right, val, starts, classes, lin_feat, lin_coef, lin_lo, lin_hi = packed
                    return _forest_scores_affine_nb(
                        feat,
                        thr,
                        left,
                        right,
                        val,
                        starts,
                        classes,
                        lin_feat,
                        lin_coef,
                        lin_lo,
                        lin_hi,
                        base,
                        self.lr_,
                        Xc,
                    )[:, 0]
                feat, thr, left, right, val, starts, classes = packed
                return _forest_scores_flat_nb(
                    feat, thr, left, right, val, starts, classes, base, self.lr_, Xc
                )[:, 0]
        return boost_predict(self.model_, X)

    # ---- soundness: the FOLKernel reproduces each stage's region MEMBERSHIP exactly ----
    def _path_mask(self, preds, X):
        m = np.ones(len(X), bool)
        for pred in preds:
            if pred[0] == "cat":
                _, cols, op, levels = pred
                member = _cat_member(X, cols, levels)
                m &= member if op == "in" else ~member
            else:
                j, op, thr = pred
                m &= (X[:, j] <= thr) if op == "<=" else (X[:, j] > thr)
        return m

    def kernel_certify(self, X, n_trees=40, sample=120):
        """For each checked stage, the FOLKernel's region membership (closure over the leaf threshold clause)
        must equal the numpy evaluation of that same clause, for EVERY region on the sample. Given exact
        membership the leaf value is determined, so this is the soundness that matters; composes to all rounds."""
        X = np.asarray(X, float)
        idx = np.random.default_rng(self.seed).choice(len(X), size=min(sample, len(X)), replace=False)
        Xs = X[idx]
        n = len(Xs)
        agrees = []
        covered_ok = 0.0
        checks = 0
        for tree in self.trees_[:n_trees]:
            covered = np.zeros(n, bool)
            for rid, (preds, _) in enumerate(_leaf_regions(tree)):
                if not preds:
                    covered[:] = True
                    continue  # a stump leaf covers everything trivially
                head, body, inputs = _region_rule(rid, preds)
                facts = _region_facts(Xs, range(n), inputs)
                fired, _ = FOLKernel([(head, body)]).closure(facts)
                k_mask = np.zeros(n, bool)
                for tp in fired:
                    if tp[0] == head[0]:
                        k_mask[tp[1]] = True
                agrees.append(float((k_mask == self._path_mask(preds, Xs)).mean()))
                covered |= k_mask
            covered_ok += covered.mean()
            checks += 1
        agree = float(np.allclose(np.mean(agrees), 1.0)) if agrees else 1.0
        return {
            "membership_reproduced": agree,
            "region_membership_recovered": float(covered_ok / max(checks, 1)),
            "mean_clause_agreement": float(np.mean(agrees)) if agrees else 1.0,
            "stages_checked": checks,
            "of_total": len(self.trees_),
            "prediction_reproduced": agree,
        }

    @staticmethod
    def _row_path(tree, x):
        """The threshold clauses fired routing x to its leaf in one stage: list of (feature, op, thr)."""
        path = []
        while tree[0] == "node":
            _, j, thr, L, R = tree
            if x[j] <= thr:
                path.append((j, "<=", thr))
                tree = L
            else:
                path.append((j, ">", thr))
                tree = R
        return path

    def stability_box(self, X, row, samples=12):
        """EXACT robustness certificate — the axis-aligned box over which this row's prediction is PROVABLY
        constant. Intersecting the threshold clauses fired across ALL boosted stages gives, per feature, an
        interval (lo, hi]; any input inside the box takes the same leaf in every stage, so the additive
        prediction is identical. Features on no fired path don't gate the prediction at all (unconstrained —
        the prediction is invariant to them entirely). The FOLKernel verifies box membership by closure over
        the conjoined clause; a sample of interior points (constrained features inside the box, unconstrained
        features perturbed freely) confirms the prediction is unchanged. This is an EXACT guarantee, not the
        conformal (statistical) one — the two complement each other: how far inputs can move with NO change
        (this), and how wrong the value might be (the bound). No user knobs. Returns per-feature radius, the
        certified-constant prediction, the soundness check, and the kernel proof."""
        X = np.asarray(X, float)
        x = X[row].copy()
        d = X.shape[1]
        lo, hi = {}, {}
        for tree in self.trees_:
            for j, op, thr in self._row_path(tree, x):
                if op == "<=":
                    hi[j] = min(hi.get(j, np.inf), thr)
                else:
                    lo[j] = max(lo.get(j, -np.inf), thr)
        feats = sorted(set(lo) | set(hi))
        pred = float(boost_predict(self.model_, x[None, :])[0])
        linear = getattr(self, "linear_", False)
        box = {}
        for j in feats:
            L, H = lo.get(j, -np.inf), hi.get(j, np.inf)
            box[int(j)] = {
                "lo": (None if L == -np.inf else round(L, 4)),
                "hi": (None if H == np.inf else round(H, 4)),
                "down": (float("inf") if L == -np.inf else round(float(x[j] - L), 4)),
                "up": (float("inf") if H == np.inf else round(float(H - x[j]), 4)),
            }
        # Constant leaves -> the prediction is IDENTICAL over the box (ε=0). Linear leaves -> it is affine over
        # the box, so it is provably within a finite band [a,b] (ε=(b-a)/2); the leaf clamps keep it bounded.
        band_a = band_b = pred
        if linear:
            lo_vec, hi_vec = np.full(d, -np.inf), np.full(d, np.inf)
            for j in lo:
                lo_vec[j] = lo[j]
            for j in hi:
                hi_vec[j] = hi[j]
            band_a, band_b = self._interval(lo_vec, hi_vec)
        rng = np.random.default_rng(self.seed)
        fset = set(feats)
        # Build ALL soundness samples first (same rng draw order as the per-sample loop -> identical points),
        # then score them in ONE batched boost_predict instead of `samples` separate all-tree passes.
        XP = np.tile(x, (samples, 1))
        for si in range(samples):
            for j in range(d):
                if j in fset:  # constrained: sample STRICTLY interior
                    L, H = lo.get(j, -np.inf), hi.get(j, np.inf)  # (open lower, closed upper)
                    L = x[j] - abs(x[j]) - 1.0 if L == -np.inf else L
                    H = x[j] + abs(x[j]) + 1.0 if H == np.inf else H
                    span = H - L
                    if span <= 1e-9 * (abs(H) + 1.0):  # degenerate width -> float-unsafe, keep x
                        continue
                    XP[si, j] = np.clip(
                        L + span * rng.uniform(0.25, 0.75),  # 25–75% + clamp off both edges
                        np.nextafter(L, H),
                        H,
                    )
                else:  # unconstrained: perturb (mustn't matter)
                    XP[si, j] = x[j] + rng.normal() * (0.1 * abs(x[j]) + 1.0)
        sp = boost_predict(self.model_, XP)
        if linear:  # soundness: interior points fall within the certified band
            tol = 1e-6 * (abs(band_a) + abs(band_b) + 1.0)
            ok = int(np.sum((sp >= band_a - tol) & (sp <= band_b + tol)))
        else:  # soundness: interior points give the identical prediction
            ok = int(np.sum(np.abs(sp - pred) < 1e-6))
        preds = [(j, ">", lo[j]) for j in lo] + [(j, "<=", hi[j]) for j in hi]
        proof = None
        if preds:
            head, body, inputs = _region_rule("box", preds)
            facts = _region_facts(X, [row], inputs)
            fired, prov = FOLKernel([(head, body)]).closure(facts)
            if any(t[0] == head[0] for t in fired):
                proof = FOLKernel([(head, body)]).proof((head[0], row), prov)
        out = {
            "prediction": round(pred, 2),
            "n_constrained_features": len(feats),
            "box": box,
            "proof": proof,
        }
        if linear:  # provably WITHIN ±ε over the box (affine leaves)
            out["certified_band"] = [round(float(band_a), 2), round(float(band_b), 2)]
            out["epsilon"] = round(float((band_b - band_a) / 2.0), 3)
            out["interior_samples_within"] = f"{ok}/{samples}"
        else:  # provably CONSTANT over the box (ε=0)
            out["interior_samples_constant"] = f"{ok}/{samples}"
        return out

    def _thresholds(self):
        """Per-feature sorted split thresholds across the whole ensemble (cached)."""
        if not hasattr(self, "_thr_cache"):
            thr, stack = {}, list(self.trees_)
            while stack:
                t = stack.pop()
                if t[0] == "node":
                    _, j, th, lt, rt = t
                    thr.setdefault(j, set()).add(float(th))
                    stack += [lt, rt]
                elif t[0] == "cat":
                    stack += [t[3], t[4]]
            self._thr_cache = {j: np.array(sorted(s)) for j, s in thr.items()}
        return self._thr_cache

    def _interval_ibp(self, lo, hi):
        """SOUND but loose [min, max] by independent per-tree interval propagation (the cap fallback). Uses the
        compiled flat-tree interval kernel when numba is present (the certificate hot path); falls back to the
        pure-Python tuple recursion otherwise. Identical result either way."""
        smin = smax = 0.0
        if getattr(self, "linear_", False):  # affine leaves -> Python interval (flats hold only intercepts)
            for t in self.trees_:
                a, b = _tree_interval(t, lo, hi)
                smin += a
                smax += b
        elif _HAS_NUMBA:
            lo = np.ascontiguousarray(lo, np.float64)
            hi = np.ascontiguousarray(hi, np.float64)
            for feat, thr, left, right, val in self._flats():
                a, b = _tree_interval_flat_nb(feat, thr, left, right, val, lo, hi)
                smin += a
                smax += b
        else:
            for t in self.trees_:
                a, b = _tree_interval(t, lo, hi)
                smin += a
                smax += b
        return self.base_ + self.lr_ * smin, self.base_ + self.lr_ * smax

    def _interval(self, lo, hi, cap=4096):
        """EXACT [min, max] of the additive prediction over the box {lo[j] < x_j <= hi[j]} when tractable: the
        ensemble changes only where a split threshold lies inside the box, so we enumerate the sub-cell grid of
        those ACTIVE features and evaluate exactly — tight, not the loose independent-per-tree sum. Falls back
        to the sound IBP bound if the grid exceeds `cap`. (Mirrors the classifier's `_score_interval`.)"""
        if getattr(self, "linear_", False):  # affine leaves vary WITHIN a cell -> the sound per-tree affine
            return self._interval_ibp(lo, hi)  # IBP bound (looser but exact-sound); no constant-cell shortcut
        thr = self._thresholds()
        active, reps, combos = [], [], 1
        for j, ts in thr.items():
            inside = ts[(ts > lo[j]) & (ts < hi[j])]
            if len(inside) == 0:
                continue
            flo = inside[0] - 1.0 if not np.isfinite(lo[j]) else lo[j]
            fhi = inside[-1] + 1.0 if not np.isfinite(hi[j]) else hi[j]
            bounds = np.concatenate(([flo], inside, [fhi]))
            active.append(j)
            reps.append((bounds[:-1] + bounds[1:]) / 2.0)
            combos *= len(reps[-1])
            if combos > cap:
                return self._interval_ibp(lo, hi)
        base_pt = np.empty(len(lo))
        for j in range(len(lo)):
            if np.isfinite(lo[j]) and np.isfinite(hi[j]):
                base_pt[j] = (lo[j] + hi[j]) / 2.0
            elif np.isfinite(hi[j]):
                base_pt[j] = hi[j] - 1.0
            elif np.isfinite(lo[j]):
                base_pt[j] = lo[j] + 1.0
            else:
                base_pt[j] = 0.0
        if not active:  # ensemble constant over the box -> one exact evaluation
            v = float(
                self.base_ + self.lr_ * sum(float(_flat_pred(f, base_pt[None, :])[0]) for f in self._flats())
            )
            return v, v
        import itertools

        grid = np.tile(base_pt, (combos, 1))
        for i, combo in enumerate(itertools.product(*reps)):
            for jj, v in zip(active, combo, strict=False):
                grid[i, jj] = v
        # Only trees splitting on an ACTIVE feature vary over the grid; sum the rest once at base_pt.
        act = set(active)
        tf = self._tree_feats()
        flats = self._flats()
        const = sum(
            float(_flat_pred(flats[ti], base_pt[None, :])[0])
            for ti in range(len(self.trees_))
            if not (tf[ti] & act)
        )
        s = np.full(combos, self.base_ + self.lr_ * const)
        for ti in range(len(self.trees_)):
            if tf[ti] & act:
                s += self.lr_ * _flat_pred(flats[ti], grid)
        return float(s.min()), float(s.max())

    def _tree_feats(self):
        """Per-tree split-feature sets (cached) — lets `_interval` skip trees constant over the box."""
        if not hasattr(self, "_tf_cache"):
            self._tf_cache = [_tree_features(t) for t in self.trees_]
        return self._tf_cache

    def _flats(self):
        """Flattened (array) form of every tree, cached — for vectorized batch prediction."""
        if not hasattr(self, "_flat_cache"):
            self._flat_cache = [_flatten_tree(t) for t in self.trees_]
        return self._flat_cache

    def predict_interval(self, X, row, rel=0.1, delta=None):
        """Certified output band under BOUNDED INPUT UNCERTAINTY: if every feature may vary by ±delta_j, the
        prediction is GUARANTEED to stay in the returned interval (exact interval propagation over the additive
        stages — no sampling). delta defaults to rel · per-feature IQR (a data-driven 'measurement
        granularity', so it is knob-free by default); pass delta (scalar or per-feature array) to use known
        sensor precision. Unlike the conformal bound (statistical, over the label), this is an EXACT bound
        over the model's response to input error."""
        X = np.asarray(X, float)
        x = X[row]
        d = X.shape[1]
        if delta is None:
            dvec = rel * self.scale_
        else:
            dvec = np.full(d, float(delta)) if np.isscalar(delta) else np.asarray(delta, float)
        a, b = self._interval(x - dvec, x + dvec)
        return {
            "prediction": round(float(boost_predict(self.model_, x[None, :])[0]), 2),
            "band": [round(a, 2), round(b, 2)],
            "half_width": round(0.5 * (b - a), 2),
        }

    def recourse(self, X, row, target, max_options=3, meta=None):
        """Certified COUNTERFACTUAL RECOURSE: the minimal change that brings the prediction to ≤ target.
        Numeric features move across tree thresholds; a categorical (one-hot group in `meta`) is switched to
        another LEVEL atomically — so every candidate is a VALID input. Each option's counterfactual is
        FOLKernel-checked (it reaches the region that yields the new prediction). A change on a
        certified-monotone feature, in the prediction-lowering direction, is GUARANTEED to hold (flagged
        `monotone_certified`). `reachable` is False when no single-feature change reaches the target."""
        X = np.asarray(X, float)
        x = X[row]
        pred = float(boost_predict(self.model_, x[None, :])[0])
        if pred <= target:
            return {
                "prediction": round(pred, 2),
                "target": round(float(target), 2),
                "satisfied": True,
                "reachable": True,
                "options": [],
            }
        mono = self.mono if isinstance(self.mono, dict) else None
        opts = _recourse_search(
            lambda A: boost_predict(self.model_, A),
            lambda v: v <= target,
            x,
            self._thresholds(),
            self.scale_,
            meta,
            max_options,
        )
        for o in opts:  # kernel-verify each counterfactual reproduces its pred
            xp = x.copy()
            if o["kind"] == "categorical":
                xp[o["cols"]] = 0.0
                xp[o["to_col"]] = 1.0
                o["monotone_certified"] = False
            else:
                j = o["col"]
                xp[j] = o["to"]
                o["monotone_certified"] = bool(
                    mono and j in mono and np.sign(o["to"] - x[j]) == -np.sign(mono[j])
                )
            o["new_pred"] = round(float(o.pop("outcome")), 2)
            tol = max(
                1e-2, 1e-3 * abs(o["new_pred"])
            )  # scale-relative (was absolute 1.0); tolerates rounding
            o["kernel_reproduced"] = bool(abs(self.proof(xp[None, :], 0)["prediction"] - o["new_pred"]) < tol)
            o.pop("cols", None)
            o.pop("to_col", None)
        return {
            "prediction": round(pred, 2),
            "target": round(float(target), 2),
            "satisfied": False,
            "reachable": bool(opts),
            "options": opts,
        }

    def sufficient_reason(self, X, row, eps, groups=None, widen=None):
        """Minimal certified SUFFICIENT REASON (abduction): the smallest subset of threshold conditions that
        ALONE forces the prediction into a band no wider than 2·eps — every other feature free and provably
        irrelevant to that guarantee. `groups` (column-index lists) are dropped/kept ATOMICALLY (e.g. one-hot
        dummies of one categorical). ALWAYS CERTIFIED: we start from the row's SUB-CELL box on every split
        feature (the widest box crossing no threshold), which pins every tree to the row's leaf so the band
        width is 0 (≤ 2·eps by construction); we then greedily DROP features while the band stays ≤ 2·eps, and
        finally WIDEN each surviving bound outward as far as the band allows. The surviving conjunction is a
        Horn clause the FOLKernel verifies the row satisfies; the band [a,b] is exact (interval propagation).
        eps defaults (via the base) to the row's own conformal error bound, so the reason explains the answer
        to the same precision the model guarantees."""
        X = np.asarray(X, float)
        x = X[row]
        d = X.shape[1]
        thr = self._thresholds()  # per-feature split thresholds — cached once, not rebuilt every call
        universe = sorted(thr)
        sub_lo, sub_hi = {}, {}
        for j, ts in thr.items():
            below = ts[ts < x[j]]
            atabove = ts[ts >= x[j]]
            sub_lo[j] = float(below[-1]) if len(below) else -np.inf
            sub_hi[j] = float(atabove[0]) if len(atabove) else np.inf
        pred = float(boost_predict(self.model_, x[None, :])[0])

        def band(box):  # box: {j: (lo, hi)}; features not in box are free
            lo = np.full(d, -np.inf)
            hi = np.full(d, np.inf)
            for j, (a, b) in box.items():
                lo[j], hi[j] = a, b
            return self._interval(lo, hi)

        box = {j: (sub_lo[j], sub_hi[j]) for j in universe}  # band width 0 by construction (trees pinned)
        gs = _group_universe(groups, universe)
        feature_use = {}
        for features in self._tree_feats():
            for feature in features:
                feature_use[feature] = feature_use.get(feature, 0) + 1

        def band_slack(candidate):
            lower, upper = band(candidate)
            return 2 * eps - (upper - lower)

        box = _smallest_reason_box(
            box,
            gs,
            valid=lambda slack: slack >= -1e-12,
            quality=band_slack,
            feature_use=feature_use,
        )
        for j in list(box):  # WIDEN: relax each numeric bound to +-inf where the band stays ≤ 2·eps
            if widen is not None and j not in widen:
                continue
            a, b = box[j]
            lo_a, lo_b = band({**box, j: (-np.inf, b)})
            if lo_b - lo_a <= 2 * eps:
                a = -np.inf
            hi_a, hi_b = band({**box, j: (a, np.inf)})
            if hi_b - hi_a <= 2 * eps:
                b = np.inf
            box[j] = (a, b)

        kept = sorted(box)
        a, b = band(box)
        preds = [(j, ">", box[j][0]) for j in kept if np.isfinite(box[j][0])]
        preds += [(j, "<=", box[j][1]) for j in kept if np.isfinite(box[j][1])]
        proof = None
        if preds:
            head, body, inputs = _region_rule("suff", preds)
            facts = _region_facts(X, [row], inputs)
            fired_c, prov = FOLKernel([(head, body)]).closure(facts)
            if any(t[0] == head[0] for t in fired_c):
                proof = FOLKernel([(head, body)]).proof((head[0], row), prov)
        conds = [
            {
                "feature": int(j),
                "lo": (None if not np.isfinite(box[j][0]) else round(box[j][0], 4)),
                "hi": (None if not np.isfinite(box[j][1]) else round(box[j][1], 4)),
            }
            for j in kept
        ]
        return {
            "prediction": round(pred, 2),
            "eps": round(float(eps), 2),
            "certified_band": [round(a, 2), round(b, 2)],
            "band_width": round(b - a, 2),
            "certified": bool(b - a <= 2 * eps),
            "n_conditions": len(kept),
            "of_features": len(universe),
            "conditions": conds,
            "proof": proof,
        }

    def proof(self, X, row, n_trees=8):
        """The checkable witness for row's prediction: for the first n_trees stages, the fired region clause +
        its FOLKernel membership proof + the stage contribution; plus base and the running sum."""
        X = np.asarray(X, float)
        terms = []
        total = self.base_
        for m, tree in enumerate(self.trees_):
            contrib = self.lr_ * float(
                _affine_tree_pred(tree, X[row : row + 1])[0]
            )  # affine or constant leaf
            total += contrib
            if m < n_trees:
                for rid, (preds, _) in enumerate(_leaf_regions(tree)):
                    if not preds:
                        terms.append(
                            {
                                "stage": m,
                                "region": rid,
                                "body_len": 0,
                                "contribution": float(contrib),
                                "proof": "root",
                            }
                        )
                        break
                    head, body, inputs = _region_rule(rid, preds)
                    facts = _region_facts(X, [row], inputs)
                    fired, prov = FOLKernel([(head, body)]).closure(facts)
                    if any(tp[0] == head[0] for tp in fired):
                        terms.append(
                            {
                                "stage": m,
                                "region": rid,
                                "body_len": len(body),
                                "contribution": float(contrib),
                                "proof": FOLKernel([(head, body)]).proof((head[0], row), prov),
                            }
                        )
                        break
        return {
            "base": float(self.base_),
            "terms_shown": terms,
            "n_stages": len(self.trees_),
            "prediction": float(total),
        }


class AdditiveCertifiedClassifier:
    """Softmax-boosted certified region trees: per class k an additive score F_k(x) = Σ lr·value(region);
    predict argmax_k F_k. Each leaf region is a threshold clause the FOLKernel verifies, so the argmax is a
    decision over KERNEL-VERIFIED region memberships — full-coverage classification that keeps the proof."""

    def __init__(
        self,
        rounds=200,
        lr=0.15,
        depth=4,
        leaf=20,
        holdout=0.3,
        patience=30,
        class_weight=None,
        seed=0,
        refit=True,
        linear_leaf=False,
        lam_lin=1.0,
        categorical_groups=(),
        honest_categorical=False,
        fit_cap=None,
        rare_event=False,
        rare_min_events=20_000,
        min_verifier_events=500,
        shared_structure=False,
        max_leaves=None,
        best_first_pair=False,
        adaptive_best_first_pair=False,
        verifier_gated_pair_growth=False,
        coupled_pair_growth=False,
        honest_pair_growth=False,
        validation_metric="logloss",
        track_validation_metrics=(),
        track_residual_dynamics=False,
        stratified_holdout=False,
        base_feature_count=None,
        residual_stumps=(),
        allowed=None,
        independent_calibration=False,
    ):
        self.rounds, self.lr, self.depth, self.leaf = rounds, lr, depth, leaf
        self.holdout, self.patience, self.class_weight, self.seed = holdout, patience, class_weight, seed
        self.refit = refit
        self.linear_leaf, self.lam_lin = linear_leaf, lam_lin
        self.categorical_groups = tuple(tuple(int(j) for j in group) for group in categorical_groups)
        self.honest_categorical = bool(honest_categorical)
        self.fit_cap = fit_cap
        self.rare_event = bool(rare_event)
        self.rare_min_events = int(rare_min_events)
        self.min_verifier_events = int(min_verifier_events)
        self.shared_structure = bool(shared_structure)
        self.max_leaves = None if max_leaves is None else int(max_leaves)
        self.best_first_pair = bool(best_first_pair)
        self.adaptive_best_first_pair = bool(adaptive_best_first_pair)
        self.verifier_gated_pair_growth = bool(verifier_gated_pair_growth)
        self.coupled_pair_growth = bool(coupled_pair_growth)
        self.honest_pair_growth = bool(honest_pair_growth)
        self.validation_metric = str(validation_metric)
        self.track_validation_metrics = tuple(str(metric) for metric in track_validation_metrics)
        self.track_residual_dynamics = bool(track_residual_dynamics)
        self.stratified_holdout = bool(stratified_holdout)
        self.base_feature_count = None if base_feature_count is None else int(base_feature_count)
        self.residual_stumps = tuple(residual_stumps)
        self.allowed = None if allowed is None else tuple(int(feature) for feature in allowed)
        self.independent_calibration = bool(independent_calibration)

    def fit(self, X, y, sample_weight=None, validation_groups=None):
        # OUR OWN multinomial boosting (trees.py reason_boost_softmax) — no external model.
        X = np.asarray(X, float)
        y = np.asarray(y)
        groups = None if validation_groups is None else np.asarray(validation_groups, dtype=np.int64)
        if groups is not None and (groups.ndim != 1 or len(groups) != len(y)):
            raise ValueError("validation_groups must have one value per classification row")
        source_classes, source_counts = np.unique(y, return_counts=True)
        if groups is None:
            fit_rows, inclusion_weight = _fit_sample(
                y,
                self.fit_cap,
                self.seed * 7 + 3,
                stratified=True,
                min_class_rows=(self.rare_min_events if self.rare_event else 0),
            )
            self.fit_sampling_ = {
                "mode": ("exchangeable_full" if fit_rows is None else "exchangeable_stratified_reservoir"),
                "source_rows": int(len(y)),
                "sample_rows": int(len(y) if fit_rows is None else len(fit_rows)),
            }
        else:
            fit_rows, inclusion_weight, self.fit_sampling_ = _temporal_fit_sample(
                y,
                groups,
                self.fit_cap,
                self.seed * 7 + 3,
                stratified=True,
                min_class_rows=(self.rare_min_events if self.rare_event else 0),
            )
        self.fit_rows_ = fit_rows
        X_fit = X if fit_rows is None else X[fit_rows]
        y_fit = y if fit_rows is None else y[fit_rows]
        groups_fit = groups if fit_rows is None or groups is None else groups[fit_rows]
        fit_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if fit_weight is not None:
            if len(fit_weight) != len(y):
                raise ValueError("sample_weight must have one value per source row")
            fit_weight = fit_weight if fit_rows is None else fit_weight[fit_rows]
        if inclusion_weight is not None:
            fit_weight = inclusion_weight if fit_weight is None else fit_weight * inclusion_weight
        if fit_weight is not None:
            fit_weight = fit_weight / np.mean(fit_weight)
        self.fit_sample_weight_ = fit_weight
        self.source_classes_ = source_classes
        self.source_class_counts_ = source_counts
        self.sample_classes_, self.sample_class_counts_ = np.unique(y_fit, return_counts=True)
        if self.residual_stumps:
            if len(source_classes) <= 2:
                raise ValueError("residual_stumps require a multiclass target")
            if self.linear_leaf:
                raise ValueError("residual_stumps do not support affine leaves")
            if self.base_feature_count is None:
                raise ValueError("residual_stumps require base_feature_count")
        if self.base_feature_count is not None and not (1 <= self.base_feature_count <= X.shape[1]):
            raise ValueError("base_feature_count must select a non-empty prefix of X")
        quartiles = np.nanpercentile(X_fit, (25, 75), axis=0)
        iqr = quartiles[1] - quartiles[0]  # per-feature scale (for robustness radii)
        self.scale_ = np.where(iqr > 0, iqr, 1.0)
        early_parallel = (
            self.refit
            and not self.linear_leaf
            and len(X_fit) >= _PARALLEL_K_FINAL_MIN_ROWS
            and np.unique(y_fit).size <= 4
        )
        result = reason_boost_softmax(
            X_fit,
            y_fit,
            rounds=self.rounds,
            lr=self.lr,
            depth=self.depth,
            min_leaf=self.leaf,
            holdout=self.holdout,
            patience=self.patience,
            class_weight=self.class_weight,
            seed=self.seed,
            refit=self.refit,
            parallel_k=(len(X_fit) >= _PARALLEL_K_MIN_ROWS or early_parallel)
            and _numba_threading_is_threadsafe(),
            linear_leaf=self.linear_leaf,
            lam_lin=self.lam_lin,
            categorical_groups=self.categorical_groups,
            honest_categorical=self.honest_categorical,
            sample_weight=fit_weight,
            stratified_holdout=(
                self.rare_event
                or self.stratified_holdout
                or (len(source_classes) > 2 and self.validation_metric in {"auc", "macro_ovo_auc"})
            ),
            min_verifier_events=(self.min_verifier_events if self.rare_event else 0),
            shared_structure=self.shared_structure,
            max_leaves=self.max_leaves,
            best_first_pair=self.best_first_pair,
            adaptive_best_first_pair=self.adaptive_best_first_pair,
            verifier_gated_pair_growth=self.verifier_gated_pair_growth,
            coupled_pair_growth=self.coupled_pair_growth,
            honest_pair_growth=self.honest_pair_growth,
            validation_metric=self.validation_metric,
            track_validation_metrics=self.track_validation_metrics,
            track_residual_dynamics=self.track_residual_dynamics,
            feature_count=self.base_feature_count,
            allowed=self.allowed,
            validation_groups=groups_fit,
            independent_calibration=self.independent_calibration,
        )
        base, lr, trees, classes, flats, ver, linear = result[:7]
        training_trace = result[7] if len(result) > 7 else None
        self._checkpoint_trace = (
            training_trace if training_trace is not None and "tree_counts" in training_trace else None
        )
        self.residual_dynamics_ = None if training_trace is None else training_trace.get("residual_dynamics")
        self.residual_dynamics_monitored_rounds_ = (
            0 if training_trace is None else int(training_trace.get("dynamics_monitored_rounds", 0))
        )
        selected_dynamics_rounds = (
            0 if training_trace is None else int(training_trace.get("selected_rounds", 0))
        )
        self.pair_growth_schedule_ = (
            ()
            if training_trace is None
            else tuple(
                training_trace.get(
                    "deployed_pair_growth_schedule",
                    training_trace.get("pair_growth_schedule", ()),
                )
            )[:selected_dynamics_rounds]
        )
        self.attempted_pair_growth_schedule_ = (
            ()
            if training_trace is None
            else tuple(
                training_trace.get(
                    "attempted_pair_growth_schedule",
                    training_trace.get("pair_growth_schedule", ()),
                )
            )
        )
        self.pair_expert_decisions_ = (
            ()
            if training_trace is None
            else tuple(dict(row) for row in training_trace.get("pair_expert_decisions", ()))
        )
        if self.residual_stumps:
            trees, flats = list(trees), list(flats)
            class_index = {label: index for index, label in enumerate(classes)}
            for feature, class_label, false_value, true_value in self.residual_stumps:
                if class_label not in class_index:
                    raise ValueError("residual stump class is absent from fitted target")
                tree = (
                    "node",
                    int(feature),
                    0.5,
                    ("leaf", float(false_value)),
                    ("leaf", float(true_value)),
                )
                trees.append((class_index[class_label], tree))
                flats.append(_flatten_tree(tree))
        self.base_, self.lr_, self.trees_, self.classes_ = base, lr, trees, list(classes)
        self._flat_cache = flats  # flats built once by the booster (aligned to trees_), no re-flatten
        self.linear_ = linear  # affine (linear-leaf) prediction path when True
        self.shared_structure_ = bool(
            self.shared_structure
            and len(self.classes_) > 2
            and not self.linear_
            and not self.categorical_groups
            and not self.residual_stumps
        )
        self._serving_forest_cache = None  # compiled lazily; not part of the proof representation
        self._serving_forest_ready = False
        self._adaptive_depth_cache = None
        self._adaptive_depth_ready = False
        self._adaptive_depth_reason = None
        self._adaptive_depth_selected = False
        self.adaptive_depth_selection_ = None
        self.ver_ = fit_rows[ver] if fit_rows is not None and ver is not None else ver
        self.verifier_evidence_role_ = (
            "independent_calibration"
            if self.independent_calibration and not self.refit and ver is not None
            else ("checkpoint_selection" if ver is not None else None)
        )
        self.ver_weight_ = (
            _prior_preserving_subset_weight(y_fit, fit_weight, ver)
            if self.rare_event and ver is not None and groups_fit is None
            else (None if fit_weight is None or ver is None else fit_weight[ver])
        )
        self.fit_evidence_count_ = len(X_fit)
        # Certificate contraction is independent of normal prediction. Build
        # its static topology only when a robustness/report query needs it, so
        # temporary gate and OOF models do not pay verifier setup during fit.
        self._region_graph = None
        self._region_graph_ready = False
        self._region_graph_n_features = int(X.shape[1])
        audit_rows = X_fit[ver] if ver is not None and not self.refit else X_fit
        self._select_adaptive_depth(audit_rows, held_out=bool(ver is not None and not self.refit))
        return self

    def _scores_at_checkpoint(self, X, metric):
        """Score an internally tracked verifier-selected prefix on arbitrary rows."""
        metric = str(metric)
        if metric == self.validation_metric:
            return self._scores(X)
        trace = self._checkpoint_trace
        if trace is None or metric not in trace["tree_counts"]:
            raise ValueError(f"validation checkpoint was not tracked: {metric}")
        X = np.asarray(X, float)
        keep = int(trace["tree_counts"][metric])
        F = np.tile(np.asarray(self.base_, float), (len(X), 1))
        for (class_index, tree), flat in zip(
            trace["trees"][:keep],
            trace["flats"][:keep],
            strict=False,
        ):
            prediction = _tree_pred(tree, X) if flat is None else _flat_pred(flat, X)
            F[:, class_index] += self.lr_ * prediction
        return F

    def _thresholds(self):
        """Per-feature sorted split thresholds across the whole ensemble (cached)."""
        if not hasattr(self, "_thr_cache"):
            thr, stack = {}, [t for _, t in self.trees_]
            while stack:
                t = stack.pop()
                if t[0] == "node":
                    _, j, th, lt, rt = t
                    thr.setdefault(j, set()).add(float(th))
                    stack += [lt, rt]
                elif t[0] == "cat":
                    stack += [t[3], t[4]]
            self._thr_cache = {j: np.array(sorted(s)) for j, s in thr.items()}
        return self._thr_cache

    def _score_interval_ibp(self, lo, hi):
        """SOUND but loose per-class score bounds by independent per-tree interval propagation (the fallback
        when the exact enumeration below would blow up)."""
        smin = np.array(self.base_, float).copy()
        smax = smin.copy()
        flats = self._flats()
        if _HAS_NUMBA and not getattr(self, "linear_", False) and all(flat is not None for flat in flats):
            lo = np.ascontiguousarray(lo, np.float64)
            hi = np.ascontiguousarray(hi, np.float64)
            for (k, _), (feat, thr, left, right, val) in zip(self.trees_, flats, strict=False):
                a, b = _tree_interval_flat_nb(feat, thr, left, right, val, lo, hi)
                smin[k] += self.lr_ * a
                smax[k] += self.lr_ * b
        else:  # tuple `_tree_interval` bounds AFFINE (linear-leaf) regions too via _leaf_affine_interval
            for k, t in self.trees_:
                a, b = _tree_interval(t, lo, hi)
                smin[k] += self.lr_ * a
                smax[k] += self.lr_ * b
        return smin, smax

    def _score_interval(self, lo, hi, cap=4096):
        """EXACT [min,max] of every class's additive score over the box {lo[j] < x_j <= hi[j]}, when tractable.
        The ensemble is piecewise-constant, changing only where a split threshold lies inside the box; so the
        exact per-class range is realised on the grid of sub-cells induced by the ACTIVE features (those with a
        threshold strictly inside the box). We enumerate that grid and evaluate the scores exactly — tight, not
        the loose independent-per-tree bound. Falls back to the sound IBP bound if the grid exceeds `cap`."""
        if getattr(self, "linear_", False):  # affine leaves vary WITHIN a sub-cell -> grid-rep enumeration is
            return self._score_interval_ibp(lo, hi)  # unsound; the affine-aware IBP bound is sound (looser)
        if any(flat is None for flat in self._flats()):
            # Category membership is atomic rather than an axis-aligned numeric threshold. The tuple IBP union
            # is conservative but sound; enumerating numeric sub-cells alone could miss a valid level switch.
            return self._score_interval_ibp(lo, hi)
        thr = self._thresholds()
        active, reps, combos = [], [], 1
        for j, ts in thr.items():
            inside = ts[(ts > lo[j]) & (ts < hi[j])]
            if len(inside) == 0:
                continue
            flo = inside[0] - 1.0 if not np.isfinite(lo[j]) else lo[j]
            fhi = inside[-1] + 1.0 if not np.isfinite(hi[j]) else hi[j]
            bounds = np.concatenate(([flo], inside, [fhi]))
            active.append(j)
            reps.append((bounds[:-1] + bounds[1:]) / 2.0)  # one representative strictly inside each sub-cell
            combos *= len(reps[-1])
            if combos > cap:
                return self._score_interval_ibp(lo, hi)
        # representative strictly inside the box, on the correct side of every threshold OUTSIDE it. A feature
        # with an infinite bound must NOT default to 0.0 (that can land on the wrong side of a split and route
        # trees to the wrong leaves) — offset from the finite bound instead.
        base_pt = np.empty(len(lo))
        for j in range(len(lo)):
            if np.isfinite(lo[j]) and np.isfinite(hi[j]):
                base_pt[j] = (lo[j] + hi[j]) / 2.0
            elif np.isfinite(hi[j]):
                base_pt[j] = hi[j] - 1.0
            elif np.isfinite(lo[j]):
                base_pt[j] = lo[j] + 1.0
            else:
                base_pt[j] = 0.0
        if not active:  # ensemble constant over the box -> one exact evaluation
            F = self._scores(base_pt[None, :])[0]
            return F.copy(), F.copy()
        import itertools

        grid = np.tile(base_pt, (combos, 1))
        for i, combo in enumerate(itertools.product(*reps)):
            for jj, v in zip(active, combo, strict=False):
                grid[i, jj] = v
        # Only trees splitting on an ACTIVE feature vary over the grid; evaluate the rest ONCE at base_pt.
        act = set(active)
        tf = self._tree_feats()
        flats = self._flats()
        F = np.tile(np.asarray(self.base_, float), (combos, 1))
        for ti, (k, _) in enumerate(self.trees_):
            if tf[ti] & act:
                F[:, k] += self.lr_ * _flat_pred(flats[ti], grid)
            else:
                F[:, k] += self.lr_ * float(_flat_pred(flats[ti], base_pt[None, :])[0])
        return F.min(0), F.max(0)

    def certified_robustness(self, X, row, rel=0.1, delta=None, k=None):
        """CERTIFIED ROBUSTNESS: is the predicted class PROVABLY unable to flip for any input within ±delta of
        this row? True iff the predicted class's MIN possible score over the box exceeds every other class's
        MAX possible score (exact interval propagation — no sampling/attacks). delta defaults to rel·per-feature
        IQR. Returns the predicted class, whether it is certified stable, and the score margin (>0 => certified).
        `k` (predicted class index) may be passed in to skip the argmax pass — it is invariant to delta, so a
        bisection over radii (`robust_radius`) computes it once instead of on every step."""
        X = np.asarray(X, float)
        x = X[row]
        d = X.shape[1]
        dvec = (
            rel * self.scale_
            if delta is None
            else (np.full(d, float(delta)) if np.isscalar(delta) else np.asarray(delta, float))
        )
        if k is None:
            k = int(self._scores(x[None, :])[0].argmax())
        margin = self._margin_lower_bound(x - dvec, x + dvec, k)
        return {
            "class": (
                int(self.classes_[k])
                if np.issubdtype(type(self.classes_[k]), np.integer)
                else self.classes_[k]
            ),
            "certified_stable": bool(margin > 0),
            "margin": round(margin, 4),
        }

    def sufficient_reason(self, X, row, groups=None, widen=None):
        """Minimal certified SUFFICIENT REASON for the predicted CLASS: the smallest set of threshold
        conditions under which the predicted class's score-interval PROVABLY dominates every other class (the
        class cannot change), every other feature free and provably irrelevant. ALWAYS CERTIFIED for a
        non-tied prediction: we start from the row's SUB-CELL box on every split feature (the widest box that
        crosses no threshold), which pins every tree to the row's leaf, so the score is exact and the margin
        equals the true class gap (>0). We then greedily DROP features while domination holds, and finally
        WIDEN each surviving box outward as far as domination allows — yielding the most general readable rule.
        `groups` (list of column-index lists, e.g. the one-hot dummies of one categorical) are dropped/kept
        ATOMICALLY — fewer, coherent, faster. Kernel-verified. The classification analog of the regression one."""
        X = np.asarray(X, float)
        x = X[row]
        d = X.shape[1]
        thr = self._thresholds()
        k = int(self._scores(x[None, :])[0].argmax())
        # sub-cell box per split feature: widest interval around x[j] crossing no threshold of j -> pins trees
        sub_lo, sub_hi = {}, {}
        for j, ts in thr.items():
            below = ts[ts < x[j]]
            atabove = ts[ts >= x[j]]
            sub_lo[j] = float(below[-1]) if len(below) else -np.inf
            sub_hi[j] = float(atabove[0]) if len(atabove) else np.inf
        universe = sorted(thr)

        def margin(box):  # box: {j: (lo, hi)}; features not in box are free
            lo = np.full(d, -np.inf)
            hi = np.full(d, np.inf)
            for j, (a, b) in box.items():
                lo[j], hi[j] = a, b
            return self._margin_lower_bound(lo, hi, k)

        box = {j: (sub_lo[j], sub_hi[j]) for j in universe}  # certified by construction (all trees pinned)
        gs = _group_universe(groups, universe)
        feature_use = {}
        for features in self._tree_feats():
            for feature in features:
                feature_use[feature] = feature_use.get(feature, 0) + 1
        box = _smallest_reason_box(
            box,
            gs,
            valid=lambda value: value > 0,
            quality=margin,
            feature_use=feature_use,
        )
        # WIDEN: relax each surviving NUMERIC bound to +-inf where domination still holds (more general rule).
        # Restricted to `widen` cols (one-hot dummies are already collapsed downstream, pointless to widen).
        for j in list(box):
            if widen is not None and j not in widen:
                continue
            a, b = box[j]
            if margin({**box, j: (-np.inf, b)}) > 0:
                a = -np.inf
            if margin({**box, j: (a, np.inf)}) > 0:
                b = np.inf
            box[j] = (a, b)

        kept = sorted(box)
        preds = [(j, ">", box[j][0]) for j in kept if np.isfinite(box[j][0])]
        preds += [(j, "<=", box[j][1]) for j in kept if np.isfinite(box[j][1])]
        proof = None
        if preds:
            head, body, inputs = _region_rule("suffc", preds)
            facts = _region_facts(X, [row], inputs)
            fired_c, prov = FOLKernel([(head, body)]).closure(facts)
            if any(t[0] == head[0] for t in fired_c):
                proof = FOLKernel([(head, body)]).proof((head[0], row), prov)
        cls = self.classes_[k]
        conds = [
            {
                "feature": int(j),
                "lo": (None if not np.isfinite(box[j][0]) else round(box[j][0], 4)),
                "hi": (None if not np.isfinite(box[j][1]) else round(box[j][1], 4)),
            }
            for j in kept
        ]
        m_final = float(margin(box))
        return {
            "class": int(cls) if np.issubdtype(type(cls), np.integer) else cls,
            "margin": round(m_final, 4),
            "certified": bool(m_final > 0),
            "n_conditions": len(kept),
            "of_features": len(universe),
            "conditions": conds,
            "proof": proof,
        }

    def recourse(self, X, row, target=None, max_options=3, meta=None):
        """Certified counterfactual recourse for CLASSIFICATION: the minimal change that flips the predicted
        class (to `target` if given, else any other class). Numeric features move across tree thresholds; a
        categorical (one-hot group in `meta`) is switched to another LEVEL atomically — so every candidate is a
        VALID input. Each option is kernel-verified (re-predicting the candidate yields the stated new class).
        `reachable` is False when no single-feature change reaches the goal. `target` is the business question."""
        X = np.asarray(X, float)
        x = X[row]
        k0 = self.predict(x[None, :])[0]

        def _int(c):
            return int(c) if np.issubdtype(type(c), np.integer) else c

        goal = (lambda c: c != k0) if target is None else (lambda c: c == target)
        if goal(k0):
            return {"class": _int(k0), "target": target, "satisfied": True, "reachable": True, "options": []}
        opts = _recourse_search(self.predict, goal, x, self._thresholds(), self.scale_, meta, max_options)
        for o in opts:
            xp = x.copy()
            if o["kind"] == "categorical":
                xp[o["cols"]] = 0.0
                xp[o["to_col"]] = 1.0
            else:
                xp[o["col"]] = o["to"]
            chk = self.predict(xp[None, :])[0]
            o["new_class"] = _int(o.pop("outcome"))
            o["kernel_reproduced"] = bool(goal(chk) and chk == o["new_class"])
            o.pop("cols", None)
            o.pop("to_col", None)
        return {
            "class": _int(k0),
            "target": target,
            "satisfied": False,
            "reachable": bool(opts),
            "options": opts,
        }

    def robust_radius(self, X, row, hi_rel=3.0, iters=18):
        """Largest uniform perturbation (as a multiple of per-feature IQR) within which the predicted class is
        CERTIFIED unable to flip — found by bisection on the exact score-interval domination test."""
        k = int(
            self._scores(np.asarray(X, float)[row][None, :])[0].argmax()
        )  # predicted class, invariant to rel
        lorel, hirel = 0.0, hi_rel
        if not self.certified_robustness(X, row, rel=lorel + 1e-9, k=k)["certified_stable"]:
            return 0.0
        if self.certified_robustness(X, row, rel=hirel, k=k)["certified_stable"]:
            return float(hi_rel)
        for _ in range(iters):
            mid = 0.5 * (lorel + hirel)
            if self.certified_robustness(X, row, rel=mid, k=k)["certified_stable"]:
                lorel = mid
            else:
                hirel = mid
        return round(float(lorel), 4)

    def _flats(self):
        """Flattened (array) form of every tree, cached — for vectorized batch prediction."""
        if not hasattr(self, "_flat_cache"):
            self._flat_cache = [_flatten_tree(t) for _, t in self.trees_]
        return self._flat_cache

    def _serving_forest(self):
        """Cached packed numeric forest for the normal prediction path.

        Tuple trees remain authoritative for proofs and certificate queries.
        Categorical trees fail closed to the existing per-tree evaluator because
        their atomic membership rules intentionally do not share the numeric
        flat ABI.
        """
        if not getattr(self, "_serving_forest_ready", False):
            stages = [k for k, _tree in self.trees_]
            if getattr(self, "linear_", False):
                packed = _pack_affine_forest([tree for _k, tree in self.trees_], stages)
                self._serving_forest_cache = None if packed is None else ("affine", packed)
            else:
                mirrored = _pack_binary_mirrored_forest(self._flats(), stages)
                if mirrored is not None:
                    self._serving_forest_cache = ("binary_mirrored", mirrored)
                else:
                    shared = _pack_multiclass_shared_forest(self._flats(), stages, len(self.classes_))
                    if shared is not None:
                        self._serving_forest_cache = ("multiclass_shared", shared)
                    else:
                        packed = _pack_flat_forest(self._flats(), stages)
                        self._serving_forest_cache = None if packed is None else ("flat", packed)
            self._serving_forest_ready = True
        return self._serving_forest_cache

    def _adaptive_depth_forest(self):
        """Return a lazily compiled exact early-exit representation.

        Only constant-leaf numeric forests are supported. Every unsupported or
        shallow layout fails closed to the existing fused full-score kernel.
        """
        if getattr(self, "_adaptive_depth_ready", False):
            return getattr(self, "_adaptive_depth_cache", None)
        self._adaptive_depth_ready = True
        self._adaptive_depth_cache = None
        self._adaptive_depth_reason = None
        if not _HAS_NUMBA:
            self._adaptive_depth_reason = "numba_unavailable"
            return None
        if not all(hasattr(self, name) for name in ("base_", "lr_", "trees_", "classes_")):
            self._adaptive_depth_reason = "incomplete_fitted_state"
            return None
        forest = self._serving_forest()
        if forest is None:
            self._adaptive_depth_reason = "unsupported_forest_representation"
            return None
        kind, packed = forest
        if kind == "affine":
            self._adaptive_depth_reason = "affine_leaves_require_full_scores"
            return None
        if kind not in {"flat", "binary_mirrored", "multiclass_shared"}:
            self._adaptive_depth_reason = "unsupported_forest_representation"
            return None
        stage_class = packed[6] if kind == "flat" else ()
        stride = _adaptive_checkpoint_stride(kind, stage_class, len(self.classes_))
        stages = len(packed[5])
        if stages <= stride:
            self._adaptive_depth_reason = "insufficient_routing_depth"
            return None
        bounds = _adaptive_suffix_bounds(kind, packed, self.lr_, len(self.classes_))
        if bounds is None:
            self._adaptive_depth_reason = "nonfinite_or_invalid_leaf_bounds"
            return None
        self._adaptive_depth_cache = (kind, packed, *bounds, int(stride))
        return self._adaptive_depth_cache

    def _adaptive_predict_indices(self, X, *, return_evaluated=False):
        """Return exact class indices, or ``None`` when adaptive depth is unsupported."""
        if not return_evaluated and getattr(self, "_adaptive_depth_selected", False) is False:
            return None
        compiled = self._adaptive_depth_forest()
        if compiled is None:
            return None
        kind, packed, suffix_min, suffix_max, suffix_abs, suffix_count, stride = compiled
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        base = np.ascontiguousarray(np.asarray(self.base_, dtype=np.float64))
        if kind == "flat":
            feat, thr, left, right, val, starts, stage_class = packed
            prediction, evaluated = _forest_predict_adaptive_flat_nb(
                feat,
                thr,
                left,
                right,
                val,
                starts,
                stage_class,
                base,
                self.lr_,
                suffix_min,
                suffix_max,
                suffix_abs,
                suffix_count,
                stride,
                X,
            )
        elif kind == "binary_mirrored":
            feat, thr, left, right, val, starts = packed
            prediction, evaluated = _forest_predict_adaptive_binary_mirrored_nb(
                feat,
                thr,
                left,
                right,
                val,
                starts,
                base,
                self.lr_,
                suffix_min,
                suffix_max,
                suffix_abs,
                suffix_count,
                stride,
                X,
            )
        else:
            feat, thr, left, right, val, starts = packed
            prediction, evaluated = _forest_predict_adaptive_multiclass_shared_nb(
                feat,
                thr,
                left,
                right,
                val,
                starts,
                base,
                self.lr_,
                suffix_min,
                suffix_max,
                suffix_abs,
                suffix_count,
                stride,
                X,
            )
        if return_evaluated:
            return prediction, evaluated, len(packed[5]), kind, stride
        return prediction

    def _select_adaptive_depth(self, X, *, held_out):
        """Deploy adaptive depth only after a bounded exactness and work audit."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X):
            self._adaptive_depth_selected = False
            self.adaptive_depth_selection_ = {
                "selected": False,
                "reason": "audit_rows_unavailable",
                "held_out": bool(held_out),
            }
            return
        if len(X) > _ADAPTIVE_DEPTH_AUDIT_ROWS:
            rows = np.linspace(0, len(X) - 1, _ADAPTIVE_DEPTH_AUDIT_ROWS, dtype=np.int64)
            X = X[rows]
        adaptive = self._adaptive_predict_indices(X, return_evaluated=True)
        if adaptive is None:
            self._adaptive_depth_selected = False
            self.adaptive_depth_selection_ = {
                "selected": False,
                "reason": getattr(self, "_adaptive_depth_reason", "unsupported_forest_representation"),
                "held_out": bool(held_out),
                "audit_rows": int(len(X)),
            }
            return
        prediction, evaluated, stages, kind, stride = adaptive
        full_prediction = self._scores(X).argmax(axis=1)
        exact = bool(np.array_equal(prediction, full_prediction))
        reduction = float(1.0 - evaluated.sum() / (len(evaluated) * stages))
        minimum_reduction = (
            _ADAPTIVE_DEPTH_SHALLOW_MIN_STAGE_REDUCTION
            if stages < 64
            else _ADAPTIVE_DEPTH_MIN_STAGE_REDUCTION
        )
        if kind == "multiclass_shared":
            minimum_reduction = max(minimum_reduction, _ADAPTIVE_DEPTH_SHARED_MIN_STAGE_REDUCTION)
        selected = bool(exact and reduction >= minimum_reduction)
        self._adaptive_depth_selected = selected
        self.adaptive_depth_selection_ = {
            "selected": selected,
            "reason": (
                "verified_stage_reduction"
                if selected
                else ("prediction_mismatch" if not exact else "stage_reduction_below_gate")
            ),
            "held_out": bool(held_out),
            "audit_rows": int(len(X)),
            "forest_kind": kind,
            "routing_stages": int(stages),
            "checkpoint_stride": int(stride),
            "stage_reduction": reduction,
            "predictions_match_full_forest": exact,
            "minimum_stage_reduction": minimum_reduction,
        }

    def adaptive_depth_report(self, X):
        """Audit exact input-dependent depth against the full fused forest."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X):
            raise ValueError("X must contain at least one row in a 2-D table")
        adaptive = self._adaptive_predict_indices(X, return_evaluated=True)
        if adaptive is None:
            return {
                "active": False,
                "reason": getattr(self, "_adaptive_depth_reason", "unsupported_forest_representation"),
                "rows": int(len(X)),
                "probability_path": "full_forest_required",
            }
        prediction, evaluated, stages, kind, stride = adaptive
        full_prediction = self._scores(X).argmax(axis=1)
        early = evaluated < stages
        return {
            "active": True,
            "selected_for_predict": bool(getattr(self, "_adaptive_depth_selected", False)),
            "certificate": "global_leaf_suffix_bounds",
            "forest_kind": kind,
            "rows": int(len(X)),
            "routing_stages": int(stages),
            "checkpoint_stride": int(stride),
            "rows_exited_early": int(early.sum()),
            "early_exit_rate": float(early.mean()),
            "mean_routing_stages": float(evaluated.mean()),
            "p50_routing_stages": float(np.percentile(evaluated, 50)),
            "p95_routing_stages": float(np.percentile(evaluated, 95)),
            "routing_stage_reduction": float(1.0 - evaluated.sum() / (len(evaluated) * stages)),
            "predictions_match_full_forest": bool(np.array_equal(prediction, full_prediction)),
            "probability_path": "full_forest_required",
        }

    def _tree_feats(self):
        """Per-tree split-feature sets (cached) — lets `_score_interval` skip trees constant over the box."""
        if not hasattr(self, "_tf_cache"):
            self._tf_cache = [_tree_features(t) for _, t in self.trees_]
        return self._tf_cache

    def _margin_lower_bound(self, lo, hi, winner, cap=4096):
        """Sound lower bound on the winner's margin over every challenger.

        The compiled region graph retains score correlations across trees and
        can therefore certify a margin that independent per-class intervals
        miss.  Its hard state cap is fail-closed: any unsupported query uses
        the original interval implementation unchanged.
        """
        graph = self._compiled_region_graph()
        if graph is not None:
            margins = [
                graph.margin_lower(lo, hi, winner, challenger, cap=cap)
                for challenger in range(len(self.classes_))
                if challenger != winner
            ]
            if margins and all(value is not None for value in margins):
                return float(min(margins))
        smin, smax = self._score_interval(lo, hi, cap=cap)
        other = smax.copy()
        other[winner] = -np.inf
        return float(smin[winner] - other.max())

    def _compiled_region_graph(self):
        """Return the lazily built exact certificate contraction graph."""
        if not getattr(self, "_region_graph_ready", True):
            self._region_graph = (
                None
                if self.linear_
                else _CompiledRegionGraph(
                    self.trees_,
                    self._flat_cache,
                    self.lr_,
                    self.base_,
                    self._region_graph_n_features,
                )
            )
            self._region_graph_ready = True
        return getattr(self, "_region_graph", None)

    def compiled_region_report(self):
        """Static CCH-style verifier compilation summary for this fitted model."""
        graph = self._compiled_region_graph()
        return (
            {
                "active": False,
                "factors": 0,
                "atomic_predicates": 0,
                "separator_features": 0,
                "skip_reason": "linear_leaf",
            }
            if graph is None
            else graph.report()
        )

    def _scores(self, X):
        X = np.asarray(X, float)
        if _HAS_NUMBA:
            forest = self._serving_forest()
            if forest is not None:
                kind, packed = forest
                Xc = np.ascontiguousarray(X, np.float64)
                base = np.ascontiguousarray(np.asarray(self.base_, float), np.float64)
                if kind == "affine":
                    feat, thr, left, right, val, starts, classes, lin_feat, lin_coef, lin_lo, lin_hi = packed
                    return _forest_scores_affine_nb(
                        feat,
                        thr,
                        left,
                        right,
                        val,
                        starts,
                        classes,
                        lin_feat,
                        lin_coef,
                        lin_lo,
                        lin_hi,
                        base,
                        self.lr_,
                        Xc,
                    )
                if kind == "binary_mirrored":
                    feat, thr, left, right, val, starts = packed
                    return _forest_scores_binary_mirrored_nb(
                        feat, thr, left, right, val, starts, base, self.lr_, Xc
                    )
                if kind == "multiclass_shared":
                    feat, thr, left, right, val, starts = packed
                    return _forest_scores_multiclass_shared_nb(
                        feat, thr, left, right, val, starts, base, self.lr_, Xc
                    )
                feat, thr, left, right, val, starts, classes = packed
                return _forest_scores_flat_nb(
                    feat, thr, left, right, val, starts, classes, base, self.lr_, Xc
                )
        F = np.tile(np.asarray(self.base_, float), (len(X), 1))
        if getattr(self, "linear_", False):  # affine leaves vary within a region -> predict on the tuple tree
            for k, t in self.trees_:
                F[:, k] += self.lr_ * _affine_tree_pred(t, X)
            return F
        for (k, tree), flat in zip(self.trees_, self._flats(), strict=False):
            F[:, k] += self.lr_ * (_tree_pred(tree, X) if flat is None else _flat_pred(flat, X))
        return F

    def predict(self, X):
        X = np.asarray(X, float)
        adaptive = self._adaptive_predict_indices(X)
        indices = self._scores(X).argmax(1) if adaptive is None else adaptive
        return np.array(self.classes_)[indices]

    def kernel_certify(self, X, n_trees=40, sample=120):
        """The FOLKernel must reproduce each per-class tree's region membership (hence the additive scores and
        the argmax). Verifies membership of every leaf clause against numpy on a sample; composes to all rounds."""
        X = np.asarray(X, float)
        idx = np.random.default_rng(self.seed).choice(len(X), size=min(sample, len(X)), replace=False)
        Xs = X[idx]
        n = len(Xs)
        agrees = []
        for _k, t in self.trees_[:n_trees]:
            for rid, (preds, _) in enumerate(_leaf_regions(t)):
                if not preds:
                    continue
                head, body, inputs = _region_rule(rid, preds)
                facts = _region_facts(Xs, range(n), inputs)
                fired, _ = FOLKernel([(head, body)]).closure(facts)
                km = np.zeros(n, bool)
                for tp in fired:
                    if tp[0] == head[0]:
                        km[tp[1]] = True
                m = np.ones(n, bool)
                for pred in preds:
                    if pred[0] == "cat":
                        _, cols, op, levels = pred
                        member = _cat_member(Xs, cols, levels)
                        m &= member if op == "in" else ~member
                    else:
                        j, op, thr = pred
                        m &= (Xs[:, j] <= thr) if op == "<=" else (Xs[:, j] > thr)
                agrees.append(float((km == m).mean()))
        rep = float(np.allclose(np.mean(agrees), 1.0)) if agrees else 1.0
        return {
            "scores_reproduced": rep,
            "membership_reproduced": rep,
            "region_membership_recovered": float(np.mean(agrees)) if agrees else 1.0,
            "stages_checked": len(self.trees_[:n_trees]),
        }

    def proof(self, X, row, n_trees=8):
        """Witness for row's class: the predicted class's fired region per stage (kernel-proven membership)."""
        X = np.asarray(X, float)
        F = self._scores(X[row : row + 1])[0]
        k = int(F.argmax())
        cls = self.classes_[k]
        terms = []
        shown = 0
        for stage, (kk, t) in enumerate(self.trees_):
            if kk != k or shown >= n_trees:
                continue
            for rid, (preds, val) in enumerate(_leaf_regions(t)):
                if not preds:
                    terms.append(
                        {
                            "stage": stage,
                            "region": rid,
                            "logit_contribution": float(self.lr_ * val),
                            "proof": "root",
                        }
                    )
                    shown += 1
                    break
                head, body, inputs = _region_rule(rid, preds)
                facts = _region_facts(X, [row], inputs)
                fired, prov = FOLKernel([(head, body)]).closure(facts)
                if any(tp[0] == head[0] for tp in fired):
                    terms.append(
                        {
                            "stage": stage,
                            "region": rid,
                            "logit_contribution": float(self.lr_ * val),
                            "proof": FOLKernel([(head, body)]).proof((head[0], row), prov),
                        }
                    )
                    shown += 1
                    break
        return {
            "class": int(cls) if np.issubdtype(type(cls), np.integer) else cls,
            "score": float(F[k]),
            "n_stages": len(self.trees_),
            "terms_shown": terms,
        }
