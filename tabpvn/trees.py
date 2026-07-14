"""Boosting core — histogram-binned CART trees + second-order residual boosting, all our own (no
external model). The proposer behind TabPVN's certified predictors: each boosted term is a shallow
CART tree on the (2nd-order) residual, fit on a random row+feature subsample and KEPT only while it
improves a HELD-OUT split (early stopping = the verifier). Trees store real thresholds, so every
region compiles to a clause the FOLKernel can verify. Provides regression (`reason_boost_2nd`) and
multinomial classification (`reason_boost_softmax`) boosters.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tabpvn.residual_dynamics import ResidualDynamicsTracker, hardest_class_pair

# A tree's histogram build is independent across split features.  Above this
# amount of row-feature work, assigning one feature at a time to Numba workers
# amortizes the parallel-region overhead on ordinary desktop CPUs.  Smaller
# nodes retain the serial, cache-friendly loop below.
_PARALLEL_HIST_TREE_MIN_WORK = 1_000_000
_PARALLEL_HIST_NODE_MIN_WORK = 300_000
_PARALLEL_PREBIN_MIN_WORK = 100_000
_PARALLEL_PREBIN_MAX_WORKERS = 8
_PARALLEL_PREBIN_TEMP_BUDGET = 256 * 1024 * 1024

try:  # optional JIT for the histogram kernel
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # graceful fallback: pure-numpy path below
    _HAS_NUMBA = False
    _route_flat_nb = None  # name must exist for importers; only used when _HAS_NUMBA
    _route_flat_rows_nb = None
    _add_flat_binned_rows_nb = None
    _add_rows_nb = None
    _route_flat_leaf_nb = None
    _route_flat_affine_nb = None
    _forest_scores_flat_nb = None
    _forest_scores_binary_mirrored_nb = None
    _forest_scores_multiclass_shared_nb = None
    _forest_scores_affine_nb = None
    _forest_predict_adaptive_flat_nb = None
    _forest_predict_adaptive_binary_mirrored_nb = None
    _forest_predict_adaptive_multiclass_shared_nb = None
    _tree_interval_flat_nb = None
    _gather_binary_tree_fit_data_nb = None
    _grow_softmax_shared_binned_nb = None


def _numba_threading_is_threadsafe():
    """Whether Python worker threads may enter the active Numba backend concurrently."""
    if not _HAS_NUMBA:
        return True
    try:
        from numba import threading_layer

        return threading_layer() != "workqueue"
    except (ImportError, RuntimeError, ValueError):
        # An uninitialized or unavailable backend has not established that
        # concurrent entry is safe. It may later resolve to workqueue.
        return False


if _HAS_NUMBA:

    @njit(cache=True, nogil=True)
    def _route_flat_nb(feat, thr, left, right, val, X):
        """Leaf value per row through a flat tree, one row at a time in numba — no per-level numpy array
        allocations (nonzero / fancy-index / where), which dominate the vectorized path on the small holdout
        slices predicted every boosting round. Bit-identical routing: same `<=` split and child pointers."""
        n = X.shape[0]
        out = np.empty(n)
        for i in range(n):
            nd = 0
            while feat[nd] >= 0:
                nd = left[nd] if X[i, feat[nd]] <= thr[nd] else right[nd]
            out[i] = val[nd]
        return out

    @njit(cache=True, nogil=True)
    def _route_flat_rows_nb(feat, thr, left, right, val, X, rows):
        """Route selected source rows without materializing a dense row slice."""
        out = np.empty(rows.shape[0])
        for i in range(rows.shape[0]):
            nd = 0
            row = rows[i]
            while feat[nd] >= 0:
                nd = left[nd] if X[row, feat[nd]] <= thr[nd] else right[nd]
            out[i] = val[nd]
        return out

    @njit(cache=True, nogil=True)
    def _add_flat_binned_rows_nb(feat, bink, left, right, val, Xb, rows, scores, scale):
        """Route binned source rows and add their leaf values without temporaries."""
        for i in range(rows.shape[0]):
            row = rows[i]
            nd = 0
            while feat[nd] >= 0:
                nd = left[nd] if Xb[row, feat[nd]] <= bink[nd] else right[nd]
            scores[row] = scores[row] + scale * val[nd]

    @njit(cache=True, nogil=True)
    def _add_rows_nb(rows, values, scores, scale):
        """Scatter one ordered vector into source-row scores without a mask copy."""
        for i in range(rows.shape[0]):
            row = rows[i]
            scores[row] = scores[row] + scale * values[i]

    @njit(cache=True, nogil=True)
    def _route_flat_leaf_nb(feat, thr, left, right, X):
        """Terminal flat-tree node per row.

        The proof-path memory needs the certified region identity rather than
        its numeric leaf value. Keeping this companion kernel separate from
        ``_route_flat_nb`` preserves the hot prediction ABI and lets the
        auxiliary index reuse the exact same ``<=`` routing semantics.
        """
        n = X.shape[0]
        out = np.empty(n, np.int64)
        for i in range(n):
            nd = 0
            while feat[nd] >= 0:
                nd = left[nd] if X[i, feat[nd]] <= thr[nd] else right[nd]
            out[i] = nd
        return out

    @njit(cache=True, nogil=True)
    def _route_flat_affine_nb(feat, thr, left, right, val, lin_feat, lin_coef, lin_lo, lin_hi, X):
        """Route one numeric affine-leaf tree without Python leaf recursion.

        ``lin_*`` is a sparse, fixed-width representation: every terminal node
        carries its intercept in ``val`` plus only the path-constrained linear
        coefficients that are nonzero. This keeps serving work proportional to
        tree depth rather than feature width.
        """
        n = X.shape[0]
        width = lin_feat.shape[1]
        out = np.empty(n)
        for i in range(n):
            nd = 0
            while feat[nd] >= 0:
                nd = left[nd] if X[i, feat[nd]] <= thr[nd] else right[nd]
            score = val[nd]
            for slot in range(width):
                feature = lin_feat[nd, slot]
                if feature >= 0:
                    x = X[i, feature]
                    if x < lin_lo[nd, slot]:
                        x = lin_lo[nd, slot]
                    elif x > lin_hi[nd, slot]:
                        x = lin_hi[nd, slot]
                    score += x * lin_coef[nd, slot]
            out[i] = score
        return out

    @njit(cache=True, nogil=True)
    def _forest_scores_flat_nb(feat, thr, left, right, val, starts, stage_class, base, lr, X):
        """Evaluate a packed numeric forest in one compiled serving pass.

        The old path crossed Python/Numba once per tree. Packing tree arrays
        preserves every exact route and stage order while crossing that boundary
        once per request, which matters most for singleton tabular queries.
        """
        n = X.shape[0]
        classes = base.shape[0]
        out = np.empty((n, classes))
        for row in range(n):
            for cls in range(classes):
                out[row, cls] = base[cls]
            for stage in range(starts.shape[0]):
                start = starts[stage]
                nd = start
                while feat[nd] >= 0:
                    child = left[nd] if X[row, feat[nd]] <= thr[nd] else right[nd]
                    nd = start + child
                out[row, stage_class[stage]] += lr * val[nd]
        return out

    @njit(cache=True, nogil=True)
    def _forest_scores_binary_mirrored_nb(feat, thr, left, right, val, starts, base, lr, X):
        """Score paired binary proof trees by routing each shared partition once."""
        n = X.shape[0]
        out = np.empty((n, 2))
        for row in range(n):
            out[row, 0] = base[0]
            out[row, 1] = base[1]
            for stage in range(starts.shape[0]):
                start = starts[stage]
                nd = start
                while feat[nd] >= 0:
                    child = left[nd] if X[row, feat[nd]] <= thr[nd] else right[nd]
                    nd = start + child
                contribution = lr * val[nd]
                out[row, 0] += contribution
                out[row, 1] -= contribution
        return out

    @njit(cache=True, nogil=True)
    def _forest_scores_multiclass_shared_nb(feat, thr, left, right, val, starts, base, lr, X):
        """Score vector-leaf rounds by routing each shared multiclass partition once."""
        n = X.shape[0]
        classes = base.shape[0]
        out = np.empty((n, classes))
        for row in range(n):
            for cls in range(classes):
                out[row, cls] = base[cls]
            for stage in range(starts.shape[0]):
                start = starts[stage]
                nd = start
                while feat[nd] >= 0:
                    child = left[nd] if X[row, feat[nd]] <= thr[nd] else right[nd]
                    nd = start + child
                for cls in range(classes):
                    out[row, cls] += lr * val[nd, cls]
        return out

    @njit(cache=True, nogil=True)
    def _forest_scores_affine_nb(
        feat, thr, left, right, val, starts, stage_class, lin_feat, lin_coef, lin_lo, lin_hi, base, lr, X
    ):
        """Packed-forest scorer for numeric path-constrained affine leaves."""
        n = X.shape[0]
        classes = base.shape[0]
        width = lin_feat.shape[1]
        out = np.empty((n, classes))
        for row in range(n):
            for cls in range(classes):
                out[row, cls] = base[cls]
            for stage in range(starts.shape[0]):
                start = starts[stage]
                nd = start
                while feat[nd] >= 0:
                    child = left[nd] if X[row, feat[nd]] <= thr[nd] else right[nd]
                    nd = start + child
                score = val[nd]
                for slot in range(width):
                    feature = lin_feat[nd, slot]
                    if feature >= 0:
                        x = X[row, feature]
                        if x < lin_lo[nd, slot]:
                            x = lin_lo[nd, slot]
                        elif x > lin_hi[nd, slot]:
                            x = lin_hi[nd, slot]
                        score += x * lin_coef[nd, slot]
                out[row, stage_class[stage]] += lr * score
        return out

    @njit(cache=True, nogil=True)
    def _certified_suffix_winner_nb(
        scores,
        suffix_min,
        suffix_max,
        suffix_abs,
        suffix_count,
        suffix_index,
    ):
        """Return a class only when every possible remaining leaf keeps it ahead."""
        classes = scores.shape[0]
        candidate = 0
        best_lower = -np.inf
        for cls in range(classes):
            # Cover both the accumulated prefix and the independently summed
            # suffix bounds. The factor of two also bounds a large base score
            # hidden by cancellation: |base| <= |score| + prefix_abs.
            magnitude = abs(scores[cls]) + 2.0 * suffix_abs[0, cls]
            roundoff = 8.0 * (suffix_count[0, cls] + 2.0) * np.finfo(np.float64).eps * max(1.0, magnitude)
            lower = scores[cls] + suffix_min[suffix_index, cls] - roundoff
            if lower > best_lower:
                candidate = cls
                best_lower = lower
        for cls in range(classes):
            if cls == candidate:
                continue
            magnitude = abs(scores[cls]) + 2.0 * suffix_abs[0, cls]
            roundoff = 8.0 * (suffix_count[0, cls] + 2.0) * np.finfo(np.float64).eps * max(1.0, magnitude)
            upper = scores[cls] + suffix_max[suffix_index, cls] + roundoff
            if best_lower <= upper:
                return -1
        return candidate

    @njit(cache=True, nogil=True)
    def _forest_predict_adaptive_flat_nb(
        feat,
        thr,
        left,
        right,
        val,
        starts,
        stage_class,
        base,
        lr,
        suffix_min,
        suffix_max,
        suffix_abs,
        suffix_count,
        checkpoint_stride,
        X,
    ):
        """Predict a generic packed forest with exact suffix-certified exits."""
        rows = X.shape[0]
        classes = base.shape[0]
        stages = starts.shape[0]
        prediction = np.empty(rows, np.int64)
        evaluated = np.full(rows, stages, np.int64)
        for row in range(rows):
            scores = np.empty(classes)
            for cls in range(classes):
                scores[cls] = base[cls]
            decided = False
            for stage in range(stages):
                start = starts[stage]
                node = start
                while feat[node] >= 0:
                    child = left[node] if X[row, feat[node]] <= thr[node] else right[node]
                    node = start + child
                scores[stage_class[stage]] += lr * val[node]
                completed = stage + 1
                if completed < stages and completed % checkpoint_stride == 0:
                    winner = _certified_suffix_winner_nb(
                        scores,
                        suffix_min,
                        suffix_max,
                        suffix_abs,
                        suffix_count,
                        completed,
                    )
                    if winner >= 0:
                        prediction[row] = winner
                        evaluated[row] = completed
                        decided = True
                        break
            if not decided:
                winner = 0
                for cls in range(1, classes):
                    if scores[cls] > scores[winner]:
                        winner = cls
                prediction[row] = winner
        return prediction, evaluated

    @njit(cache=True, nogil=True)
    def _forest_predict_adaptive_binary_mirrored_nb(
        feat,
        thr,
        left,
        right,
        val,
        starts,
        base,
        lr,
        suffix_min,
        suffix_max,
        suffix_abs,
        suffix_count,
        checkpoint_stride,
        X,
    ):
        """Predict mirrored binary rounds while routing each shared tree once."""
        rows = X.shape[0]
        stages = starts.shape[0]
        prediction = np.empty(rows, np.int64)
        evaluated = np.full(rows, stages, np.int64)
        for row in range(rows):
            scores = np.empty(2)
            scores[0] = base[0]
            scores[1] = base[1]
            decided = False
            for stage in range(stages):
                start = starts[stage]
                node = start
                while feat[node] >= 0:
                    child = left[node] if X[row, feat[node]] <= thr[node] else right[node]
                    node = start + child
                contribution = lr * val[node]
                scores[0] += contribution
                scores[1] -= contribution
                completed = stage + 1
                if completed < stages and completed % checkpoint_stride == 0:
                    winner = _certified_suffix_winner_nb(
                        scores,
                        suffix_min,
                        suffix_max,
                        suffix_abs,
                        suffix_count,
                        completed,
                    )
                    if winner >= 0:
                        prediction[row] = winner
                        evaluated[row] = completed
                        decided = True
                        break
            if not decided:
                prediction[row] = 0 if scores[0] >= scores[1] else 1
        return prediction, evaluated

    @njit(cache=True, nogil=True)
    def _forest_predict_adaptive_multiclass_shared_nb(
        feat,
        thr,
        left,
        right,
        val,
        starts,
        base,
        lr,
        suffix_min,
        suffix_max,
        suffix_abs,
        suffix_count,
        checkpoint_stride,
        X,
    ):
        """Predict vector-leaf rounds with exact suffix-certified exits."""
        rows = X.shape[0]
        classes = base.shape[0]
        stages = starts.shape[0]
        prediction = np.empty(rows, np.int64)
        evaluated = np.full(rows, stages, np.int64)
        for row in range(rows):
            scores = np.empty(classes)
            for cls in range(classes):
                scores[cls] = base[cls]
            decided = False
            for stage in range(stages):
                start = starts[stage]
                node = start
                while feat[node] >= 0:
                    child = left[node] if X[row, feat[node]] <= thr[node] else right[node]
                    node = start + child
                for cls in range(classes):
                    scores[cls] += lr * val[node, cls]
                completed = stage + 1
                if completed < stages and completed % checkpoint_stride == 0:
                    winner = _certified_suffix_winner_nb(
                        scores,
                        suffix_min,
                        suffix_max,
                        suffix_abs,
                        suffix_count,
                        completed,
                    )
                    if winner >= 0:
                        prediction[row] = winner
                        evaluated[row] = completed
                        decided = True
                        break
            if not decided:
                winner = 0
                for cls in range(1, classes):
                    if scores[cls] > scores[winner]:
                        winner = cls
                prediction[row] = winner
        return prediction, evaluated

    @njit(cache=True, nogil=True)
    def _tree_interval_flat_nb(feat, thr, left, right, val, lo, hi):
        """EXACT [min,max] leaf value reachable in ONE flat tree over the box {lo[j] < x_j <= hi[j]} — the
        compiled iterative form of the `_tree_interval` recursion (the per-row certificate hot primitive, was
        ~40% of certificate time as pure-Python recursion). Explicit node-index stack: at a split, recurse the
        one reachable child, or push BOTH when the box straddles the threshold; union leaf values. Numerically
        identical to the recursion (same values, same min/max)."""
        n = feat.shape[0]
        stack = np.empty(n + 1, np.int64)
        stack[0] = 0
        sp = 1
        vmin = 1e18
        vmax = -1e18
        while sp > 0:
            sp -= 1
            nd = stack[sp]
            f = feat[nd]
            if f < 0:  # leaf
                v = val[nd]
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
                continue
            t = thr[nd]
            if hi[f] <= t:  # box entirely on the '<=' side
                stack[sp] = left[nd]
                sp += 1
            elif lo[f] >= t:  # box entirely on the '>' side
                stack[sp] = right[nd]
                sp += 1
            else:  # straddles -> both children reachable
                stack[sp] = left[nd]
                sp += 1
                stack[sp] = right[nd]
                sp += 1
        return vmin, vmax

    def _scan_feature(Gh, Hh, Ch, maxk, n, G, H, parent, min_leaf, lam, NB):
        """Best split for one feature's histogram (inlined by numba). Returns (gain, split-bin)."""
        gl = 0.0
        hl = 0.0
        nl = 0.0
        bg = 1e-12
        bk = -1
        for b in range(NB - 1):
            gl += Gh[b]
            hl += Hh[b]
            nl += Ch[b]
            if b >= maxk:
                continue
            nr = n - nl
            if nl < min_leaf or nr < min_leaf:
                continue
            gr = G - gl
            hr = H - hl
            gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent
            if gain > bg:
                bg = gain
                bk = b
        return bg, bk

    _scan_feature = njit(cache=True, fastmath=True, nogil=True)(_scan_feature)

    @njit(cache=True, fastmath=True, nogil=True)  # SERIAL: one pass over rows, 2D histogram
    def _hist_best_split_serial(Xb, g, h, feats, edge_lens, min_leaf, lam, NB, G, H):
        F = feats.shape[0]
        n = g.shape[0]
        parent = G * G / (H + lam)
        Gh = np.zeros((F, NB))
        Hh = np.zeros((F, NB))
        Ch = np.zeros((F, NB))
        for i in range(n):
            gi = g[i]
            hi = h[i]
            for fi in range(F):
                b = Xb[i, feats[fi]]
                Gh[fi, b] += gi
                Hh[fi, b] += hi
                Ch[fi, b] += 1.0
        best_gain = 1e-12
        best_f = -1
        best_k = -1
        for fi in range(F):
            bg, bk = _scan_feature(Gh[fi], Hh[fi], Ch[fi], edge_lens[fi], n, G, H, parent, min_leaf, lam, NB)
            if bk >= 0 and bg > best_gain:
                best_gain = bg
                best_f = fi
                best_k = bk
        return best_gain, best_f, best_k

    @njit(
        parallel=True, cache=True, fastmath=True, nogil=True
    )  # PARALLEL: one thread per feature (big nodes)
    def _hist_best_split_parallel(Xb, g, h, feats, edge_lens, min_leaf, lam, NB, G, H):
        F = feats.shape[0]
        n = g.shape[0]
        parent = G * G / (H + lam)
        fgain = np.full(F, 1e-12)
        fk = np.full(F, -1, dtype=np.int64)
        for fi in prange(F):  # parallel over features (thread-local hist)
            col = feats[fi]
            Gh = np.zeros(NB)
            Hh = np.zeros(NB)
            Ch = np.zeros(NB)
            for i in range(n):
                b = Xb[i, col]
                Gh[b] += g[i]
                Hh[b] += h[i]
                Ch[b] += 1.0
            fgain[fi], fk[fi] = _scan_feature(Gh, Hh, Ch, edge_lens[fi], n, G, H, parent, min_leaf, lam, NB)
        best_gain = 1e-12
        best_f = -1
        best_k = -1  # serial reduction
        for fi in range(F):
            if fk[fi] >= 0 and fgain[fi] > best_gain:
                best_gain = fgain[fi]
                best_f = fi
                best_k = fk[fi]
        return best_gain, best_f, best_k

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _build_histogram_parallel(Xb, g, h, feats, ridx, s, e, Gh, Hh, Ch):
        """Build a node histogram in parallel over features.

        Each worker visits its rows in the original stable order.  That keeps
        the per-(feature, bin) floating-point accumulation order identical to
        the serial grower while using all available cores for wide, large nodes.
        """
        F = feats.shape[0]
        NB = Gh.shape[1]
        for fi in prange(F):
            for b in range(NB):
                Gh[fi, b] = 0.0
                Hh[fi, b] = 0.0
                Ch[fi, b] = 0.0
            col = feats[fi]
            for ii in range(s, e):
                r = ridx[ii]
                b = Xb[r, col]
                Gh[fi, b] += g[r]
                Hh[fi, b] += h[r]
                Ch[fi, b] += 1.0

    @njit(cache=True, nogil=True)
    def _gather_binary_tree_fit_data_nb(Xb, probability, target_zero, weight, rows, feats):
        """Gather only sampled rows/features and form their exact Newton terms."""
        n_rows = rows.shape[0]
        n_feats = feats.shape[0]
        Xout = np.empty((n_rows, n_feats), dtype=Xb.dtype)
        gout = np.empty(n_rows, dtype=np.float64)
        hout = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            row = rows[i]
            p = probability[row]
            w = weight[row]
            gout[i] = w * (p - (1.0 if target_zero[row] else 0.0))
            hout[i] = w * p * (1.0 - p)
            for j in range(n_feats):
                Xout[i, j] = Xb[row, feats[j]]
        return Xout, gout, hout

    @njit(cache=True, fastmath=True, nogil=True)
    def _grow_2nd_binned_nb(  # noqa: C901 - compiled tree-growth kernel
        Xb,
        g,
        h,
        feats,
        edge_lens,
        cat_codes,
        cat_feats,
        cat_levels,
        max_cat,
        depth,
        min_leaf,
        cat_min_leaf,
        lam,
        NB,
        mono,
        use_mono,
        honest_cat,
        honest_seed,
        parallel_hist,
    ):
        """Grow the whole 2nd-order histogram tree ITERATIVELY in numba: an explicit stack over ranges of a
        row-index array `ridx`, partitioned STABLY (left rows then right rows, original order preserved) at
        each split. HISTOGRAM SUBTRACTION: a node's per-feature histogram is either built from its rows or,
        when both children of a split will themselves be scanned, only the SMALLER child is built from its
        rows and the LARGER is derived by subtracting it from the parent's histogram (parent = left + right
        per bin) — roughly halving the O(rows·F) accumulation, the dominant grow cost. Histograms live in a
        slot pool indexed by the stack position (LIFO DFS keeps ≤ depth+1 slots live). NOT bit-identical to a
        pure from-scratch build — float subtraction differs from a fresh sum in the last ULPs, which can flip
        a split on an exact tie (the kernel already uses fastmath, so it was never bit-frozen); validated to
        PRESERVE accuracy across the arena. Returns flat node arrays, per-row leaf preds, node count."""
        n = g.shape[0]
        F = feats.shape[0]
        CF = cat_feats.shape[0]
        cap = 1 << (depth + 1)  # ≥ 2^(depth+1)-1 max nodes of a depth-`depth` binary tree
        feat = np.full(cap, -1, np.int64)
        bink = np.zeros(cap, np.int64)
        catg = np.full(cap, -1, np.int64)
        catmask = np.zeros(cap, np.uint64)
        left = np.full(cap, -1, np.int64)
        right = np.full(cap, -1, np.int64)
        val = np.zeros(cap)
        preds = np.empty(n)
        ridx = np.arange(n, dtype=np.int32)
        tmp = np.empty(n, np.int32)
        HB = depth + 3  # stack/hist slots: LIFO DFS keeps sp ≤ depth+1 (proven); +margin for safety
        st_node = np.empty(HB, np.int64)
        st_s = np.empty(HB, np.int64)
        st_e = np.empty(HB, np.int64)
        st_lo = np.empty(HB)
        st_hi = np.empty(HB)
        st_d = np.empty(HB, np.int64)
        st_hasH = np.zeros(
            HB, np.int64
        )  # 1 -> hist[slot] was pre-built by the parent (subtraction), skip rebuild
        Gp = np.empty((HB, F, NB))  # per-slot histogram pool (G / H / count)
        Hp = np.empty((HB, F, NB))
        Cp = np.empty((HB, F, NB))
        Gs = np.empty((F, NB))  # scratch: the SMALLER child's freshly-built histogram
        Hs = np.empty((F, NB))
        Cs = np.empty((F, NB))
        st_node[0], st_s[0], st_e[0] = 0, 0, n
        st_lo[0], st_hi[0], st_d[0] = -1e18, 1e18, depth
        st_hasH[0] = 0  # root has no parent -> built from scratch
        sp = 1
        nnodes = 1
        while sp > 0:
            sp -= 1
            node = st_node[sp]
            s = st_s[sp]
            e = st_e[sp]
            lov = st_lo[sp]
            hiv = st_hi[sp]
            d = st_d[sp]
            hasH = st_hasH[sp]
            G = 0.0
            H = 0.0
            for ii in range(s, e):
                r = ridx[ii]
                G += g[r]
                H += h[r]
            v = -G / (H + lam)
            if v < lov:
                v = lov
            if v > hiv:
                v = hiv
            val[node] = v
            cnt = e - s
            if d == 0 or (cnt < 2 * min_leaf and (CF == 0 or cnt < 2 * cat_min_leaf)):
                for ii in range(s, e):
                    preds[ridx[ii]] = v
                continue
            if (
                hasH == 0
            ):  # build this node's histogram from its rows (root, or a child a leaf-sibling blocked)
                if parallel_hist and cnt * F >= _PARALLEL_HIST_NODE_MIN_WORK:
                    _build_histogram_parallel(Xb, g, h, feats, ridx, s, e, Gp[sp], Hp[sp], Cp[sp])
                else:
                    for fi in range(F):
                        for b in range(NB):
                            Gp[sp, fi, b] = 0.0
                            Hp[sp, fi, b] = 0.0
                            Cp[sp, fi, b] = 0.0
                    for ii in range(s, e):
                        r = ridx[ii]
                        gr = g[r]
                        hr = h[r]
                        for fi in range(F):
                            b = Xb[r, feats[fi]]
                            Gp[sp, fi, b] += gr
                            Hp[sp, fi, b] += hr
                            Cp[sp, fi, b] += 1.0
            parent = G * G / (H + lam)
            best_gain = 1e-12
            best_f = -1
            best_k = -1
            for fi in range(F):
                bg, bk = _scan_feature(
                    Gp[sp, fi], Hp[sp, fi], Cp[sp, fi], edge_lens[fi], cnt, G, H, parent, min_leaf, lam, NB
                )
                if bk >= 0 and bg > best_gain:
                    best_gain = bg
                    best_f = fi
                    best_k = bk
            best_cat = -1
            best_cat_mask = np.uint64(0)
            # Native category split. The direct path preserves the ordinary
            # ordered-partition scan. The research path builds the level order
            # and prefix on a deterministic proposal half, then scores that
            # one fixed set on the complementary half. This removes the
            # category-set search's target reuse without changing the numeric
            # splitter or the public default.
            for cfi in range(CF):
                ci = cat_feats[cfi]
                L = cat_levels[ci]
                if L < 2:
                    continue
                if not honest_cat:
                    cg = np.zeros(max_cat)
                    ch = np.zeros(max_cat)
                    cn = np.zeros(max_cat)
                    for ii in range(s, e):
                        r = ridx[ii]
                        lev = cat_codes[r, ci]
                        if lev >= 0 and lev < L:
                            cg[lev] += g[r]
                            ch[lev] += h[r]
                            cn[lev] += 1.0
                    order = np.empty(L, np.int64)
                    for q in range(L):
                        order[q] = q
                    # Stable insertion sort: ties break by source level index, preserving determinism.
                    for q in range(1, L):
                        key = order[q]
                        ks = -cg[key] / (ch[key] + lam)
                        p = q - 1
                        while p >= 0:
                            prev = order[p]
                            ps = -cg[prev] / (ch[prev] + lam)
                            if ks < ps or (ks == ps and key < prev):
                                order[p + 1] = prev
                                p -= 1
                            else:
                                break
                        order[p + 1] = key
                    gl = 0.0
                    hl = 0.0
                    nl = 0.0
                    mask = np.uint64(0)
                    for q in range(L):
                        lev = order[q]
                        if cn[lev] <= 0.0:
                            continue
                        gl += cg[lev]
                        hl += ch[lev]
                        nl += cn[lev]
                        mask = mask | (np.uint64(1) << np.uint64(lev))
                        nr = cnt - nl
                        if nl < cat_min_leaf or nr < cat_min_leaf:
                            continue
                        gain = gl * gl / (hl + lam) + (G - gl) * (G - gl) / (H - hl + lam) - parent
                        if gain > best_gain:
                            best_gain = gain
                            best_cat = ci
                            best_cat_mask = mask
                    continue

                # Honest category-set proposal: the deterministic hash is a
                # function only of row identity, node and tree seed. It never
                # reads a gradient, label or candidate score. Evaluation uses
                # the same row split for every categorical feature at a node.
                pg = np.zeros(max_cat)
                ph = np.zeros(max_cat)
                pn = np.zeros(max_cat)
                eg = np.zeros(max_cat)
                eh = np.zeros(max_cat)
                en = np.zeros(max_cat)
                pG = 0.0
                pH = 0.0
                eG = 0.0
                eH = 0.0
                ptotal = 0.0
                etotal = 0.0
                for ii in range(s, e):
                    r = ridx[ii]
                    # Murmur-style integer mix: stable across processes and
                    # intentionally independent of category level / residual.
                    z = np.uint64(r + 1)
                    z = z ^ (np.uint64(honest_seed + 1) * np.uint64(0x9E3779B1))
                    z = z ^ (np.uint64(node + 1) * np.uint64(0x85EBCA77))
                    z = z ^ (z >> np.uint64(16))
                    z = z * np.uint64(0x7FEB352D)
                    z = z ^ (z >> np.uint64(15))
                    lev = cat_codes[r, ci]
                    if (z & np.uint64(1)) == np.uint64(0):
                        pG += g[r]
                        pH += h[r]
                        ptotal += 1.0
                        if lev >= 0 and lev < L:
                            pg[lev] += g[r]
                            ph[lev] += h[r]
                            pn[lev] += 1.0
                    else:
                        eG += g[r]
                        eH += h[r]
                        etotal += 1.0
                        if lev >= 0 and lev < L:
                            eg[lev] += g[r]
                            eh[lev] += h[r]
                            en[lev] += 1.0
                part_leaf = max(1, (cat_min_leaf + 1) // 2)
                if ptotal < 2.0 * part_leaf or etotal < 2.0 * part_leaf:
                    continue
                half_lam = 0.5 * lam
                pparent = pG * pG / (pH + half_lam)
                eparent = eG * eG / (eH + half_lam)
                order = np.empty(L, np.int64)
                for q in range(L):
                    order[q] = q
                # The level order is proposed from proposal gradients only.
                for q in range(1, L):
                    key = order[q]
                    ks = -pg[key] / (ph[key] + half_lam)
                    p = q - 1
                    while p >= 0:
                        prev = order[p]
                        ps = -pg[prev] / (ph[prev] + half_lam)
                        if ks < ps or (ks == ps and key < prev):
                            order[p + 1] = prev
                            p -= 1
                        else:
                            break
                    order[p + 1] = key
                pgl = 0.0
                phl = 0.0
                pnl = 0.0
                egl = 0.0
                ehl = 0.0
                enl = 0.0
                proposal_gain = 1e-12
                proposal_mask = np.uint64(0)
                current_mask = np.uint64(0)
                proposal_eg = 0.0
                proposal_eh = 0.0
                for q in range(L):
                    lev = order[q]
                    # A level unseen by the proposer cannot contribute to a
                    # proposed finite set; it deterministically stays right.
                    if pn[lev] <= 0.0:
                        continue
                    pgl += pg[lev]
                    phl += ph[lev]
                    pnl += pn[lev]
                    egl += eg[lev]
                    ehl += eh[lev]
                    enl += en[lev]
                    current_mask = current_mask | (np.uint64(1) << np.uint64(lev))
                    pnr = ptotal - pnl
                    enr = etotal - enl
                    fullnl = pnl + enl
                    fullnr = cnt - fullnl
                    # Evaluation counts are structural (not label-derived),
                    # so they may ensure a valid partition without spending
                    # evaluation gradients to choose among category sets.
                    if (
                        pnl < part_leaf
                        or pnr < part_leaf
                        or enl < part_leaf
                        or enr < part_leaf
                        or fullnl < cat_min_leaf
                        or fullnr < cat_min_leaf
                    ):
                        continue
                    gain = (
                        pgl * pgl / (phl + half_lam)
                        + (pG - pgl) * (pG - pgl) / (pH - phl + half_lam)
                        - pparent
                    )
                    if gain > proposal_gain:
                        proposal_gain = gain
                        proposal_mask = current_mask
                        proposal_eg = egl
                        proposal_eh = ehl
                if proposal_gain <= 1e-12:
                    continue
                # One fixed proposal is scored from the held-out gradients.
                # Scaling by two puts the half-sample objective on the full
                # data scale used by the numeric split search above.
                eval_gain = 2.0 * (
                    proposal_eg * proposal_eg / (proposal_eh + half_lam)
                    + (eG - proposal_eg) * (eG - proposal_eg) / (eH - proposal_eh + half_lam)
                    - eparent
                )
                if eval_gain > best_gain:
                    best_gain = eval_gain
                    best_cat = ci
                    best_cat_mask = proposal_mask
            is_cat = best_cat >= 0
            if best_f < 0 and not is_cat:
                for ii in range(s, e):
                    preds[ridx[ii]] = v
                continue
            j = -1
            k = -1
            if not is_cat:
                j = feats[best_f]
                k = best_k
            nl = 0  # STABLE partition by the chosen numeric bin or categorical set membership.
            for ii in range(s, e):
                r = ridx[ii]
                go_left = (
                    Xb[r, j] <= k
                    if not is_cat
                    else (
                        cat_codes[r, best_cat] >= 0
                        and ((best_cat_mask >> np.uint64(cat_codes[r, best_cat])) & np.uint64(1)) != 0
                    )
                )
                if go_left:
                    nl += 1
            lp = s
            rp = s + nl
            for ii in range(s, e):
                r = ridx[ii]
                go_left = (
                    Xb[r, j] <= k
                    if not is_cat
                    else (
                        cat_codes[r, best_cat] >= 0
                        and ((best_cat_mask >> np.uint64(cat_codes[r, best_cat])) & np.uint64(1)) != 0
                    )
                )
                if go_left:
                    tmp[lp] = r
                    lp += 1
                else:
                    tmp[rp] = r
                    rp += 1
            for ii in range(s, e):
                ridx[ii] = tmp[ii]
            mid = s + nl
            loL = lov
            hiL = hiv
            loR = lov
            hiR = hiv
            if not is_cat and use_mono and mono[j] != 0:  # only ordered numeric features admit monotonicity
                GL = 0.0
                HL = 0.0
                for ii in range(s, mid):
                    r = ridx[ii]
                    GL += g[r]
                    HL += h[r]
                wl = -GL / (HL + lam)
                if wl < lov:
                    wl = lov
                if wl > hiv:
                    wl = hiv
                wr = -(G - GL) / ((H - HL) + lam)
                if wr < lov:
                    wr = lov
                if wr > hiv:
                    wr = hiv
                midv = 0.5 * (wl + wr)
                if mono[j] > 0:
                    hiL = midv
                    loR = midv
                else:
                    loL = midv
                    hiR = midv
            if is_cat:
                feat[node] = -2
                catg[node] = best_cat
                catmask[node] = best_cat_mask
            else:
                feat[node] = j
                bink[node] = k
            lch = nnodes
            rch = nnodes + 1
            nnodes += 2
            left[node] = lch
            right[node] = rch
            # children occupy stack slots sp (left) and sp+1 (right). If BOTH will be scanned further, build only
            # the smaller from its rows and derive the larger from the parent's histogram (slot sp) by subtraction.
            cntL = mid - s
            cntR = e - mid
            d1 = d - 1
            build_l = (d1 > 0) and (cntL >= 2 * min_leaf)
            build_r = (d1 > 0) and (cntR >= 2 * min_leaf)
            has_l = 0
            has_r = 0
            if build_l and build_r:
                if (
                    cntL <= cntR
                ):  # smaller = left -> scratch; right (larger) = parent - scratch into slot sp+1
                    if parallel_hist and cntL * F >= _PARALLEL_HIST_NODE_MIN_WORK:
                        _build_histogram_parallel(Xb, g, h, feats, ridx, s, mid, Gs, Hs, Cs)
                    else:
                        for fi in range(F):
                            for b in range(NB):
                                Gs[fi, b] = 0.0
                                Hs[fi, b] = 0.0
                                Cs[fi, b] = 0.0
                        for ii in range(s, mid):
                            r = ridx[ii]
                            gr = g[r]
                            hr = h[r]
                            for fi in range(F):
                                b = Xb[r, feats[fi]]
                                Gs[fi, b] += gr
                                Hs[fi, b] += hr
                                Cs[fi, b] += 1.0
                    for fi in range(
                        F
                    ):  # right (slot sp+1) FIRST (reads parent in slot sp), then left overwrites sp
                        for b in range(NB):
                            Gp[sp + 1, fi, b] = Gp[sp, fi, b] - Gs[fi, b]
                            Hp[sp + 1, fi, b] = Hp[sp, fi, b] - Hs[fi, b]
                            Cp[sp + 1, fi, b] = Cp[sp, fi, b] - Cs[fi, b]
                    for fi in range(F):
                        for b in range(NB):
                            Gp[sp, fi, b] = Gs[fi, b]
                            Hp[sp, fi, b] = Hs[fi, b]
                            Cp[sp, fi, b] = Cs[fi, b]
                else:  # smaller = right -> scratch; left (larger) = parent - scratch in place in slot sp
                    if parallel_hist and cntR * F >= _PARALLEL_HIST_NODE_MIN_WORK:
                        _build_histogram_parallel(Xb, g, h, feats, ridx, mid, e, Gs, Hs, Cs)
                    else:
                        for fi in range(F):
                            for b in range(NB):
                                Gs[fi, b] = 0.0
                                Hs[fi, b] = 0.0
                                Cs[fi, b] = 0.0
                        for ii in range(mid, e):
                            r = ridx[ii]
                            gr = g[r]
                            hr = h[r]
                            for fi in range(F):
                                b = Xb[r, feats[fi]]
                                Gs[fi, b] += gr
                                Hs[fi, b] += hr
                                Cs[fi, b] += 1.0
                    for fi in range(
                        F
                    ):  # right (slot sp+1) = scratch; left (slot sp) = parent - scratch in place
                        for b in range(NB):
                            Gp[sp + 1, fi, b] = Gs[fi, b]
                            Hp[sp + 1, fi, b] = Hs[fi, b]
                            Cp[sp + 1, fi, b] = Cs[fi, b]
                            Gp[sp, fi, b] = Gp[sp, fi, b] - Gs[fi, b]
                            Hp[sp, fi, b] = Hp[sp, fi, b] - Hs[fi, b]
                            Cp[sp, fi, b] = Cp[sp, fi, b] - Cs[fi, b]
                has_l = 1
                has_r = 1
            st_node[sp] = lch
            st_s[sp] = s
            st_e[sp] = mid
            st_lo[sp] = loL
            st_hi[sp] = hiL
            st_d[sp] = d1
            st_hasH[sp] = has_l
            sp += 1
            st_node[sp] = rch
            st_s[sp] = mid
            st_e[sp] = e
            st_lo[sp] = loR
            st_hi[sp] = hiR
            st_d[sp] = d1
            st_hasH[sp] = has_r
            sp += 1
        return feat, bink, catg, catmask, left, right, val, preds, nnodes

    @njit(cache=True, fastmath=True, nogil=True)
    def _grow_softmax_shared_binned_nb(  # noqa: C901 - compiled tree-growth kernel
        Xb, g, h, feats, edge_lens, depth, min_leaf, lam, NB
    ):
        """Grow one numeric partition with a Newton leaf vector for every class."""
        n = g.shape[0]
        K = g.shape[1]
        F = feats.shape[0]
        cap = 1 << (depth + 1)
        feat = np.full(cap, -1, np.int64)
        bink = np.zeros(cap, np.int64)
        left = np.full(cap, -1, np.int64)
        right = np.full(cap, -1, np.int64)
        val = np.zeros((cap, K))
        preds = np.empty((n, K))
        ridx = np.arange(n, dtype=np.int32)
        tmp = np.empty(n, dtype=np.int32)
        stack_cap = depth + 3
        st_node = np.empty(stack_cap, np.int64)
        st_start = np.empty(stack_cap, np.int64)
        st_stop = np.empty(stack_cap, np.int64)
        st_depth = np.empty(stack_cap, np.int64)
        st_node[0] = 0
        st_start[0] = 0
        st_stop[0] = n
        st_depth[0] = depth
        stack_size = 1
        nodes = 1
        Gh = np.zeros((F, NB, K))
        Hh = np.zeros((F, NB, K))
        Ch = np.zeros((F, NB))
        total_g = np.zeros(K)
        total_h = np.zeros(K)
        left_g = np.zeros(K)
        left_h = np.zeros(K)
        while stack_size > 0:
            stack_size -= 1
            node = st_node[stack_size]
            start = st_start[stack_size]
            stop = st_stop[stack_size]
            remaining_depth = st_depth[stack_size]
            count = stop - start
            for cls in range(K):
                total_g[cls] = 0.0
                total_h[cls] = 0.0
            for ii in range(start, stop):
                row = ridx[ii]
                for cls in range(K):
                    total_g[cls] += g[row, cls]
                    total_h[cls] += h[row, cls]
            mean_value = 0.0
            for cls in range(K):
                value = -total_g[cls] / (total_h[cls] + lam)
                val[node, cls] = value
                mean_value += value
            mean_value /= K
            for cls in range(K):
                val[node, cls] -= mean_value
            if remaining_depth == 0 or count < 2 * min_leaf:
                for ii in range(start, stop):
                    row = ridx[ii]
                    for cls in range(K):
                        preds[row, cls] = val[node, cls]
                continue
            for fi in range(F):
                for bin_index in range(NB):
                    Ch[fi, bin_index] = 0.0
                    for cls in range(K):
                        Gh[fi, bin_index, cls] = 0.0
                        Hh[fi, bin_index, cls] = 0.0
            for ii in range(start, stop):
                row = ridx[ii]
                for fi in range(F):
                    bin_index = Xb[row, feats[fi]]
                    Ch[fi, bin_index] += 1.0
                    for cls in range(K):
                        Gh[fi, bin_index, cls] += g[row, cls]
                        Hh[fi, bin_index, cls] += h[row, cls]
            parent_gain = 0.0
            for cls in range(K):
                parent_gain += total_g[cls] * total_g[cls] / (total_h[cls] + lam)
            best_gain = 1e-12
            best_feature = -1
            best_bin = -1
            for fi in range(F):
                for cls in range(K):
                    left_g[cls] = 0.0
                    left_h[cls] = 0.0
                left_count = 0.0
                for bin_index in range(NB - 1):
                    left_count += Ch[fi, bin_index]
                    for cls in range(K):
                        left_g[cls] += Gh[fi, bin_index, cls]
                        left_h[cls] += Hh[fi, bin_index, cls]
                    if bin_index >= edge_lens[fi]:
                        continue
                    right_count = count - left_count
                    if left_count < min_leaf or right_count < min_leaf:
                        continue
                    gain = -parent_gain
                    for cls in range(K):
                        right_g = total_g[cls] - left_g[cls]
                        right_h = total_h[cls] - left_h[cls]
                        gain += left_g[cls] * left_g[cls] / (left_h[cls] + lam)
                        gain += right_g * right_g / (right_h + lam)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = fi
                        best_bin = bin_index
            if best_feature < 0:
                for ii in range(start, stop):
                    row = ridx[ii]
                    for cls in range(K):
                        preds[row, cls] = val[node, cls]
                continue
            source_feature = feats[best_feature]
            left_count = 0
            for ii in range(start, stop):
                if Xb[ridx[ii], source_feature] <= best_bin:
                    left_count += 1
            left_pos = start
            right_pos = start + left_count
            for ii in range(start, stop):
                row = ridx[ii]
                if Xb[row, source_feature] <= best_bin:
                    tmp[left_pos] = row
                    left_pos += 1
                else:
                    tmp[right_pos] = row
                    right_pos += 1
            for ii in range(start, stop):
                ridx[ii] = tmp[ii]
            middle = start + left_count
            feat[node] = source_feature
            bink[node] = best_bin
            left_child = nodes
            right_child = nodes + 1
            nodes += 2
            left[node] = left_child
            right[node] = right_child
            next_depth = remaining_depth - 1
            st_node[stack_size] = left_child
            st_start[stack_size] = start
            st_stop[stack_size] = middle
            st_depth[stack_size] = next_depth
            stack_size += 1
            st_node[stack_size] = right_child
            st_start[stack_size] = middle
            st_stop[stack_size] = stop
            st_depth[stack_size] = next_depth
            stack_size += 1
        return feat, bink, left, right, val, preds, nodes


def _gather_binary_tree_fit_data(Xb, probability, target_zero, weight, rows, feats):
    """Materialize one compact numeric-tree sample without full-width Newton arrays."""
    rows = np.ascontiguousarray(rows, np.int64)
    feats = np.ascontiguousarray(feats, np.int64)
    probability = np.ascontiguousarray(probability, np.float64)
    target_zero = np.ascontiguousarray(target_zero, np.bool_)
    weight = np.ascontiguousarray(weight, np.float64)
    if _HAS_NUMBA:
        return _gather_binary_tree_fit_data_nb(
            np.ascontiguousarray(Xb), probability, target_zero, weight, rows, feats
        )
    sampled_probability = probability[rows]
    sampled_weight = weight[rows]
    sampled_target = target_zero[rows]
    return (
        np.ascontiguousarray(Xb[np.ix_(rows, feats)]),
        sampled_weight * (sampled_probability - sampled_target),
        sampled_weight * sampled_probability * (1.0 - sampled_probability),
    )


def _should_parallel_hist(n_rows, n_features):
    """Whether one numeric tree has enough work to pay for a Numba thread team."""
    return bool(
        _HAS_NUMBA and int(n_features) >= 4 and int(n_rows) * int(n_features) >= _PARALLEL_HIST_TREE_MIN_WORK
    )


# --------------------------------------------------------------------------- CART primitives
def _best_split(X, r, min_leaf=20, feats=None):
    """CART split: (feature, threshold) that maximally reduces residual variance. O(d·n log n)."""
    n = len(r)
    g0 = r.sum() ** 2 / n
    best = (0.0, None, None)
    for j in range(X.shape[1]) if feats is None else feats:
        o = np.argsort(X[:, j], kind="stable")
        xs = X[o, j]
        rs = r[o]
        cs = np.cumsum(rs)
        tot = cs[-1]
        i = np.arange(1, n)
        SL = cs[:-1]
        nL = i
        SR = tot - SL
        nR = n - i
        valid = (xs[1:] > xs[:-1]) & (nL >= min_leaf) & (nR >= min_leaf)
        gain = np.where(valid, SL**2 / nL + SR**2 / nR - g0, -np.inf)
        k = int(gain.argmax())
        if gain[k] > best[0]:
            best = (float(gain[k]), j, float((xs[k] + xs[k + 1]) / 2))
    return best if best[1] is not None else None


def _fit_tree(X, r, depth=2, min_leaf=20, feats=None):
    if depth == 0 or len(r) < 2 * min_leaf:
        return ("leaf", float(r.mean()))
    s = _best_split(X, r, min_leaf, feats)
    if s is None:
        return ("leaf", float(r.mean()))
    _, j, t = s
    m = X[:, j] <= t
    return (
        "node",
        j,
        t,
        _fit_tree(X[m], r[m], depth - 1, min_leaf, feats),
        _fit_tree(X[~m], r[~m], depth - 1, min_leaf, feats),
    )


def _flatten_tree(tree):
    """Compile a tuple-tree into flat arrays (feat, thr, left, right, val). Leaves have feat=-1 and hold
    their value in `val`; internal nodes point to child indices."""
    feat, thr, left, right, val = [], [], [], [], []

    def rec(t):
        i = len(feat)
        if t[0] == "leaf":
            feat.append(-1)
            thr.append(0.0)
            left.append(-1)
            right.append(-1)
            val.append(float(t[1]))
            return i
        if t[0] == "cat":
            raise ValueError("categorical partition trees do not have a numeric flat form")
        feat.append(t[1])
        thr.append(float(t[2]))
        left.append(-1)
        right.append(-1)
        val.append(0.0)
        left[i] = rec(t[3])
        right[i] = rec(t[4])
        return i

    rec(tree)
    return (
        np.array(feat, np.int64),
        np.array(thr),
        np.array(left, np.int64),
        np.array(right, np.int64),
        np.array(val),
    )


def _cat_member(X, cols, levels):
    """Whether each encoded row belongs to a finite categorical level set.

    ``cols`` is one mutually-exclusive one-hot block and ``levels`` are offsets
    inside that block. An unseen all-zero block deterministically takes the
    right branch. The training preprocessor includes a missing-level fact, so
    this only occurs for genuinely unseen prediction-time levels.
    """
    if not levels:
        return np.zeros(len(X), dtype=bool)
    picked = np.asarray([cols[int(level)] for level in levels], dtype=int)
    return (np.asarray(X)[:, picked] > 0.5).any(1)


def _has_cat_node(tree):
    """True when a tuple tree contains a native categorical partition node."""
    if isinstance(tree[0], np.ndarray):
        return False
    stack = [tree]
    while stack:
        node = stack.pop()
        if node[0] == "cat":
            return True
        if node[0] == "node":
            stack += [node[3], node[4]]
    return False


def _tree_pred_categorical(tree, X):
    """Vectorized tuple routing for the rare trees which contain a category set.

    Numeric-only trees retain the flattened numba hot path below. A mixed tree
    is shallow, so batch-routing its live row subsets is cheaper and clearer
    than forcing a variable-size category bitset into the numeric flat ABI.
    """
    X = np.asarray(X, float)
    out = np.empty(len(X))
    stack = [(tree, np.arange(len(X), dtype=int))]
    while stack:
        node, rows = stack.pop()
        if len(rows) == 0:
            continue
        if node[0] == "leaf":
            out[rows] = float(node[1])
            continue
        if node[0] == "cat":
            _, cols, levels, left, right = node
            mask = _cat_member(X[rows], cols, levels)
        else:
            _, j, thr, left, right = node
            mask = X[rows, j] <= thr
        stack.append((right, rows[~mask]))
        stack.append((left, rows[mask]))
    return out


def _tree_pred(tree, X):
    """Leaf value per row. Accepts EITHER a nested tuple-tree OR an already-flat tree (the 5-array tuple
    `_fit_tree_2nd_binned` returns directly) — the flat form skips re-flattening, so the numba grower's output
    feeds prediction with no flat→tuple→flat round-trip. Routes in numba (per-row, no per-level array allocs)
    when available, else the vectorized numpy fallback. Dominant fit/predict path."""
    if not isinstance(tree[0], np.ndarray) and _has_cat_node(tree):
        return _tree_pred_categorical(tree, X)
    feat, thr, left, right, val = tree if isinstance(tree[0], np.ndarray) else _flatten_tree(tree)
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


def _tree_pred_rows(tree, X, rows):
    """Leaf values for ordered source rows without copying their feature matrix."""
    rows = np.ascontiguousarray(rows, np.int64)
    if not isinstance(tree[0], np.ndarray) and _has_cat_node(tree):
        return _tree_pred_categorical(tree, X[rows])
    feat, thr, left, right, val = tree if isinstance(tree[0], np.ndarray) else _flatten_tree(tree)
    if _HAS_NUMBA:
        return _route_flat_rows_nb(
            np.ascontiguousarray(feat, np.int64),
            np.ascontiguousarray(thr, np.float64),
            np.ascontiguousarray(left, np.int64),
            np.ascontiguousarray(right, np.int64),
            np.ascontiguousarray(val, np.float64),
            np.ascontiguousarray(X, np.float64),
            rows,
        )
    return _tree_pred((feat, thr, left, right, val), X[rows])


def _add_rows_in_place(scores, rows, values, scale):
    """Add ordered per-row values without constructing a boolean score slice."""
    rows = np.ascontiguousarray(rows, np.int64)
    values = np.ascontiguousarray(values, np.float64)
    if _HAS_NUMBA:
        _add_rows_nb(rows, values, scores, float(scale))
        return
    scores[rows] += float(scale) * values


def _add_flat_binned_rows_in_place(flat, Xb, rows, scores, scale):
    """Route sampled-out training rows in the existing compact bin matrix."""
    feat, bink, left, right, val = flat
    rows = np.ascontiguousarray(rows, np.int64)
    if _HAS_NUMBA:
        _add_flat_binned_rows_nb(
            np.ascontiguousarray(feat, np.int64),
            np.ascontiguousarray(bink, np.int64),
            np.ascontiguousarray(left, np.int64),
            np.ascontiguousarray(right, np.int64),
            np.ascontiguousarray(val, np.float64),
            np.ascontiguousarray(Xb),
            rows,
            scores,
            float(scale),
        )
        return
    node = np.zeros(len(rows), np.int64)
    while True:
        feature = feat[node]
        internal = feature >= 0
        if not internal.any():
            break
        active = np.flatnonzero(internal)
        source_rows = rows[active]
        nd = node[active]
        go_left = Xb[source_rows, feature[active]] <= bink[nd]
        node[active] = np.where(go_left, left[nd], right[nd])
    scores[rows] += float(scale) * val[node]


def _flat_leaf_ids(flat, X):
    """Certified terminal-node identities for a numeric flat tree.

    Flat node ids are stable within one fitted tree, so they are suitable as
    exact region facts for an auxiliary memory. Category and affine trees do
    not have this ABI and are rejected by their caller.
    """
    feat, thr, left, right, _val = flat
    X = np.asarray(X, float)
    if _HAS_NUMBA:
        return _route_flat_leaf_nb(
            np.ascontiguousarray(feat, np.int64),
            np.ascontiguousarray(thr, np.float64),
            np.ascontiguousarray(left, np.int64),
            np.ascontiguousarray(right, np.int64),
            np.ascontiguousarray(X, np.float64),
        )
    node = np.zeros(len(X), np.int64)
    while True:
        f = feat[node]
        internal = f >= 0
        if not internal.any():
            return node
        rows = np.nonzero(internal)[0]
        nd = node[rows]
        go_left = X[rows, f[rows]] <= thr[nd]
        node[rows] = np.where(go_left, left[nd], right[nd])


def _flatten_affine_tree(tree):
    """Compile one numeric tuple tree with optional affine leaves.

    The canonical tuple remains authoritative for certificates. This compact
    read-only form exists solely to make normal prediction avoid recursive
    Python routing. A categorical split has no numeric flat ABI and returns
    ``None`` so callers retain the established exact fallback.
    """
    feat, thr, left, right, val, terms = [], [], [], [], [], []

    def visit(node):
        kind = node[0]
        if kind == "cat":
            return None
        index = len(feat)
        if kind == "leaf":
            feat.append(-1)
            thr.append(0.0)
            left.append(-1)
            right.append(-1)
            val.append(float(node[1]))
            if len(node) == 2:
                terms.append(())
            else:
                _, _intercept, coef, lo, hi = node
                nz = np.flatnonzero(np.asarray(coef) != 0).astype(np.int64, copy=False)
                terms.append(tuple((int(j), float(coef[j]), float(lo[j]), float(hi[j])) for j in nz))
            return index
        _, feature, threshold, child_left, child_right = node
        feat.append(int(feature))
        thr.append(float(threshold))
        left.append(-1)
        right.append(-1)
        val.append(0.0)
        terms.append(())
        li = visit(child_left)
        if li is None:
            return None
        ri = visit(child_right)
        if ri is None:
            return None
        left[index], right[index] = li, ri
        return index

    if visit(tree) is None:
        return None
    width = max((len(part) for part in terms), default=0)
    lin_feat = np.full((len(feat), width), -1, dtype=np.int64)
    lin_coef = np.zeros((len(feat), width), dtype=float)
    lin_lo = np.zeros((len(feat), width), dtype=float)
    lin_hi = np.zeros((len(feat), width), dtype=float)
    for node, part in enumerate(terms):
        for slot, (feature, coef, lo, hi) in enumerate(part):
            lin_feat[node, slot] = feature
            lin_coef[node, slot] = coef
            lin_lo[node, slot] = lo
            lin_hi[node, slot] = hi
    return (
        np.asarray(feat, dtype=np.int64),
        np.asarray(thr, dtype=float),
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(val, dtype=float),
        lin_feat,
        lin_coef,
        lin_lo,
        lin_hi,
    )


def _affine_flat_pred(flat, X):
    """Predict an affine-flat numeric tree, preserving tuple-tree semantics."""
    feat, thr, left, right, val, lin_feat, lin_coef, lin_lo, lin_hi = flat
    X = np.asarray(X, float)
    if _HAS_NUMBA:
        return _route_flat_affine_nb(
            np.ascontiguousarray(feat, np.int64),
            np.ascontiguousarray(thr, np.float64),
            np.ascontiguousarray(left, np.int64),
            np.ascontiguousarray(right, np.int64),
            np.ascontiguousarray(val, np.float64),
            np.ascontiguousarray(lin_feat, np.int64),
            np.ascontiguousarray(lin_coef, np.float64),
            np.ascontiguousarray(lin_lo, np.float64),
            np.ascontiguousarray(lin_hi, np.float64),
            np.ascontiguousarray(X, np.float64),
        )
    out = np.empty(len(X), dtype=float)
    for row in range(len(X)):
        node = 0
        while feat[node] >= 0:
            node = left[node] if X[row, feat[node]] <= thr[node] else right[node]
        score = val[node]
        for slot, feature in enumerate(lin_feat[node]):
            if feature >= 0:
                score += (
                    np.clip(X[row, feature], lin_lo[node, slot], lin_hi[node, slot]) * lin_coef[node, slot]
                )
        out[row] = score
    return out


def _pack_flat_forest(flats, stage_class):
    """Pack numeric constant-leaf flat trees for one fused serving kernel."""
    flats = tuple(flats)
    stage_class = np.asarray(stage_class, dtype=np.int64)
    if not flats or len(flats) != len(stage_class) or any(flat is None for flat in flats):
        return None
    lengths = [len(flat[0]) for flat in flats]
    if any(length <= 0 for length in lengths):
        return None
    starts = np.empty(len(flats), dtype=np.int64)
    total = 0
    for index, length in enumerate(lengths):
        starts[index] = total
        total += length
    feat = np.empty(total, dtype=np.int64)
    thr = np.empty(total, dtype=float)
    left = np.empty(total, dtype=np.int64)
    right = np.empty(total, dtype=np.int64)
    val = np.empty(total, dtype=float)
    for start, flat in zip(starts, flats, strict=False):
        f, t, child_left, child_right, v = flat
        stop = start + len(f)
        feat[start:stop] = f
        thr[start:stop] = t
        left[start:stop] = child_left
        right[start:stop] = child_right
        val[start:stop] = v
    return feat, thr, left, right, val, starts, stage_class


def _pack_binary_mirrored_forest(flats, stage_class):
    """Pack canonical halves of exact ``(+tree, -tree)`` binary stage pairs."""
    flats = tuple(flats)
    stage_class = np.asarray(stage_class, dtype=np.int64)
    if len(flats) == 0 or len(flats) % 2 or len(flats) != len(stage_class):
        return None
    canonical = []
    for index in range(0, len(flats), 2):
        left_flat, right_flat = flats[index], flats[index + 1]
        if (
            left_flat is None
            or right_flat is None
            or stage_class[index] != 0
            or stage_class[index + 1] != 1
            or any(
                not np.array_equal(left, right)
                for left, right in zip(left_flat[:4], right_flat[:4], strict=False)
            )
            or not np.array_equal(left_flat[4], -right_flat[4])
        ):
            return None
        canonical.append(left_flat)
    packed = _pack_flat_forest(canonical, np.zeros(len(canonical), dtype=np.int64))
    return None if packed is None else packed[:6]


def _pack_multiclass_shared_forest(flats, stage_class, n_classes):
    """Pack K scalar views of each vector-leaf round into one routed stage."""
    flats = tuple(flats)
    stage_class = np.asarray(stage_class, dtype=np.int64)
    n_classes = int(n_classes)
    if n_classes < 3 or not flats or len(flats) % n_classes or len(flats) != len(stage_class):
        return None
    canonical = []
    groups = []
    expected_classes = np.arange(n_classes, dtype=np.int64)
    for start in range(0, len(flats), n_classes):
        group = flats[start : start + n_classes]
        if (
            any(flat is None for flat in group)
            or not np.array_equal(stage_class[start : start + n_classes], expected_classes)
            or any(not np.array_equal(group[0][part], flat[part]) for flat in group[1:] for part in range(4))
        ):
            return None
        canonical.append(group[0])
        groups.append(group)
    packed = _pack_flat_forest(canonical, np.zeros(len(canonical), dtype=np.int64))
    if packed is None:
        return None
    feat, thr, left, right, _canonical_value, starts = packed[:6]
    values = np.zeros((len(feat), n_classes), dtype=float)
    for forest_start, group in zip(starts, groups, strict=False):
        stop = int(forest_start) + len(group[0][0])
        for cls, flat in enumerate(group):
            values[int(forest_start) : stop, cls] = flat[4]
    return feat, thr, left, right, values, starts


def _pack_affine_forest(trees, stage_class):
    """Pack numeric tuple trees with sparse affine leaves for fused serving."""
    trees = tuple(trees)
    stage_class = np.asarray(stage_class, dtype=np.int64)
    if not trees or len(trees) != len(stage_class):
        return None
    parts = tuple(_flatten_affine_tree(tree) for tree in trees)
    if any(part is None for part in parts):
        return None
    lengths = [len(part[0]) for part in parts]
    starts = np.empty(len(parts), dtype=np.int64)
    total = 0
    for index, length in enumerate(lengths):
        starts[index] = total
        total += length
    width = max((part[5].shape[1] for part in parts), default=0)
    feat = np.empty(total, dtype=np.int64)
    thr = np.empty(total, dtype=float)
    left = np.empty(total, dtype=np.int64)
    right = np.empty(total, dtype=np.int64)
    val = np.empty(total, dtype=float)
    lin_feat = np.full((total, width), -1, dtype=np.int64)
    lin_coef = np.zeros((total, width), dtype=float)
    lin_lo = np.zeros((total, width), dtype=float)
    lin_hi = np.zeros((total, width), dtype=float)
    for start, part in zip(starts, parts, strict=False):
        f, t, child_left, child_right, v, af, ac, alo, ahi = part
        stop = start + len(f)
        feat[start:stop] = f
        thr[start:stop] = t
        left[start:stop] = child_left
        right[start:stop] = child_right
        val[start:stop] = v
        if af.shape[1]:
            lin_feat[start:stop, : af.shape[1]] = af
            lin_coef[start:stop, : ac.shape[1]] = ac
            lin_lo[start:stop, : alo.shape[1]] = alo
            lin_hi[start:stop, : ahi.shape[1]] = ahi
    return feat, thr, left, right, val, starts, stage_class, lin_feat, lin_coef, lin_lo, lin_hi


def _cap_rows(X, y, cap, seed):
    """Optional row subsample for boosting. Default is now None (use ALL data) — histogram binning made
    full-data fitting fast, so the old cap (needed when _best_split sorted every feature per node) is
    obsolete. Pass a cap only to trade a little accuracy for speed on very large sets."""
    if cap is None or len(y) <= cap:
        return X, y
    s = np.random.default_rng(seed * 7 + 3).choice(len(y), cap, replace=False)
    return X[s], y[s]


def _newton_leaves(tree, X, g, h, lam):
    """XGBoost-style 2nd-order leaf reweighting: leaf value = -Σg / (Σh + λ)."""
    if tree[0] == "leaf":
        return ("leaf", float(-g.sum() / (h.sum() + lam))) if len(g) else tree
    _, j, t, lt, rt = tree
    m = X[:, j] <= t
    return (
        "node",
        j,
        t,
        _newton_leaves(lt, X[m], g[m], h[m], lam),
        _newton_leaves(rt, X[~m], g[~m], h[~m], lam),
    )


def _best_split_2nd(X, g, h, min_leaf, feats, lam):
    """SECOND-ORDER (XGBoost) split: maximize G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ), Hessian-weighted.
    Beats the first-order variance split for classification, where the Hessian p(1-p) is heterogeneous."""
    n = len(g)
    G = g.sum()
    H = h.sum()
    parent = G * G / (H + lam)
    best = (1e-12, None, None)
    for j in range(X.shape[1]) if feats is None else feats:
        o = np.argsort(X[:, j], kind="stable")
        xs = X[o, j]
        gs = g[o]
        hs = h[o]
        GL = np.cumsum(gs)[:-1]
        HL = np.cumsum(hs)[:-1]
        GR = G - GL
        HR = H - HL
        i = np.arange(1, n)
        nL = i
        nR = n - i
        valid = (xs[1:] > xs[:-1]) & (nL >= min_leaf) & (nR >= min_leaf)
        gain = np.where(valid, GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent, -np.inf)
        k = int(gain.argmax())
        if gain[k] > best[0]:
            best = (float(gain[k]), j, float((xs[k] + xs[k + 1]) / 2))
    return best if best[1] is not None else None


def _fit_tree_2nd(X, g, h, depth=3, min_leaf=20, feats=None, lam=1.0):
    """CART grown by second-order gain; leaf value = -Σg/(Σh+λ). g = ∂loss/∂F, h = ∂²loss/∂F²."""
    if depth == 0 or len(g) < 2 * min_leaf:
        return ("leaf", float(-g.sum() / (h.sum() + lam)))
    s = _best_split_2nd(X, g, h, min_leaf, feats, lam)
    if s is None:
        return ("leaf", float(-g.sum() / (h.sum() + lam)))
    _, j, t = s
    m = X[:, j] <= t
    return (
        "node",
        j,
        t,
        _fit_tree_2nd(X[m], g[m], h[m], depth - 1, min_leaf, feats, lam),
        _fit_tree_2nd(X[~m], g[~m], h[~m], depth - 1, min_leaf, feats, lam),
    )


# --------------------------------------- histogram-binned second-order split (the speed fix) ------
# Pre-bin features ONCE; split-finding then accumulates per-bin gradients (O(n) bincount) instead of
# re-sorting every feature at every node (O(n log n)). This is the LightGBM/HistGBM trick — the dominant
# speedup for the boosting. Trees still store REAL thresholds, so `_tree_pred` on raw X is unchanged.


def _prebin(X, nbins=64):
    """Quantile-bin each feature to <=nbins bins. Returns (Xb bins, per-feature edges, NB=max bins). Bins are
    stored in the SMALLEST int dtype that holds them (uint8 for nbins≤256) — 4× less memory than int32, so
    every node's `Xb[rows]` slice and every histogram read moves far less data (a broad fit-time speedup)."""
    n, D = X.shape
    Xb = np.empty((n, D), dtype=(np.uint8 if nbins <= 256 else np.int16))
    qs = np.linspace(0, 1, nbins + 1)[1:-1]

    def _bin_column(j):
        e = np.unique(np.quantile(X[:, j], qs)) if n else np.array([])
        Xb[:, j] = np.searchsorted(e, X[:, j], side="left")  # side=left => ("bin<=k") EXACTLY == (x<=e[k]),
        return e

    temp_per_worker = max(1, n * 16)
    memory_workers = max(1, _PARALLEL_PREBIN_TEMP_BUDGET // temp_per_worker)
    workers = min(_PARALLEL_PREBIN_MAX_WORKERS, os.cpu_count() or 1, D, memory_workers)
    if n * D >= _PARALLEL_PREBIN_MIN_WORK and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            edges = list(pool.map(_bin_column, range(D)))
    else:
        edges = [_bin_column(j) for j in range(D)]
    # so the training bin-partition matches the real-threshold used at predict time (critical with ties).
    NB = int(Xb.max()) + 1 if n and D else 1
    return Xb, edges, NB


def _categorical_codes(X, groups):
    """Encode mutually-exclusive one-hot blocks as compact level IDs for native split search.

    The booster still receives the original numeric matrix. These codes are an
    auxiliary representation used only while proposing a finite category set;
    the stored node carries the source one-hot columns and is replayed directly
    at prediction and proof time.
    """
    X = np.asarray(X, float)
    valid = []
    for group in groups or ():
        cols = tuple(int(j) for j in group)
        if 2 <= len(cols) <= 63 and len(set(cols)) == len(cols) and all(0 <= j < X.shape[1] for j in cols):
            valid.append(cols)
    if not valid:
        return (), np.empty((len(X), 0), np.int16)
    codes = np.empty((len(X), len(valid)), np.int16)
    for ci, cols in enumerate(valid):
        block = X[:, cols]
        level = block.argmax(1).astype(np.int16)
        level[block.max(1) <= 0.5] = -1  # unseen all-zero one-hot block -> deterministic right branch
        codes[:, ci] = level
    return tuple(valid), codes


def _best_split_2nd_binned(Xb, edges, g, h, min_leaf, feats, lam, NB):
    """VECTORIZED split search: build ALL selected features' bin-histograms in ONE offset-bincount (no
    Python per-feature loop), then score every (feature, split) at once. The dominant boosting speedup
    for wide data — replaces D python bincount calls per node with a single C-level pass."""
    feats = np.asarray(list(feats), dtype=np.int64)
    F = len(feats)
    n = len(g)
    if F == 0 or n == 0:
        return None
    if _HAS_NUMBA:  # compiled path; parallelize only big nodes
        edge_lens = np.array([len(edges[j]) for j in feats], dtype=np.int64)
        gg = np.ascontiguousarray(g, np.float64)
        hh = np.ascontiguousarray(h, np.float64)
        Xbc = np.ascontiguousarray(Xb)
        kern = _hist_best_split_parallel if n * F > 300_000 else _hist_best_split_serial
        bg, bf, bk = kern(
            Xbc, gg, hh, feats, edge_lens, int(min_leaf), float(lam), NB, float(gg.sum()), float(hh.sum())
        )
        if bf < 0:
            return None
        j = int(feats[bf])
        return (float(bg), j, int(bk), float(edges[j][bk]))
    sub = Xb[:, feats]  # numpy fallback: vectorized offset-bincount
    keys = (sub + (np.arange(F) * NB)[None, :]).ravel()  # flatten into per-feature bin ranges
    m = F * NB
    Gh = np.bincount(keys, weights=np.repeat(g, F), minlength=m).reshape(F, NB)
    Hh = np.bincount(keys, weights=np.repeat(h, F), minlength=m).reshape(F, NB)
    Ch = np.bincount(keys, minlength=m).reshape(F, NB)
    G = g.sum()
    H = h.sum()
    parent = G * G / (H + lam)
    GL = np.cumsum(Gh, 1)[:, :-1]
    HL = np.cumsum(Hh, 1)[:, :-1]
    nL = np.cumsum(Ch, 1)[:, :-1]
    GR = G - GL
    HR = H - HL
    nR = n - nL
    valid = (nL >= min_leaf) & (nR >= min_leaf)
    for fi, j in enumerate(feats):  # kill split points past this feature's edges
        if len(edges[j]) < NB - 1:
            valid[fi, len(edges[j]) :] = False
    gain = np.where(valid, GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent, -np.inf)
    flat = int(gain.argmax())
    fi, k = divmod(flat, gain.shape[1])
    if gain[fi, k] <= 1e-12:
        return None
    j = int(feats[fi])
    return (float(gain[fi, k]), j, int(k), float(edges[j][k]))


class _LazyTree:
    """A nested tuple-tree that is built ONLY when first read. Prediction routes through the flat arrays
    (returned separately by the grower), so during the auto-tune CV search — where trees are fit, predicted
    once, and discarded without ever requesting a certificate — the O(nodes) tuple is never materialized.
    Materializes the whole tree on first access and caches it; children are plain tuples, so the recursive
    certificate walkers (_leaf_regions / _tree_interval / _row_path / …) see an ordinary tuple-tree."""

    __slots__ = ("_build", "_t")

    def __init__(self, build):
        self._build = build
        self._t = None

    def _mat(self):
        if self._t is None:
            if self._build is None:
                raise RuntimeError("serialized lazy tree has no materialized representation")
            self._t = self._build()
        return self._t

    def __getstate__(self):
        """Materialize before pickling; local grower closures are not portable."""
        return {"tree": self._mat()}

    def __setstate__(self, state):
        self._build = None
        self._t = state["tree"]

    def __getitem__(self, i):
        return self._mat()[i]

    def __iter__(self):
        return iter(self._mat())

    def __len__(self):
        return len(self._mat())


def _remap_tree_features(tree, source_features):
    """Map a compact numeric tree's local feature ids back to source columns."""
    source_features = np.asarray(source_features, np.int64)
    if isinstance(tree, _LazyTree):
        return _LazyTree(lambda: _remap_tree_features(tree._mat(), source_features))
    if tree[0] == "leaf":
        return tree
    if tree[0] == "cat":
        return (
            "cat",
            tree[1],
            tree[2],
            _remap_tree_features(tree[3], source_features),
            _remap_tree_features(tree[4], source_features),
        )
    return (
        "node",
        int(source_features[tree[1]]),
        tree[2],
        _remap_tree_features(tree[3], source_features),
        _remap_tree_features(tree[4], source_features),
    )


def _remap_flat_features(flat, source_features):
    """Share a compact flat tree's values/routing while restoring source ids."""
    if flat is None:
        return None
    feat = flat[0].copy()
    internal = feat >= 0
    feat[internal] = np.asarray(source_features, np.int64)[feat[internal]]
    return (feat, flat[1], flat[2], flat[3], flat[4])


def _flat_to_binned(flat, edges):
    """Compile real training thresholds back to their exact source-bin ids."""
    if flat is None:
        return None
    feat, threshold, left, right, val = flat
    bink = np.zeros(len(feat), np.int64)
    for node in np.flatnonzero(feat >= 0):
        feature = int(feat[node])
        bink[node] = int(np.searchsorted(edges[feature], threshold[node], side="left"))
    return (feat, bink, left, right, val)


def _negate_tree_values(tree):
    """Return the same certified regions with every leaf contribution negated."""
    if isinstance(tree, _LazyTree):
        return _LazyTree(lambda: _negate_tree_values(tree._mat()))
    if tree[0] == "leaf":
        return ("leaf", -float(tree[1]))
    if tree[0] == "cat":
        return (
            "cat",
            tree[1],
            tree[2],
            _negate_tree_values(tree[3]),
            _negate_tree_values(tree[4]),
        )
    return (
        "node",
        tree[1],
        tree[2],
        _negate_tree_values(tree[3]),
        _negate_tree_values(tree[4]),
    )


def _negate_flat_values(flat):
    """Share a flat tree's routing arrays while negating its leaf values."""
    if flat is None:
        return None
    return (flat[0], flat[1], flat[2], flat[3], -flat[4])


def _fit_tree_2nd_binned_best_first(
    Xb,
    edges,
    g,
    h,
    *,
    max_depth,
    max_leaves,
    min_leaf,
    feats,
    lam,
    NB,
):
    """Grow a bounded leaf-wise Newton tree for compact research candidates.

    Every proposed leaf stores its own best histogram split in a max-priority
    queue.  Expanding the globally strongest remaining leaf allocates depth to
    sparse local structure without paying for a complete deep tree.  The
    output is the ordinary scalar tuple/flat representation, so proof and
    serving semantics are unchanged.
    """
    Xb = np.asarray(Xb)
    g = np.asarray(g, dtype=float)
    h = np.asarray(h, dtype=float)
    feats = np.arange(Xb.shape[1], dtype=np.int64) if feats is None else np.asarray(feats, dtype=np.int64)
    import heapq

    rows_by_node: list[np.ndarray] = [np.arange(len(g), dtype=np.int64)]
    depths = [0]
    feature = [-1]
    split_bin = [-1]
    threshold = [0.0]
    left = [-1]
    right = [-1]
    value = [float(-g.sum() / (h.sum() + lam)) if len(g) else 0.0]
    queue: list[tuple[float, int, int, int, float]] = []

    def propose(node):
        rows = rows_by_node[node]
        if depths[node] >= max_depth or len(rows) < 2 * min_leaf:
            return
        split = _best_split_2nd_binned(
            Xb[rows],
            edges,
            g[rows],
            h[rows],
            min_leaf,
            feats,
            lam,
            NB,
        )
        if split is not None:
            gain, source_feature, bin_index, real_threshold = split
            heapq.heappush(
                queue,
                (
                    -float(gain),
                    int(node),
                    int(source_feature),
                    int(bin_index),
                    float(real_threshold),
                ),
            )

    propose(0)
    leaves = 1
    while queue and leaves < max_leaves:
        _negative_gain, node, source_feature, bin_index, real_threshold = heapq.heappop(queue)
        rows = rows_by_node[node]
        goes_left = Xb[rows, source_feature] <= bin_index
        left_rows = rows[goes_left]
        right_rows = rows[~goes_left]
        if len(left_rows) < min_leaf or len(right_rows) < min_leaf:
            continue
        first_child = len(feature)
        second_child = first_child + 1
        feature[node] = source_feature
        split_bin[node] = bin_index
        threshold[node] = real_threshold
        left[node] = first_child
        right[node] = second_child
        for child_rows in (left_rows, right_rows):
            rows_by_node.append(child_rows)
            depths.append(depths[node] + 1)
            feature.append(-1)
            split_bin.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(float(-g[child_rows].sum() / (h[child_rows].sum() + lam)))
        leaves += 1
        propose(first_child)
        propose(second_child)

    feat_array = np.asarray(feature, dtype=np.int64)
    threshold_array = np.asarray(threshold, dtype=float)
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    value_array = np.asarray(value, dtype=float)
    binned_array = np.asarray(split_bin, dtype=np.int64)
    prediction: np.ndarray = np.empty(len(g), dtype=float)
    for node, rows in enumerate(rows_by_node):
        if feat_array[node] < 0:
            prediction[rows] = value_array[node]

    def to_tuple(node):
        if feat_array[node] < 0:
            return ("leaf", float(value_array[node]))
        return (
            "node",
            int(feat_array[node]),
            float(threshold_array[node]),
            to_tuple(int(left_array[node])),
            to_tuple(int(right_array[node])),
        )

    flat = (
        feat_array,
        threshold_array,
        left_array,
        right_array,
        value_array,
    )
    binned = (
        feat_array,
        binned_array,
        left_array,
        right_array,
        value_array,
    )
    return _LazyTree(lambda: to_tuple(0)), prediction, flat, binned


def _fit_tree_2nd_binned(
    Xb,
    edges,
    g,
    h,
    depth=3,
    min_leaf=20,
    feats=None,
    lam=1.0,
    NB=None,
    mono=None,
    cat_codes=None,
    cat_groups=(),
    cat_feats=None,
    cat_min_leaf=None,
    honest_categorical=False,
    honest_seed=0,
    parallel_hist=False,
    return_binned=False,
    max_leaves=None,
):
    """Grow a 2nd-order histogram CART and return (tuple-tree, per-row leaf preds, FLAT tree). Uses the
    iterative numba grower when available (no Python recursion, no per-node array copies — bit-identical trees
    via a stable partition), else the pure-Python recursion. Trees store REAL thresholds (edges[j][bin]).
    The flat form (feat/thr/left/right/val arrays) is built ONCE directly from the numba node table so
    prediction never re-flattens — the tuple is for the certificate engine, the flat for fit/predict."""
    if max_leaves is not None:
        if mono is not None or (cat_codes is not None and cat_groups):
            raise ValueError("best-first trees currently require a numeric non-monotone schema")
        if int(max_leaves) < 2:
            raise ValueError("max_leaves must be at least two")
        if NB is None:
            NB = int(Xb.max()) + 1 if len(g) else 1
        result = _fit_tree_2nd_binned_best_first(
            Xb,
            edges,
            g,
            h,
            max_depth=int(depth),
            max_leaves=int(max_leaves),
            min_leaf=int(min_leaf),
            feats=feats,
            lam=float(lam),
            NB=int(NB),
        )
        return result if return_binned else result[:3]
    if not _HAS_NUMBA:
        t, p = _fit_tree_2nd_binned_py(Xb, edges, g, h, depth, min_leaf, feats, lam, NB)
        flat = _flatten_tree(t)
        if return_binned:
            return t, p, flat, _flat_to_binned(flat, edges)
        return t, p, flat
    n = len(g)
    if NB is None:
        NB = int(Xb.max()) + 1 if n else 1
    ff = np.arange(Xb.shape[1], dtype=np.int64) if feats is None else np.asarray(feats, np.int64)
    groups = tuple(tuple(int(j) for j in group) for group in cat_groups)
    if cat_codes is None or not groups:
        cc = np.empty((n, 0), np.int16)
        clevels = np.empty(0, np.int64)
        cf = np.empty(0, np.int64)
    else:
        cc = np.asarray(cat_codes, np.int16)
        if cc.shape != (n, len(groups)):
            raise ValueError("categorical codes must align with the fitting rows and categorical groups")
        clevels = np.asarray([len(group) for group in groups], np.int64)
        valid = np.flatnonzero((clevels >= 2) & (clevels <= 63)).astype(np.int64)
        cf = valid if cat_feats is None else np.intersect1d(np.asarray(cat_feats, np.int64), valid)
    if n == 0 or (len(ff) == 0 and len(cf) == 0):
        v = float(np.clip(-g.sum() / (h.sum() + lam), -1e18, 1e18)) if n else 0.0
        flat = _flatten_tree(("leaf", v))
        if return_binned:
            return ("leaf", v), np.full(n, v), flat, _flat_to_binned(flat, edges)
        return ("leaf", v), np.full(n, v), flat
    edge_lens = np.array([len(edges[j]) for j in ff], np.int64)
    D = Xb.shape[1]
    mono_arr = np.zeros(D, np.int64) if mono is None else np.asarray(mono, np.int64)
    max_cat = int(clevels.max()) if len(clevels) else 1
    cat_min = int(min_leaf) if cat_min_leaf is None else max(1, int(cat_min_leaf))
    feat, bink, catg, catmask, left, right, val, preds, nnodes = _grow_2nd_binned_nb(
        np.ascontiguousarray(Xb),
        np.ascontiguousarray(g, np.float64),
        np.ascontiguousarray(h, np.float64),
        ff,
        edge_lens,
        np.ascontiguousarray(cc),
        np.ascontiguousarray(cf),
        np.ascontiguousarray(clevels),
        int(max_cat),
        int(depth),
        int(min_leaf),
        int(cat_min),
        float(lam),
        int(NB),
        mono_arr,
        mono is not None,
        bool(honest_categorical),
        int(honest_seed),
        bool(parallel_hist),
    )
    # FLAT tree (truncated to live nodes): real thresholds for internal nodes, 0 at leaves (feat=-1).
    feat_t = feat[:nnodes].astype(np.int64)
    thr_t = np.zeros(nnodes)
    for i in range(nnodes):
        if feat_t[i] >= 0:
            thr_t[i] = float(edges[feat_t[i]][bink[i]])
    flat = None
    if not np.any(feat_t == -2):
        flat = (
            feat_t,
            thr_t,
            left[:nnodes].astype(np.int64),
            right[:nnodes].astype(np.int64),
            val[:nnodes].astype(float),
        )
    binned_flat = None
    if flat is not None:
        binned_flat = (
            feat_t,
            bink[:nnodes].astype(np.int64),
            left[:nnodes].astype(np.int64),
            right[:nnodes].astype(np.int64),
            val[:nnodes].astype(float),
        )

    def _to_tuple(
        nd,
    ):  # flat node table -> nested tuple-tree with REAL thresholds (O(nodes)) for certificates
        if feat[nd] == -1:
            return ("leaf", float(val[nd]))
        if feat[nd] == -2:
            ci = int(catg[nd])
            mask = int(catmask[nd])
            levels = tuple(level for level in range(int(clevels[ci])) if mask & (1 << level))
            return (
                "cat",
                groups[ci],
                levels,
                _to_tuple(int(left[nd])),
                _to_tuple(int(right[nd])),
            )
        j = int(feat[nd])
        return ("node", j, float(edges[j][bink[nd]]), _to_tuple(int(left[nd])), _to_tuple(int(right[nd])))

    result = (_LazyTree(lambda: _to_tuple(0)), preds, flat)
    return (*result, binned_flat) if return_binned else result


def _fit_tree_softmax_shared_binned(
    Xb,
    edges,
    gradient,
    hessian,
    depth=3,
    min_leaf=20,
    feats=None,
    lam=1.0,
    NB=None,
):
    """Fit one numeric partition carrying a centered Newton vector in each leaf."""
    Xb = np.asarray(Xb)
    gradient = np.asarray(gradient, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    if gradient.ndim != 2 or hessian.shape != gradient.shape:
        raise ValueError("shared softmax gradient and hessian must be aligned n-by-K matrices")
    n, n_classes = gradient.shape
    if n_classes < 2:
        raise ValueError("shared softmax trees require at least two classes")
    if NB is None:
        NB = int(Xb.max()) + 1 if n else 1
    ff = np.arange(Xb.shape[1], dtype=np.int64) if feats is None else np.asarray(feats, np.int64)

    def centered_value(rows):
        gsum = gradient[rows].sum(0)
        hsum = hessian[rows].sum(0)
        value = -gsum / (hsum + lam)
        return value - value.mean()

    if not _HAS_NUMBA:
        predictions = np.empty((n, n_classes), dtype=float)

        def grow(rows, remaining_depth):
            value = centered_value(rows)
            if remaining_depth == 0 or len(rows) < 2 * min_leaf or len(ff) == 0:
                predictions[rows] = value
                return ("leaf", value)
            total_g = gradient[rows].sum(0)
            total_h = hessian[rows].sum(0)
            parent = np.sum(total_g * total_g / (total_h + lam))
            best = (1e-12, -1, -1)
            for feature in ff:
                bins = Xb[rows, feature]
                count = np.bincount(bins, minlength=NB).astype(float)
                gh = np.column_stack(
                    [np.bincount(bins, weights=gradient[rows, cls], minlength=NB) for cls in range(n_classes)]
                )
                hh = np.column_stack(
                    [np.bincount(bins, weights=hessian[rows, cls], minlength=NB) for cls in range(n_classes)]
                )
                left_count = np.cumsum(count)[:-1]
                left_g = np.cumsum(gh, axis=0)[:-1]
                left_h = np.cumsum(hh, axis=0)[:-1]
                right_g = total_g - left_g
                right_h = total_h - left_h
                gain = (
                    np.sum(
                        left_g * left_g / (left_h + lam) + right_g * right_g / (right_h + lam),
                        axis=1,
                    )
                    - parent
                )
                valid = (
                    (left_count >= min_leaf)
                    & (len(rows) - left_count >= min_leaf)
                    & (np.arange(len(gain)) < len(edges[int(feature)]))
                )
                gain = np.where(valid, gain, -np.inf)
                split_bin = int(np.argmax(gain))
                if gain[split_bin] > best[0]:
                    best = (float(gain[split_bin]), int(feature), split_bin)
            if best[1] < 0:
                predictions[rows] = value
                return ("leaf", value)
            _gain, feature, split_bin = best
            goes_left = Xb[rows, feature] <= split_bin
            left_tree = grow(rows[goes_left], remaining_depth - 1)
            right_tree = grow(rows[~goes_left], remaining_depth - 1)
            return ("node", feature, float(edges[feature][split_bin]), left_tree, right_tree)

        vector_tree = grow(np.arange(n, dtype=np.int64), int(depth))

        def scalar_tree(node, cls):
            if node[0] == "leaf":
                return ("leaf", float(node[1][cls]))
            return (
                "node",
                node[1],
                node[2],
                scalar_tree(node[3], cls),
                scalar_tree(node[4], cls),
            )

        trees = [scalar_tree(vector_tree, cls) for cls in range(n_classes)]
        flats = [_flatten_tree(tree) for tree in trees]
        return trees, predictions, flats, [_flat_to_binned(flat, edges) for flat in flats]

    if n == 0 or len(ff) == 0:
        rows = np.arange(n, dtype=np.int64)
        value = centered_value(rows)
        trees = [("leaf", float(value[cls])) for cls in range(n_classes)]
        predictions = np.tile(value, (n, 1))
        flats = [_flatten_tree(tree) for tree in trees]
        return trees, predictions, flats, [_flat_to_binned(flat, edges) for flat in flats]
    edge_lens = np.asarray([len(edges[int(feature)]) for feature in ff], dtype=np.int64)
    feat, bink, left, right, values, predictions, nodes = _grow_softmax_shared_binned_nb(
        np.ascontiguousarray(Xb),
        np.ascontiguousarray(gradient, np.float64),
        np.ascontiguousarray(hessian, np.float64),
        np.ascontiguousarray(ff, np.int64),
        edge_lens,
        int(depth),
        int(min_leaf),
        float(lam),
        int(NB),
    )
    feat_out = feat[:nodes].astype(np.int64)
    threshold = np.zeros(nodes, dtype=float)
    for node in np.flatnonzero(feat_out >= 0):
        threshold[node] = float(edges[int(feat_out[node])][int(bink[node])])
    left_out = left[:nodes].astype(np.int64)
    right_out = right[:nodes].astype(np.int64)

    def to_tuple(node, cls):
        if feat[node] < 0:
            return ("leaf", float(values[node, cls]))
        feature = int(feat[node])
        return (
            "node",
            feature,
            float(edges[feature][bink[node]]),
            to_tuple(int(left[node]), cls),
            to_tuple(int(right[node]), cls),
        )

    trees, flats, binned_flats = [], [], []
    for cls in range(n_classes):
        trees.append(_LazyTree(lambda class_index=cls: to_tuple(0, class_index)))
        flats.append((feat_out, threshold, left_out, right_out, values[:nodes, cls].astype(float)))
        binned_flats.append(
            (feat_out, bink[:nodes].astype(np.int64), left_out, right_out, values[:nodes, cls].astype(float))
        )
    return trees, predictions, flats, binned_flats


_WARMED = False


def warm_numba():
    """Compile the numba tree kernels ONCE, single-threaded, before any concurrent (threaded) use — so a
    thread pool of config fits never races on first-call JIT compilation. Idempotent and a no-op without
    numba. Cheap: one tiny grow on an 8×2 dummy problem."""
    global _WARMED
    if _WARMED:
        return
    _WARMED = True
    if not _HAS_NUMBA:
        return
    Xb = np.zeros((8, 2), np.int64)
    g = np.linspace(-1.0, 1.0, 8)
    h = np.ones(8)
    edges = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    try:
        _fit_tree_2nd_binned(Xb, edges, g, h, depth=2, min_leaf=1, NB=2)
        shared_gradient = np.column_stack([g, -g, 0.5 * g])
        shared_hessian = np.ones_like(shared_gradient)
        _fit_tree_softmax_shared_binned(Xb, edges, shared_gradient, shared_hessian, depth=2, min_leaf=1, NB=2)
        X = np.zeros((2, 2), float)
        feat = np.array([0, -1, -1], np.int64)
        thr = np.array([0.0, 0.0, 0.0])
        left = np.array([1, -1, -1], np.int64)
        right = np.array([2, -1, -1], np.int64)
        val = np.array([0.0, -0.5, 0.5])
        _route_flat_rows_nb(feat, thr, left, right, val, X, np.array([0, 1], np.int64))
        lin_feat = np.array([[-1], [0], [0]], np.int64)
        lin_coef = np.array([[0.0], [0.1], [-0.1]])
        lin_lo = np.full((3, 1), -1.0)
        lin_hi = np.full((3, 1), 1.0)
        _route_flat_affine_nb(feat, thr, left, right, val, lin_feat, lin_coef, lin_lo, lin_hi, X)
        starts = np.array([0], np.int64)
        stage_class = np.array([0], np.int64)
        base = np.zeros(1)
        _forest_scores_flat_nb(feat, thr, left, right, val, starts, stage_class, base, 0.1, X)
        _forest_scores_binary_mirrored_nb(feat, thr, left, right, val, starts, np.zeros(2), 0.1, X)
        _forest_scores_multiclass_shared_nb(
            feat, thr, left, right, np.column_stack([val, -val, 0.5 * val]), starts, np.zeros(3), 0.1, X
        )
        suffix_min = np.zeros((2, 3))
        suffix_max = np.zeros((2, 3))
        suffix_abs = np.zeros((2, 3))
        suffix_count = np.zeros((2, 3), np.int64)
        _forest_predict_adaptive_flat_nb(
            feat,
            thr,
            left,
            right,
            val,
            starts,
            stage_class,
            np.zeros(3),
            0.1,
            suffix_min,
            suffix_max,
            suffix_abs,
            suffix_count,
            1,
            X,
        )
        _forest_predict_adaptive_binary_mirrored_nb(
            feat,
            thr,
            left,
            right,
            val,
            starts,
            np.zeros(2),
            0.1,
            suffix_min[:, :2],
            suffix_max[:, :2],
            suffix_abs[:, :2],
            suffix_count[:, :2],
            1,
            X,
        )
        _forest_predict_adaptive_multiclass_shared_nb(
            feat,
            thr,
            left,
            right,
            np.column_stack([val, -val, 0.5 * val]),
            starts,
            np.zeros(3),
            0.1,
            suffix_min,
            suffix_max,
            suffix_abs,
            suffix_count,
            1,
            X,
        )
        _forest_scores_affine_nb(
            feat, thr, left, right, val, starts, stage_class, lin_feat, lin_coef, lin_lo, lin_hi, base, 0.1, X
        )
    except Exception:  # warmup must never break fit
        pass


def _fit_tree_2nd_binned_py(
    Xb, edges, g, h, depth=3, min_leaf=20, feats=None, lam=1.0, NB=None, mono=None, lo=-1e18, hi=1e18
):
    """Pure-Python recursive grower (numba-free fallback / parity reference). Same result as the numba grower.
    Returns (tuple-tree, per-row leaf preds)."""
    if NB is None:
        NB = int(Xb.max()) + 1 if len(g) else 1
    val = float(np.clip(-g.sum() / (h.sum() + lam), lo, hi))
    if depth == 0 or len(g) < 2 * min_leaf:
        return ("leaf", val), np.full(len(g), val)
    ff = np.arange(Xb.shape[1]) if feats is None else feats
    s = _best_split_2nd_binned(Xb, edges, g, h, min_leaf, ff, lam, NB)
    if s is None:
        return ("leaf", val), np.full(len(g), val)
    _, j, k, thr = s
    m = Xb[:, j] <= k
    loL, hiL, loR, hiR = lo, hi, lo, hi
    if mono is not None and mono[j] != 0:  # split on a monotone feature -> order children
        wl = float(np.clip(-g[m].sum() / (h[m].sum() + lam), lo, hi))
        wr = float(np.clip(-g[~m].sum() / (h[~m].sum() + lam), lo, hi))
        mid = 0.5 * (wl + wr)
        if mono[j] > 0:  # increasing: left ≤ mid ≤ right
            hiL, loR = mid, mid
        else:  # decreasing: left ≥ mid ≥ right
            loL, hiR = mid, mid
    lt, lp = _fit_tree_2nd_binned_py(
        Xb[m], edges, g[m], h[m], depth - 1, min_leaf, feats, lam, NB, mono, loL, hiL
    )
    rt, rp = _fit_tree_2nd_binned_py(
        Xb[~m], edges, g[~m], h[~m], depth - 1, min_leaf, feats, lam, NB, mono, loR, hiR
    )
    preds = np.empty(len(g))
    preds[m] = lp
    preds[~m] = rp
    return ("node", j, thr, lt, rt), preds


# --------------------------------------------------------------------------- regression boosting
def reason_boost(
    X,
    y,
    holdout=0.3,
    lr=0.05,
    n_terms=500,
    depth=3,
    min_leaf=20,
    sub=0.7,
    mf=0.7,
    patience=25,
    fit_cap=None,
    seed=0,
):
    """Verified stochastic residual tree-boosting for regression; held-out RMSE early-stop = verifier."""
    X, y = _cap_rows(X, y, fit_cap, seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    c = int((1 - holdout) * len(y))
    fit, ver = idx[:c], idx[c:]
    Xf, yf, Xv, yv = X[fit], y[fit], X[ver], y[ver]
    D = X.shape[1]
    nfeat = max(1, int(mf * D))
    rm = np.random.default_rng(seed * 17 + 1)
    base = float(yf.mean())
    pf = np.full(len(fit), base)
    pv = np.full(len(ver), base)
    best = math.sqrt(((pv - yv) ** 2).mean())
    trees = []
    keep = 0
    bad = 0
    for _ in range(n_terms):
        s = rm.random(len(fit)) < sub
        feats = rm.choice(D, nfeat, replace=False)
        t = _fit_tree(Xf[s], (yf - pf)[s], depth, min_leaf, feats)
        pf = pf + lr * _tree_pred(t, Xf)
        pv = pv + lr * _tree_pred(t, Xv)
        trees.append(t)
        r = math.sqrt(((pv - yv) ** 2).mean())
        if r < best * (1 - 1e-4):  # relative improvement (scale-free; abs 1e-6 broke small-magnitude targets)
            best = r
            keep = len(trees)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return (base, lr, trees[:keep])


def _huber_grad(pred, y, huber):
    """Gradient of the loss w.r.t. the current prediction. Squared loss (huber=None): g = pred−y. Huber:
    g = clip(pred−y, −δ, δ) with δ = the `huber`-quantile of |residual| this round — outliers contribute a
    bounded gradient, so the tail can't drag the fit (a principled replacement for winsorizing the target)."""
    r = pred - y
    if huber is None:
        return r
    delta = float(np.quantile(np.abs(r), huber)) + 1e-9
    return np.clip(r, -delta, delta)


def _boost_2nd_run(
    X,
    y,
    edges,
    NB,
    n_terms,
    lr,
    depth,
    min_leaf,
    sub,
    mf,
    lam,
    seed,
    huber=None,
    mono=None,
    allowed=None,
    w=None,
    linear_leaf=False,
    lam_lin=10.0,
):
    """Run `n_terms` rounds of 2nd-order binned residual boosting on ALL of (X,y). Returns (base, trees)."""
    Xb = np.clip(
        np.stack([np.searchsorted(edges[j], X[:, j], side="left") for j in range(X.shape[1])], 1), 0, NB - 1
    ).astype(np.int16)
    pool = (
        np.arange(X.shape[1]) if allowed is None else np.asarray(allowed, int)
    )  # split only within `allowed`
    nfeat = max(1, int(mf * len(pool)))
    rm = np.random.default_rng(seed * 17 + 1)
    wv = np.ones(len(y)) if w is None else np.asarray(w, float)  # per-row weight scales grad + hessian
    base = float(y.mean())
    pred = np.full(len(y), base)
    trees, flats = [], []
    for _ in range(n_terms):
        g = wv * _huber_grad(pred, y, huber)
        h = wv
        s = rm.random(len(y)) < sub
        feats = rm.choice(pool, min(nfeat, len(pool)), replace=False)
        t, p, flat = _fit_tree_2nd_binned(Xb[s], edges, g[s], h[s], depth, min_leaf, feats, lam, NB, mono)
        if linear_leaf:  # path-constrained ridge leaves (residual = -g/h, weight = h); affine -> predict all
            t = _fit_leaf_lin(t, X[s], -g[s] / h[s], h[s], lam_lin, lam)
            pred += lr * _affine_tree_pred(t, X)
        else:
            pred[s] += lr * p  # subsample rows: leaf values captured at build (no re-predict)
            if not s.all():
                pred[~s] += lr * _tree_pred(flat, X[~s])  # only held-out-of-subsample rows need a predict
        trees.append(t)
        flats.append(flat)
    return base, trees, flats


def _ridge_leaf(Xr, target, w, feats, lam_lin, lam_const):
    """Weighted ridge for ONE leaf, over `feats` only (the path-constrained features). Returns full-D
    (coef, intercept, clamp_lo, clamp_hi): pred = intercept + coef·clip(x, lo, hi), with coef nonzero only on
    `feats`. Features standardized in-leaf for conditioning then folded back to raw form; the intercept keeps
    the constant-leaf regularized weighted mean as its anchor, so `feats=[]` reproduces the constant leaf.
    Own primitives only (numpy normal equations). CERT-SAFE: coef is nonzero only on path features (bounded
    within the leaf box) and each is clamped to the leaf's training range."""
    D = Xr.shape[1]
    coef = np.zeros(D)
    NOLO, NOHI = np.full(D, -np.inf), np.full(D, np.inf)
    wsum = float(w.sum())
    if wsum <= 0:
        return coef, 0.0, NOLO, NOHI
    b_const = float((w * target).sum() / (wsum + lam_const))  # regularized weighted mean (constant anchor)
    F = np.asarray(sorted(feats), int)
    if len(F) == 0:
        return coef, b_const, NOLO, NOHI
    A = Xr[:, F]
    mu = (w[:, None] * A).sum(0) / wsum
    sd = np.sqrt((w[:, None] * (A - mu) ** 2).sum(0) / wsum)
    good = sd > 1e-9  # drop leaf-constant features (no in-leaf variation -> no linear signal)
    if not good.any():
        return coef, b_const, NOLO, NOHI
    F, mu, sd = F[good], mu[good], sd[good]
    flo, fhi = Xr[:, F].min(0), Xr[:, F].max(0)  # per-feature clamp range (cert-safe bounded box)
    As = (Xr[:, F] - mu) / sd
    M = np.column_stack([np.ones(len(As)), As])
    reg = np.eye(M.shape[1]) * lam_lin
    reg[0, 0] = lam_const  # intercept regularized like the constant leaf, not like the linear coefs
    try:
        beta = np.linalg.solve((M * w[:, None]).T @ M + reg, (M * w[:, None]).T @ target)
    except np.linalg.LinAlgError:
        return coef, b_const, NOLO, NOHI
    wstd = beta[1:]
    coef_raw = wstd / sd  # fold standardization back to raw-feature coefficients
    coef[F] = coef_raw
    lo, hi = NOLO.copy(), NOHI.copy()
    lo[F], hi[F] = flo, fhi
    return coef, float(beta[0] - (wstd * mu / sd).sum()), lo, hi


def _fit_leaf_lin(tree, X, target, w, lam_lin, lam_const):
    """Post-pass on a grown (constant-leaf) tuple-tree: fit a `_ridge_leaf` in each leaf on that leaf's rows,
    using ONLY the features constrained on its root->leaf path (cert-safe). Returns a new tuple-tree whose
    leaves are ('leaf', b, coef, lo, hi) where linear, or the plain ('leaf', b) where there is no linear
    signal (so the constant case is byte-identical in form). Mirrors the validated prototype."""
    all_rows = np.arange(len(X))

    def rec(node, rows, path):
        if node[0] == "leaf":
            if len(rows) == 0:
                return ("leaf", float(node[1]))
            coef, b, lo, hi = _ridge_leaf(X[rows], target[rows], w[rows], path, lam_lin, lam_const)
            if not coef.any():
                return ("leaf", b)
            return ("leaf", b, coef, lo, hi)
        if node[0] == "cat":
            _, cols, levels, L, R = node
            m = _cat_member(X[rows], cols, levels)
            # A category partition is a valid leaf-region constraint but not an ordered linear feature.
            return ("cat", cols, levels, rec(L, rows[m], path), rec(R, rows[~m], path))
        _, j, thr, L, R = node
        m = X[rows, j] <= thr
        return ("node", j, thr, rec(L, rows[m], path | {j}), rec(R, rows[~m], path | {j}))

    return rec(tree, all_rows, set())


def _affine_tree_pred(tree, X):
    """Per-row prediction for a (possibly linear-leaf) tuple-tree: constant leaf -> b; linear leaf ->
    b + clip(x, lo, hi)·coef (clip keeps the linear term inside the leaf's training box)."""
    out = np.zeros(len(X))

    def rec(node, rows):
        if node[0] == "leaf":
            if len(node) == 2:
                out[rows] = node[1]
            else:
                _, b, coef, lo, hi = node
                nz = np.nonzero(coef)[
                    0
                ]  # coef is nonzero only on the ≤depth path features -> clip+matmul over
                out[rows] = b + (  # those, not all D (numerically identical; big win when D >> depth)
                    np.clip(X[np.ix_(rows, nz)], lo[nz], hi[nz]) @ coef[nz] if len(nz) else 0.0
                )
            return
        if node[0] == "cat":
            _, cols, levels, L, R = node
            m = _cat_member(X[rows], cols, levels)
        else:
            _, j, thr, L, R = node
            m = X[rows, j] <= thr
        rec(L, rows[m])
        rec(R, rows[~m])

    rec(tree, np.arange(len(X)))
    return out


def reason_boost_2nd(
    X,
    y,
    holdout=0.15,
    lr=0.05,
    n_terms=800,
    depth=4,
    min_leaf=20,
    sub=0.7,
    mf=0.7,
    patience=40,
    lam=1.0,
    refit=True,
    huber=None,
    mono=None,
    allowed=None,
    w=None,
    fit_cap=None,
    seed=0,
    nbins=64,
    linear_leaf=False,
    lam_lin=10.0,
    validation_groups=None,
):
    """2nd-order (Newton) residual boosting for regression — the stronger self-contained learner. Squared
    loss => gradient g = pred − y, Hessian h = 1, so each leaf = −Σg/(Σh+λ): an L2-REGULARIZED leaf mean.
    huber=q (a quantile in (0,1)) switches to a HUBER-style objective: the gradient is clipped at the
    q-quantile of |residual| each round, so the long tail contributes a bounded gradient (a principled,
    task-agnostic robustness — replaces winsorizing the target). Histogram-binned splits (fast). A held-out
    slice EARLY-STOPS to pick tree count; then (refit) we RETRAIN on ALL the data. No external model. For large
    n the round-count SEARCH runs on a capped subsample (the early-stopped count transfers across subsample
    size, validated) — so the expensive full-data pass happens ONCE in the refit, not twice; accuracy is set by
    the full-data refit, unchanged. n ≤ cap (or refit off) is identical to before."""
    X = np.asarray(X, float)
    source_y = np.asarray(y, float)
    source_weight = None if w is None else np.asarray(w, float)
    if source_weight is not None and (source_weight.ndim != 1 or len(source_weight) != len(source_y)):
        raise ValueError("w must have one value per regression row")
    groups = None if validation_groups is None else np.asarray(validation_groups, dtype=np.int64)
    if groups is not None and (groups.ndim != 1 or len(groups) != len(source_y)):
        raise ValueError("validation_groups must have one value per regression row")
    selected = None
    if fit_cap is not None and len(source_y) > int(fit_cap):
        if groups is None:
            selected = np.random.default_rng(seed * 7 + 3).choice(len(source_y), int(fit_cap), replace=False)
        else:
            from tabpvn.validation import FutureValidation

            selected = FutureValidation(groups).bounded_rows(int(fit_cap))
        X, source_y = X[selected], source_y[selected]
        groups = None if groups is None else groups[selected]
        source_weight = None if source_weight is None else source_weight[selected]
    y = source_y
    w = source_weight
    n = len(y)
    SEARCH_CAP = 40000  # round count is stable above this; the full-data refit below carries the accuracy
    if refit and n > SEARCH_CAP:  # search the tree count on a representative subsample (cheap at scale)
        if groups is None:
            ss = np.random.default_rng(seed * 3 + 5).choice(n, SEARCH_CAP, replace=False)
        else:
            from tabpvn.validation import FutureValidation

            ss = FutureValidation(groups).bounded_rows(SEARCH_CAP)
        Xs, ys = X[ss], y[ss]
        ws = None if w is None else np.asarray(w, float)[ss]
        search_groups = None if groups is None else groups[ss]
    else:
        Xs, ys, ws = X, y, w
        search_groups = groups
    if search_groups is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ys))
        c = int((1 - holdout) * len(ys))
        fit, ver = idx[:c], idx[c:]
    else:
        from tabpvn.validation import FutureValidation

        fit, ver = FutureValidation(search_groups).split(
            holdout=holdout,
            min_train=max(2, int(0.25 * len(ys))),
            min_valid=max(1, int(0.05 * len(ys))),
        )
    Xf, yf, Xv, yv = Xs[fit], ys[fit], Xs[ver], ys[ver]
    wf = np.ones(len(fit)) if ws is None else np.asarray(ws, float)[fit]  # per-row weight (fit slice)
    pool = (
        np.arange(X.shape[1]) if allowed is None else np.asarray(allowed, int)
    )  # split only within `allowed`
    nfeat = max(1, int(mf * len(pool)))
    rm = np.random.default_rng(seed * 17 + 1)
    Xbf, edges, NB = _prebin(Xf, nbins)
    base = float(yf.mean())
    pf = np.full(len(fit), base)
    pv = np.full(len(ver), base)
    best = math.sqrt(((pv - yv) ** 2).mean())
    trees, flats = [], []
    keep = 0
    bad = 0
    for _ in range(n_terms):
        g = wf * _huber_grad(pf, yf, huber)
        h = wf  # weighted squared/Huber-clipped gradient
        s = rm.random(len(fit)) < sub
        feats = rm.choice(pool, min(nfeat, len(pool)), replace=False)
        t, p, flat = _fit_tree_2nd_binned(Xbf[s], edges, g[s], h[s], depth, min_leaf, feats, lam, NB, mono)
        if linear_leaf:  # fit a path-constrained ridge in each leaf (residual = -g/h, weight = h)
            t = _fit_leaf_lin(t, Xf[s], -g[s] / h[s], h[s], lam_lin, lam)
            pf += lr * _affine_tree_pred(t, Xf)  # preserve the legacy fit-time accumulation exactly
            pv += lr * _affine_tree_pred(t, Xv)
        else:
            pf[s] += (
                lr * p
            )  # subsample rows free (leaf values from build); only the rest + holdout re-predict
            if not s.all():
                pf[~s] += lr * _tree_pred(flat, Xf[~s])
            pv += lr * _tree_pred(flat, Xv)
        trees.append(t)
        flats.append(flat)
        r = math.sqrt(((pv - yv) ** 2).mean())
        if r < best * (1 - 1e-4):  # relative improvement (scale-free; abs 1e-6 broke small-magnitude targets)
            best = r
            keep = len(trees)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if not refit or keep == 0:
        return (base, lr, trees[:keep], flats[:keep], linear_leaf)
    _, edges_all, NB_all = _prebin(X, nbins)  # refit on 100% of the data for `keep` rounds
    base2, trees2, flats2 = _boost_2nd_run(
        X,
        y,
        edges_all,
        NB_all,
        keep,
        lr,
        depth,
        min_leaf,
        sub,
        mf,
        lam,
        seed,
        huber,
        mono,
        allowed,
        w,
        linear_leaf,
        lam_lin,
    )
    return (base2, lr, trees2, flats2, linear_leaf)


def boost_predict(model, X):
    base, lr, trees = model[0], model[1], model[2]
    if len(model) > 4 and model[4]:  # linear leaves retain their exact legacy tuple evaluator
        return base + lr * sum((_affine_tree_pred(t, X) for t in trees), np.zeros(len(X)))
    preds = model[3] if len(model) > 3 else trees  # flats when available (no re-flatten), else tuple-trees
    return base + lr * sum((_tree_pred(t, X) for t in preds), np.zeros(len(X)))


# --------------------------------------------------------------------------- classification boosting
def reason_boost_clf(
    X,
    y,
    holdout=0.3,
    lr=0.1,
    n_terms=700,
    depth=3,
    min_leaf=20,
    sub=0.7,
    mf=0.7,
    patience=30,
    newton=True,
    lam=1.0,
    fit_cap=None,
    second_order=True,
    seed=0,
    nbins=64,
):
    """Verified LOGISTIC rule-boosting (binary): iterate on the logistic residual, each term a CART tree
    on a row+feature subsample, Newton-reweighted leaves (Hessian h=p(1-p)), kept while it improves
    HELD-OUT log-loss. Histogram-binned splits keep it fast. Full coverage + calibrated probability."""
    X, y = _cap_rows(X, np.asarray(y), fit_cap, seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    c = int((1 - holdout) * len(y))
    fit, ver = idx[:c], idx[c:]
    Xf, yf, Xv, yv = X[fit], y[fit].astype(float), X[ver], y[ver].astype(float)
    D = X.shape[1]
    nfeat = max(1, int(mf * D))
    rm = np.random.default_rng(seed * 17 + 1)
    Xbf, edges, NB = _prebin(Xf, nbins) if second_order else (None, None, None)
    p0 = float(np.clip(yf.mean(), 1e-6, 1 - 1e-6))
    base = math.log(p0 / (1 - p0))
    Ff = np.full(len(fit), base)
    Fv = np.full(len(ver), base)

    def _ll(F, t):
        p = np.clip(1.0 / (1.0 + np.exp(-F)), 1e-9, 1 - 1e-9)
        return -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()

    best = _ll(Fv, yv)
    trees = []
    keep = 0
    bad = 0
    for _ in range(n_terms):
        s = rm.random(len(fit)) < sub
        feats = rm.choice(D, nfeat, replace=False)
        pf = 1.0 / (1.0 + np.exp(-Ff))
        if second_order:  # XGBoost-style Hessian-weighted splits
            t, _, flat = _fit_tree_2nd_binned(
                Xbf[s], edges, (pf - yf)[s], (pf * (1 - pf))[s], depth, min_leaf, feats, lam, NB
            )
        else:
            t = _fit_tree(Xf[s], (yf - pf)[s], depth, min_leaf, feats)
            if newton:
                t = _newton_leaves(t, Xf[s], (pf - yf)[s], (pf * (1 - pf))[s], lam)
            flat = _flatten_tree(t)
        Ff = Ff + lr * _tree_pred(flat, Xf)
        Fv = Fv + lr * _tree_pred(flat, Xv)
        trees.append(t)
        r = _ll(Fv, yv)
        if r < best - 1e-6:
            best = r
            keep = len(trees)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return (base, lr, trees[:keep])


def boost_clf_proba(model, X):
    base, lr, trees = model
    F = base + lr * sum((_tree_pred(t, X) for t in trees), np.zeros(len(X)))
    return 1.0 / (1.0 + np.exp(-F))


def reason_boost_multi(X, y, classes=None, **kw):
    """One-vs-rest logistic rule-boosting for multiclass. Returns {class: binary boosted model}."""
    classes = list(classes) if classes is not None else sorted(set(np.asarray(y).tolist()))
    return {c: reason_boost_clf(X, (np.asarray(y) == c).astype(int), **kw) for c in classes}


def boost_multi_predict(models, X):
    """Argmax over normalized one-vs-rest probabilities. Full coverage."""
    classes = list(models)
    P = np.stack([boost_clf_proba(models[c], X) for c in classes], 1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    return np.array(classes)[P.argmax(1)], P.max(1)


def _softmax(F):
    e = np.exp(F - F.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _hardest_class_pair(scores, target, sample_weight=None):
    """Return the class pair with the largest balanced conditional log-loss."""
    return hardest_class_pair(scores, target, sample_weight)


def _binary_probability(margin):
    """P(class zero) from the logit difference ``score_zero - score_one``."""
    return 1.0 / (1.0 + np.exp(-margin))


def _binary_margin_logloss(margin, target_zero, sample_weight=None):
    """The clipped softmax log-loss expressed directly in one binary margin."""
    signed_margin = np.where(target_zero, -margin, margin)
    loss = np.minimum(np.logaddexp(0.0, signed_margin), -math.log(1e-9))
    return float(loss.mean() if sample_weight is None else np.average(loss, weights=sample_weight))


def _binary_rank_metric(margin, target_positive, sample_weight=None, metric="auc"):
    """Weighted tie-correct AUC or average precision from the class-one margin."""
    score = -np.asarray(margin, dtype=float)
    target_positive = np.asarray(target_positive, dtype=bool)
    weights = (
        np.ones(len(score), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    )
    if metric not in {"auc", "average_precision"}:
        raise ValueError("binary validation metric must be logloss, auc, or average_precision")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    positive_weight = weights[order] * target_positive[order]
    negative_weight = weights[order] * ~target_positive[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    group_positive = np.add.reduceat(positive_weight, starts)
    group_negative = np.add.reduceat(negative_weight, starts)
    total_positive = float(group_positive.sum())
    total_negative = float(group_negative.sum())
    if total_positive <= 0 or total_negative <= 0:
        return 0.5 if metric == "auc" else 0.0
    if metric == "auc":
        negative_before = np.cumsum(group_negative) - group_negative
        concordant = np.sum(group_positive * (negative_before + 0.5 * group_negative))
        return float(concordant / (total_positive * total_negative))
    group_positive = group_positive[::-1]
    group_negative = group_negative[::-1]
    cumulative_positive = np.cumsum(group_positive)
    cumulative_total = cumulative_positive + np.cumsum(group_negative)
    precision = cumulative_positive / np.maximum(cumulative_total, 1e-12)
    return float(np.sum((group_positive / total_positive) * precision))


def _multiclass_ovo_auc(scores, target, sample_weight=None):
    """Weighted macro one-vs-one AUC from coupled softmax scores.

    Each unordered class pair contributes the mean of its two one-vs-one
    class-probability AUCs.  Scoring both directions is important: a class
    probability also depends on every class outside the current pair, so one
    direction is not generally the complement of the other.
    """
    scores = np.asarray(scores, dtype=float)
    target = np.asarray(target, dtype=np.int64)
    if scores.ndim != 2 or len(scores) != len(target):
        raise ValueError("multiclass scores must have one row per target")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None and len(weights) != len(target):
        raise ValueError("sample_weight must have one value per target")
    probability = _softmax(scores)
    pair_scores = []
    for first in range(scores.shape[1]):
        for second in range(first + 1, scores.shape[1]):
            rows = (target == first) | (target == second)
            pair_weight = None if weights is None else weights[rows]
            first_auc = _binary_rank_metric(
                -probability[rows, first],
                target[rows] == first,
                pair_weight,
                metric="auc",
            )
            second_auc = _binary_rank_metric(
                -probability[rows, second],
                target[rows] == second,
                pair_weight,
                metric="auc",
            )
            pair_scores.append(0.5 * (first_auc + second_auc))
    return float(np.mean(pair_scores)) if pair_scores else 0.5


def _fit_verifier_split(
    y,
    holdout,
    seed,
    stratified=False,
    min_verifier_events=0,
    validation_groups=None,
):
    """Deterministic fit/verifier rows, with a minority floor for rare-event fits."""
    y = np.asarray(y, dtype=np.int64)
    if validation_groups is not None:
        from tabpvn.validation import FutureValidation

        groups = np.asarray(validation_groups, dtype=np.int64)
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError("validation_groups must have one value per classification row")
        minimum_counts = None
        if min_verifier_events and np.unique(y).size == 2:
            classes, counts = np.unique(y, return_counts=True)
            rare = int(np.argmin(counts))
            rare_available = max(0, int(counts[rare] - 1))
            rare_floor = min(
                int(min_verifier_events),
                rare_available,
                max(1, int(counts[rare] // 3)),
            )
            if rare_floor:
                minimum_counts = {classes[rare]: rare_floor}
        return FutureValidation(groups).split(
            y,
            holdout=holdout,
            min_train=max(len(np.unique(y)), int(0.25 * len(y))),
            min_valid=max(len(np.unique(y)), int(0.05 * len(y))),
            require_class_coverage=True,
            min_valid_class_counts=minimum_counts,
        )
    rng = np.random.default_rng(seed)
    if not stratified:
        idx = rng.permutation(len(y))
        cutoff = int((1 - holdout) * len(y))
        return idx[:cutoff], idx[cutoff:]

    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    target = len(y) - int((1 - holdout) * len(y))
    target = int(np.clip(target, len(classes), len(y) - len(classes)))
    ideal = counts.astype(float) * (target / len(y))
    quotas = np.floor(ideal).astype(np.int64)
    quotas = np.minimum(counts - 1, np.maximum(quotas, 1))
    minimum = np.minimum(np.ones(len(classes), dtype=np.int64), counts - 1)
    if min_verifier_events and len(classes) == 2:
        rare = int(np.argmin(counts))
        rare_available = max(0, int(counts[rare] - 1))
        rare_floor = min(
            int(min_verifier_events),
            rare_available,
            max(1, int(counts[rare] // 3)),
        )
        minimum[rare] = rare_floor
        quotas[rare] = max(quotas[rare], rare_floor)

    while int(quotas.sum()) > target:
        available = np.flatnonzero(quotas > minimum)
        if not len(available):
            break
        excess = quotas[available] - ideal[available]
        chosen = int(available[np.argmax(excess)])
        quotas[chosen] -= 1
    while int(quotas.sum()) < target:
        available = np.flatnonzero(quotas < counts - 1)
        if not len(available):
            break
        deficit = ideal[available] - quotas[available]
        chosen = int(available[np.argmax(deficit)])
        quotas[chosen] += 1

    fit_parts, ver_parts = [], []
    for class_idx, quota in enumerate(quotas):
        rows = np.flatnonzero(inverse == class_idx)
        rng.shuffle(rows)
        ver_parts.append(rows[: int(quota)])
        fit_parts.append(rows[int(quota) :])
    fit = np.concatenate(fit_parts).astype(np.int64, copy=False)
    ver = np.concatenate(ver_parts).astype(np.int64, copy=False)
    rng.shuffle(fit)
    rng.shuffle(ver)
    return fit, ver


def _prior_preserving_subset_weight(y, sample_weight, rows):
    """Weight a stratified subset back to the full fitted class prior.

    ``sample_weight`` may already contain inverse-inclusion weights from a
    case-control reservoir.  The verifier split has its own class-dependent
    inclusion probability, so it needs a second correction.  Normalizing to a
    mean of one keeps the booster's regularization scale unchanged.
    """
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer) and len(y) and int(y.min()) >= 0:
        encoded = y.astype(np.int64, copy=False)
        class_count = int(encoded.max()) + 1
    else:
        _classes, encoded = np.unique(y, return_inverse=True)
        class_count = len(_classes)
    rows = np.asarray(rows, dtype=np.int64)
    source_weight = (
        np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    )
    if len(source_weight) != len(y):
        raise ValueError("sample_weight must have one value per fitted row")
    source_total = np.bincount(encoded, weights=source_weight, minlength=class_count)
    subset_weight = source_weight[rows].copy()
    subset_total = np.bincount(encoded[rows], weights=subset_weight, minlength=class_count)
    factors = np.divide(
        source_total,
        subset_total,
        out=np.zeros_like(source_total),
        where=subset_total > 0,
    )
    subset_weight *= factors[encoded[rows]]
    mean_weight = float(subset_weight.mean())
    return subset_weight if mean_weight <= 0 else subset_weight / mean_weight


def _reason_boost_binary_symmetric(  # noqa: C901 - binary boost training loop
    X,
    yi,
    classes,
    holdout,
    lr,
    rounds,
    depth,
    min_leaf,
    sub,
    mf,
    patience,
    lam,
    seed,
    nbins,
    class_weight,
    refit,
    sample_weight,
    stratified_holdout,
    min_verifier_events,
    validation_metric,
    track_validation_metrics=(),
    allowed=None,
    validation_groups=None,
):
    """Binary numeric softmax boosting in one margin with mirrored proof trees.

    For two classes, the diagonal-softmax Newton systems have opposite
    gradients and equal Hessians. Growing one class-zero tree and storing its
    sign mirror for class one preserves both certified score contributions,
    while one scalar margin is sufficient for subsequent probabilities and
    validation loss.
    """
    fit, ver = _fit_verifier_split(
        yi,
        holdout,
        seed,
        stratified=stratified_holdout,
        min_verifier_events=min_verifier_events,
        validation_groups=validation_groups,
    )
    Xf, yf, Xv, yv = X[fit], yi[fit], X[ver], yi[ver]
    source_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if stratified_holdout and validation_groups is None:
        fit_source_weight = _prior_preserving_subset_weight(yi, source_weight, fit)
        ver_source_weight = _prior_preserving_subset_weight(yi, source_weight, ver)
    else:
        fit_source_weight = None if source_weight is None else source_weight[fit]
        ver_source_weight = None if source_weight is None else source_weight[ver]
    D = X.shape[1]
    numeric_pool = (
        np.arange(D, dtype=np.int64) if allowed is None else np.unique(np.asarray(allowed, dtype=np.int64))
    )
    if numeric_pool.ndim != 1 or not len(numeric_pool):
        raise ValueError("allowed must select at least one classifier feature")
    if numeric_pool[0] < 0 or numeric_pool[-1] >= D:
        raise ValueError("allowed classifier features are out of bounds")
    if sub is None:
        sub = 1.0 if len(fit) < 2500 else 0.7
    if mf is None:
        mf = 1.0 if len(numeric_pool) < 25 else 0.7
    nfeat = min(len(numeric_pool), max(1, int(mf * len(numeric_pool))))
    compact_features = np.arange(nfeat, dtype=np.int64)
    rm = np.random.default_rng(seed * 17 + 1)
    Xbf, edges, NB = _prebin(Xf, nbins)
    cnt = np.bincount(yf, weights=fit_source_weight, minlength=2).astype(float)
    base = np.log(np.clip(cnt / cnt.sum(), 1e-6, 1))
    target_zero = yf == 0
    margin_base = float(base[0] - base[1])
    margin_fit = np.full(len(fit), margin_base)
    margin_ver = np.full(len(ver), margin_base)
    if class_weight is None:
        class_weights = np.ones(2)
    elif isinstance(class_weight, str):
        total_weight = len(yf) if fit_source_weight is None else fit_source_weight.sum()
        class_weights = total_weight / (2 * np.clip(cnt, 1, None))
    else:
        class_weights = np.asarray(class_weight, float)
    weight_fit = class_weights[yf]
    if fit_source_weight is not None:
        weight_fit = weight_fit * fit_source_weight

    metrics = tuple(dict.fromkeys((validation_metric, *track_validation_metrics)))
    supported_metrics = {"logloss", "auc", "average_precision"}
    if any(metric not in supported_metrics for metric in metrics):
        raise ValueError(f"unsupported binary validation metric: {metrics}")

    def validation_score(metric):
        if metric == "logloss":
            return -_binary_margin_logloss(margin_ver, yv == 0, ver_source_weight)
        return _binary_rank_metric(
            margin_ver,
            yv == 1,
            ver_source_weight,
            metric=metric,
        )

    best_scores = {metric: validation_score(metric) for metric in metrics}
    keeps = dict.fromkeys(metrics, 0)
    bad_rounds = dict.fromkeys(metrics, 0)
    trees, flats = [], []
    for round_idx in range(rounds):
        probability_zero = _binary_probability(margin_fit)
        selected = rm.random(len(fit)) < sub
        rest_rows = np.flatnonzero(~selected)
        has_rest = len(rest_rows) > 0
        feats = rm.choice(numeric_pool, nfeat, replace=False)
        selected_rows = np.flatnonzero(selected)
        Xb_tree, gradient_tree, hessian_tree = _gather_binary_tree_fit_data(
            Xbf, probability_zero, target_zero, weight_fit, selected_rows, feats
        )
        tree, selected_pred, flat, binned_flat = _fit_tree_2nd_binned(
            Xb_tree,
            [edges[int(j)] for j in feats],
            gradient_tree,
            hessian_tree,
            depth,
            min_leaf,
            compact_features,
            lam,
            NB,
            honest_seed=seed + 1009 * round_idx,
            parallel_hist=_should_parallel_hist(len(selected_rows), len(feats)),
            return_binned=True,
        )
        tree = _remap_tree_features(tree, feats)
        flat = _remap_flat_features(flat, feats)
        binned_flat = _remap_flat_features(binned_flat, feats)
        pred_tree = flat if flat is not None else tree
        ver_pred = _tree_pred(pred_tree, Xv)
        step = 2.0 * lr
        _add_rows_in_place(margin_fit, selected_rows, selected_pred, step)
        if has_rest and binned_flat is not None:
            _add_flat_binned_rows_in_place(binned_flat, Xbf, rest_rows, margin_fit, step)
        elif has_rest:
            _add_rows_in_place(margin_fit, rest_rows, _tree_pred_rows(pred_tree, Xf, rest_rows), step)
        margin_ver += step * ver_pred
        trees.append((0, tree))
        flats.append(flat)
        trees.append((1, _negate_tree_values(tree)))
        flats.append(_negate_flat_values(flat))
        for metric in metrics:
            score = validation_score(metric)
            if score > best_scores[metric] + 1e-6:
                best_scores[metric] = score
                keeps[metric] = len(trees)
                bad_rounds[metric] = 0
            else:
                bad_rounds[metric] += 1
        if all(bad_rounds[metric] >= patience for metric in metrics):
            break
    keep = keeps[validation_metric]
    trace = None
    if track_validation_metrics:
        trace_keep = max(keeps.values(), default=0)
        trace = {
            "trees": trees[:trace_keep],
            "flats": flats[:trace_keep],
            "tree_counts": dict(keeps),
            "scores": dict(best_scores),
        }
    if not refit or keep == 0:
        result = (
            base,
            lr,
            trees[:keep],
            classes,
            flats[:keep],
            (ver if not refit else None),
            False,
        )
        return (*result, trace) if track_validation_metrics else result

    rounds_kept = keep // 2
    Xb_all, edges_all, NB_all = _prebin(X, nbins)
    cnt_all = np.bincount(yi, weights=source_weight, minlength=2).astype(float)
    base = np.log(np.clip(cnt_all / cnt_all.sum(), 1e-6, 1))
    margin = np.full(len(yi), float(base[0] - base[1]))
    target_zero = yi == 0
    if class_weight is None:
        class_weights = np.ones(2)
    elif isinstance(class_weight, str):
        total_weight = len(yi) if source_weight is None else source_weight.sum()
        class_weights = total_weight / (2 * np.clip(cnt_all, 1, None))
    else:
        class_weights = np.asarray(class_weight, float)
    weight_all = class_weights[yi]
    if source_weight is not None:
        weight_all = weight_all * source_weight
    rm_refit = np.random.default_rng(seed * 17 + 2)
    trees, flats = [], []
    for round_idx in range(rounds_kept):
        probability_zero = _binary_probability(margin)
        selected = rm_refit.random(len(yi)) < sub
        rest_rows = np.flatnonzero(~selected)
        has_rest = len(rest_rows) > 0
        feats = rm_refit.choice(numeric_pool, nfeat, replace=False)
        selected_rows = np.flatnonzero(selected)
        Xb_tree, gradient_tree, hessian_tree = _gather_binary_tree_fit_data(
            Xb_all, probability_zero, target_zero, weight_all, selected_rows, feats
        )
        tree, selected_pred, flat, binned_flat = _fit_tree_2nd_binned(
            Xb_tree,
            [edges_all[int(j)] for j in feats],
            gradient_tree,
            hessian_tree,
            depth,
            min_leaf,
            compact_features,
            lam,
            NB_all,
            honest_seed=seed + 1_000_003 + 1009 * round_idx,
            parallel_hist=_should_parallel_hist(len(selected_rows), len(feats)),
            return_binned=True,
        )
        tree = _remap_tree_features(tree, feats)
        flat = _remap_flat_features(flat, feats)
        binned_flat = _remap_flat_features(binned_flat, feats)
        pred_tree = flat if flat is not None else tree
        step = 2.0 * lr
        _add_rows_in_place(margin, selected_rows, selected_pred, step)
        if has_rest and binned_flat is not None:
            _add_flat_binned_rows_in_place(binned_flat, Xb_all, rest_rows, margin, step)
        elif has_rest:
            _add_rows_in_place(margin, rest_rows, _tree_pred_rows(pred_tree, X, rest_rows), step)
        trees.append((0, tree))
        flats.append(flat)
        trees.append((1, _negate_tree_values(tree)))
        flats.append(_negate_flat_values(flat))
    return (base, lr, trees, classes, flats, None, False)


def reason_boost_softmax(  # noqa: C901 - multiclass boost training loop
    X,
    y,
    classes=None,
    holdout=0.3,
    lr=0.15,
    rounds=200,
    depth=4,
    min_leaf=20,
    sub=None,
    mf=None,
    patience=30,
    lam=1.0,
    fit_cap=None,
    second_order=True,
    seed=0,
    nbins=64,
    class_weight=None,
    refit=True,
    parallel_k=False,
    linear_leaf=False,
    lam_lin=1.0,
    categorical_groups=(),
    honest_categorical=False,
    sample_weight=None,
    stratified_holdout=False,
    min_verifier_events=0,
    shared_structure=False,
    max_leaves=None,
    best_first_pair=False,
    adaptive_best_first_pair=False,
    validation_metric="logloss",
    track_validation_metrics=(),
    track_residual_dynamics=False,
    feature_count=None,
    allowed=None,
    validation_groups=None,
):
    """MULTINOMIAL (softmax) gradient boosting — the coupled many-class fix for one-vs-rest. Each round
    fits one CART tree per class to the softmax residual (y_onehot - softmax(F)); Newton leaves use the
    multinomial Hessian; held-out multiclass log-loss early-stop = verifier. Histogram-binned splits keep
    it fast; sub/mf default to size-ADAPTIVE (full data+features on small sets so multiclass isn't starved,
    subsampled on large sets for speed/regularization). class_weight="balanced" (or a per-class array) scales
    each sample's gradient/hessian so the minority class isn't drowned out (imbalanced tasks). Native
    category partitions are optional; ``honest_categorical`` is an internal research path that proposes the
    finite level set on one deterministic row half and scores it on the other."""
    classes = list(classes) if classes is not None else sorted(set(np.asarray(y).tolist()))
    X = np.asarray(X, dtype=float)
    source_y = np.asarray(y)
    source_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if source_weight is not None and (source_weight.ndim != 1 or len(source_weight) != len(source_y)):
        raise ValueError("sample_weight must have one value per classification row")
    groups = None if validation_groups is None else np.asarray(validation_groups, dtype=np.int64)
    if groups is not None and (groups.ndim != 1 or len(groups) != len(source_y)):
        raise ValueError("validation_groups must have one value per classification row")
    if fit_cap is not None and len(source_y) > int(fit_cap):
        if groups is None:
            selected = np.random.default_rng(seed * 7 + 3).choice(len(source_y), int(fit_cap), replace=False)
        else:
            from tabpvn.validation import FutureValidation

            selected = FutureValidation(groups).bounded_rows(int(fit_cap))
        X, source_y = X[selected], source_y[selected]
        groups = None if groups is None else groups[selected]
        source_weight = None if source_weight is None else source_weight[selected]
    y = source_y
    cat_groups, cat_codes = _categorical_codes(X, categorical_groups)
    # The non-numba fallback keeps the legacy numeric representation exactly;
    # native categories are an acceleration-path extension, never a reason to
    # drop a usable one-hot feature on a minimal installation.
    if not _HAS_NUMBA:
        cat_groups, cat_codes = (), np.empty((len(X), 0), np.int16)
    K = len(classes)
    if best_first_pair and (
        max_leaves is None or K <= 2 or not second_order or shared_structure or bool(cat_groups)
    ):
        raise ValueError(
            "best_first_pair requires a numeric, second-order, non-shared multiclass fit with max_leaves"
        )
    if adaptive_best_first_pair and not best_first_pair:
        raise ValueError("adaptive_best_first_pair requires best_first_pair")
    if track_residual_dynamics and K <= 2:
        raise ValueError("residual-dynamics tracing currently requires a multiclass target")
    use_shared_structure = bool(
        shared_structure and K > 2 and second_order and not linear_leaf and not cat_groups
    )
    ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[v] for v in np.asarray(y).tolist()])
    if track_validation_metrics and refit:
        raise ValueError("tracked validation checkpoints require refit=False")
    if K == 2 and second_order and not linear_leaf and not cat_groups:
        return _reason_boost_binary_symmetric(
            X,
            yi,
            classes,
            holdout,
            lr,
            rounds,
            depth,
            min_leaf,
            sub,
            mf,
            patience,
            lam,
            seed,
            nbins,
            class_weight,
            refit,
            source_weight,
            stratified_holdout,
            min_verifier_events,
            validation_metric,
            track_validation_metrics,
            allowed,
            groups,
        )
    fit, ver = _fit_verifier_split(
        yi,
        holdout,
        seed,
        stratified=stratified_holdout,
        min_verifier_events=min_verifier_events,
        validation_groups=groups,
    )
    Xf, yf, Xv, yv = X[fit], yi[fit], X[ver], yi[ver]
    if stratified_holdout and groups is None:
        fit_source_weight = _prior_preserving_subset_weight(yi, source_weight, fit)
        ver_source_weight = _prior_preserving_subset_weight(yi, source_weight, ver)
    else:
        fit_source_weight = None if source_weight is None else source_weight[fit]
        ver_source_weight = None if source_weight is None else source_weight[ver]
    catf = cat_codes[fit]
    D = X.shape[1]
    feature_count = D if feature_count is None else int(feature_count)
    if not 1 <= feature_count <= D:
        raise ValueError("feature_count must select a non-empty prefix of X")
    numeric_pool = (
        np.arange(feature_count, dtype=np.int64)
        if allowed is None
        else np.unique(np.asarray(allowed, dtype=np.int64))
    )
    if numeric_pool.ndim != 1 or not len(numeric_pool):
        raise ValueError("allowed must select at least one classifier feature")
    if numeric_pool[0] < 0 or numeric_pool[-1] >= feature_count:
        raise ValueError("allowed classifier features must lie inside feature_count")
    if sub is None:
        sub = 1.0 if len(fit) < 2500 else 0.7  # adaptive: don't starve small multiclass
    if mf is None:
        mf = 1.0 if len(numeric_pool) < 25 else 0.7
    nfeat = min(len(numeric_pool), max(1, int(mf * len(numeric_pool))))
    ncat = min(len(cat_groups), max(1, int(mf * len(cat_groups)))) if cat_groups else 0
    rm = np.random.default_rng(seed * 17 + 1)
    Xbf, edges, NB = _prebin(Xf, nbins) if second_order else (None, None, None)
    cnt = np.bincount(yf, weights=fit_source_weight, minlength=K).astype(float)
    base = np.log(np.clip(cnt / cnt.sum(), 1e-6, 1))
    Ff = np.tile(base, (len(fit), 1))
    Fv = np.tile(base, (len(ver), 1))
    Yf = np.eye(K)[yf]
    if class_weight is None:
        cw = np.ones(K)  # per-class weight -> per-sample gradient scale
    elif isinstance(class_weight, str):
        total_weight = len(yf) if fit_source_weight is None else fit_source_weight.sum()
        cw = total_weight / (K * np.clip(cnt, 1, None))  # "balanced"
    else:
        cw = np.asarray(class_weight, float)
    wf = cw[yf]
    if fit_source_weight is not None:
        wf = wf * fit_source_weight

    def ll(F, yidx, weights=None):
        p = np.clip(_softmax(F), 1e-9, 1)
        loss = -np.log(p[np.arange(len(yidx)), yidx])
        return float(loss.mean() if weights is None else np.average(loss, weights=weights))

    metrics = tuple(dict.fromkeys((validation_metric, *track_validation_metrics)))
    supported_metrics = {"logloss", "auc", "macro_ovo_auc"}
    if any(metric not in supported_metrics for metric in metrics):
        raise ValueError(f"unsupported multiclass validation metric: {metrics}")

    def validation_score(metric):
        if metric == "logloss":
            return -ll(Fv, yv, ver_source_weight)
        return _multiclass_ovo_auc(Fv, yv, ver_source_weight)

    best_scores = {metric: validation_score(metric) for metric in metrics}
    keeps = dict.fromkeys(metrics, 0)
    bad_rounds = dict.fromkeys(metrics, 0)
    trees, flats = [], []
    dynamics_enabled = bool(track_residual_dynamics or adaptive_best_first_pair)
    dynamics_tracker = (
        ResidualDynamicsTracker(
            Fv,
            yv,
            ver_source_weight,
            detailed=track_residual_dynamics,
        )
        if dynamics_enabled
        else None
    )
    adaptive_pair: tuple[int, int] | tuple[()] = ()
    growth_schedule: list[tuple[int, int] | tuple[()]] = []
    return_trace = bool(track_validation_metrics or dynamics_enabled)

    def checkpoint_round():
        round_scores = {}
        for metric in metrics:
            score = validation_score(metric)
            round_scores[metric] = score
            if score > best_scores[metric] + 1e-6:
                best_scores[metric] = score
                keeps[metric] = len(trees)
                bad_rounds[metric] = 0
            else:
                bad_rounds[metric] += 1
        return all(bad_rounds[metric] >= patience for metric in metrics), round_scores

    def complete_round(round_update, growth_pair):
        nonlocal adaptive_pair
        stopped, round_scores = checkpoint_round()
        if dynamics_tracker is not None:
            if round_update is None:
                raise RuntimeError("residual-dynamics round completed without verifier updates")
            adaptive_pair = dynamics_tracker.observe(
                Fv,
                round_update,
                growth_pair,
                validation_scores=round_scores,
            )
            growth_schedule.append(tuple(growth_pair) if growth_pair else ())
        return stopped

    # The generic K per-class trees in a round are independent, so large deploy
    # fits may grow them concurrently.
    from concurrent.futures import ThreadPoolExecutor

    _pool = (
        ThreadPoolExecutor(max_workers=K)
        if (parallel_k and K > 1 and second_order and not use_shared_structure)
        else None
    )

    def _map_k(fn, tree_parallel_hist):
        if _pool is not None and not tree_parallel_hist:
            return list(_pool.map(fn, range(K)))
        return [fn(k) for k in range(K)]

    try:
        for round_idx in range(rounds):
            pf = _softmax(Ff)
            # The prior-only scores in round zero express class prevalence, not
            # a learned boundary. Keep that round symmetric and allocate the
            # asymmetric leaf-wise budget only after residual evidence exists.
            proposed_pair = (
                ()
                if best_first_pair and round_idx == 0
                else (_hardest_class_pair(Ff, yf, wf) if best_first_pair else None)
            )
            # Fit residuals propose the pair. Leak-safe verifier dynamics grant
            # or deny extra capacity without replacing that proposal.
            hard_pair = proposed_pair
            if adaptive_best_first_pair and not adaptive_pair:
                hard_pair = ()
            round_update_v = np.zeros_like(Fv) if dynamics_tracker is not None else None
            s = rm.random(len(fit)) < sub
            feats = rm.choice(numeric_pool, nfeat, replace=False)
            cat_feats = rm.choice(len(cat_groups), ncat, replace=False) if ncat else np.empty(0, np.int64)
            rest = ~s
            has_rest = bool(rest.any())
            Xf_rest = Xf[rest] if has_rest else None
            Xbs = Xbf[s] if second_order else None
            catfs = catf[s] if second_order else None

            # A class pool already fills independent cores on the deploy path.
            # Do not replace it with a slower nested-free team for ordinary
            # multiclass fitting.
            tree_parallel_hist = _should_parallel_hist(int(s.sum()), len(feats)) and not (
                parallel_k and K > 1
            )
            Xfs = Xf[s]

            if use_shared_structure:
                gradient = wf[:, None] * (pf - Yf)
                hessian = wf[:, None] * pf * (1.0 - pf)
                shared_trees, selected_pred, shared_flats, _shared_binned = _fit_tree_softmax_shared_binned(
                    Xbs,
                    edges,
                    gradient[s],
                    hessian[s],
                    depth=depth,
                    min_leaf=min_leaf,
                    feats=feats,
                    lam=lam,
                    NB=NB,
                )
                for k, (tree, flat) in enumerate(zip(shared_trees, shared_flats, strict=False)):
                    Ff[s, k] += lr * selected_pred[:, k]
                    if has_rest:
                        Ff[rest, k] += lr * _tree_pred(flat, Xf_rest)
                    verifier_update = lr * _tree_pred(flat, Xv)
                    Fv[:, k] += verifier_update
                    if round_update_v is not None:
                        round_update_v[:, k] = verifier_update
                    trees.append((k, tree))
                    flats.append(flat)
                if complete_round(round_update_v, hard_pair):
                    break
                continue

            def _grow(
                k,
                pf=pf,
                s=s,
                feats=feats,
                cat_feats=cat_feats,
                Xf_rest=Xf_rest,
                has_rest=has_rest,
                Xbs=Xbs,
                catfs=catfs,
                tree_parallel_hist=tree_parallel_hist,
                Xfs=Xfs,
                round_idx=round_idx,
                hard_pair=hard_pair,
            ):  # pure per-class work over an immutable snapshot of the round
                gk = wf * (pf[:, k] - Yf[:, k])
                hk = wf * pf[:, k] * (1 - pf[:, k])  # class-weighted grad/hess
                if second_order:  # Hessian-weighted binned splits per class
                    pair_member = hard_pair is not None and k in hard_pair
                    tree_depth = max(int(depth), min(12, int(max_leaves) - 1)) if pair_member else depth
                    tree_max_leaves = max_leaves if hard_pair is None or pair_member else None
                    t, pk, flat = _fit_tree_2nd_binned(
                        Xbs,
                        edges,
                        gk[s],
                        hk[s],
                        tree_depth,
                        min_leaf,
                        feats,
                        lam,
                        NB,
                        cat_codes=catfs,
                        cat_groups=cat_groups,
                        cat_feats=cat_feats,
                        honest_categorical=honest_categorical,
                        honest_seed=seed + 1009 * round_idx + 7919 * k,
                        parallel_hist=tree_parallel_hist,
                        max_leaves=tree_max_leaves,
                    )
                    if (
                        linear_leaf
                    ):  # logit-ridge leaves (residual −gk/hk, weight hk); affine -> predict all rows
                        t = _fit_leaf_lin(t, Xfs, -gk[s] / hk[s], hk[s], lam_lin, lam)
                        return k, t, flat, None, None, _affine_tree_pred(t, Xf), _affine_tree_pred(t, Xv)
                    pred_tree = flat if flat is not None else t
                    drest = _tree_pred(pred_tree, Xf_rest) if has_rest else None
                    return k, t, flat, pk, drest, None, _tree_pred(pred_tree, Xv)
                t = _fit_tree(Xfs, (Yf[:, k] - pf[:, k])[s], depth, min_leaf, feats)
                t = _newton_leaves(t, Xfs, gk[s], hk[s], lam)
                flat = _flatten_tree(t)
                return k, t, flat, None, None, _tree_pred(flat, Xf), _tree_pred(flat, Xv)

            for k, t, flat, pk, drest, dfull, dv in _map_k(_grow, tree_parallel_hist):
                if second_order and not linear_leaf:
                    Ff[s, k] += lr * pk  # subsample rows free (leaf values from build)
                    if drest is not None:
                        Ff[rest, k] += lr * drest
                else:
                    Ff[:, k] += lr * dfull
                verifier_update = lr * dv
                Fv[:, k] += verifier_update
                if round_update_v is not None:
                    round_update_v[:, k] = verifier_update
                trees.append((k, t))
                flats.append(flat)
            if complete_round(round_update_v, hard_pair):
                break
        keep = keeps[validation_metric]
        trace = None
        if return_trace:
            trace = {}
            if track_validation_metrics:
                trace_keep = max(keeps.values(), default=0)
                trace.update(
                    trees=trees[:trace_keep],
                    flats=flats[:trace_keep],
                    tree_counts=dict(keeps),
                    scores=dict(best_scores),
                )
            if dynamics_tracker is not None:
                dynamics_records = [dict(record) for record in dynamics_tracker.records]
                selected_rounds = keep // K
                for record in dynamics_records:
                    record["selected"] = int(record["round"]) < selected_rounds
                trace.update(
                    residual_dynamics=dynamics_records,
                    pair_growth_schedule=list(growth_schedule),
                    selected_rounds=selected_rounds,
                )
        if not refit or keep == 0:
            # When not refitting, `ver` is a genuine out-of-fit holdout (the model never trained on it) — expose
            # it so the caller can calibrate the certified layer on the deploy model's own leak-safe residuals
            # instead of fitting a separate OOF ensemble. (After a refit there is no such holdout -> None below.)
            result = (
                base,
                lr,
                trees[:keep],
                classes,
                flats[:keep],
                (ver if not refit else None),
                linear_leaf,
            )
            return (*result, trace) if return_trace else result
        # REFIT on ALL the data for the early-stopped round count (was trained only on the fit slice — this
        # recovers the held-out labels; the regression booster already does this). base recomputed on full y.
        rounds_kept = keep // K
        Xb_all, edges_all, NB_all = _prebin(X, nbins) if second_order else (None, None, None)
        cnt_all = np.bincount(yi, weights=source_weight, minlength=K).astype(float)
        base = np.log(np.clip(cnt_all / cnt_all.sum(), 1e-6, 1))
        F = np.tile(base, (len(yi), 1))
        Y = np.eye(K)[yi]
        total_weight = len(yi) if source_weight is None else source_weight.sum()
        w2 = (
            np.ones(K)
            if class_weight is None
            else (
                total_weight / (K * np.clip(cnt_all, 1, None))
                if isinstance(class_weight, str)
                else np.asarray(class_weight, float)
            )
        )[yi]
        if source_weight is not None:
            w2 = w2 * source_weight
        rm2 = np.random.default_rng(seed * 17 + 2)
        trees, flats = [], []
        deployed_growth_schedule = []
        for round_idx in range(rounds_kept):
            p = _softmax(F)
            hard_pair = (
                (_hardest_class_pair(F, yi, w2) if growth_schedule[round_idx] else ())
                if adaptive_best_first_pair
                else (
                    ()
                    if best_first_pair and round_idx == 0
                    else (_hardest_class_pair(F, yi, w2) if best_first_pair else None)
                )
            )
            if adaptive_best_first_pair:
                deployed_growth_schedule.append(tuple(hard_pair) if hard_pair else ())
            s = rm2.random(len(yi)) < sub
            feats = rm2.choice(numeric_pool, nfeat, replace=False)
            cat_feats = rm2.choice(len(cat_groups), ncat, replace=False) if ncat else np.empty(0, np.int64)
            rest = ~s
            has_rest = bool(rest.any())
            Xrest = X[rest] if has_rest else None
            Xbs = Xb_all[s] if second_order else None
            catfs = cat_codes[s] if second_order else None

            tree_parallel_hist = _should_parallel_hist(int(s.sum()), len(feats)) and not (
                parallel_k and K > 1
            )
            Xs = X[s]

            if use_shared_structure:
                gradient = w2[:, None] * (p - Y)
                hessian = w2[:, None] * p * (1.0 - p)
                shared_trees, selected_pred, shared_flats, _shared_binned = _fit_tree_softmax_shared_binned(
                    Xbs,
                    edges_all,
                    gradient[s],
                    hessian[s],
                    depth=depth,
                    min_leaf=min_leaf,
                    feats=feats,
                    lam=lam,
                    NB=NB_all,
                )
                for k, (tree, flat) in enumerate(zip(shared_trees, shared_flats, strict=False)):
                    F[s, k] += lr * selected_pred[:, k]
                    if has_rest:
                        F[rest, k] += lr * _tree_pred(flat, Xrest)
                    trees.append((k, tree))
                    flats.append(flat)
                continue

            def _grow_r(
                k,
                p=p,
                s=s,
                feats=feats,
                cat_feats=cat_feats,
                Xrest=Xrest,
                has_rest=has_rest,
                Xbs=Xbs,
                catfs=catfs,
                tree_parallel_hist=tree_parallel_hist,
                Xs=Xs,
                round_idx=round_idx,
                hard_pair=hard_pair,
            ):
                gk = w2 * (p[:, k] - Y[:, k])
                hk = w2 * p[:, k] * (1 - p[:, k])
                if second_order:
                    pair_member = hard_pair is not None and k in hard_pair
                    tree_depth = max(int(depth), min(12, int(max_leaves) - 1)) if pair_member else depth
                    tree_max_leaves = max_leaves if hard_pair is None or pair_member else None
                    t, pk, flat = _fit_tree_2nd_binned(
                        Xbs,
                        edges_all,
                        gk[s],
                        hk[s],
                        tree_depth,
                        min_leaf,
                        feats,
                        lam,
                        NB_all,
                        cat_codes=catfs,
                        cat_groups=cat_groups,
                        cat_feats=cat_feats,
                        honest_categorical=honest_categorical,
                        honest_seed=seed + 1_000_003 + 1009 * round_idx + 7919 * k,
                        parallel_hist=tree_parallel_hist,
                        max_leaves=tree_max_leaves,
                    )
                    if linear_leaf:
                        t = _fit_leaf_lin(t, Xs, -gk[s] / hk[s], hk[s], lam_lin, lam)
                        return k, t, flat, None, None, _affine_tree_pred(t, X)
                    pred_tree = flat if flat is not None else t
                    drest = _tree_pred(pred_tree, Xrest) if has_rest else None
                    return k, t, flat, pk, drest, None
                t = _fit_tree(Xs, (Y[:, k] - p[:, k])[s], depth, min_leaf, feats)
                t = _newton_leaves(t, Xs, gk[s], hk[s], lam)
                flat = _flatten_tree(t)
                return k, t, flat, None, None, _tree_pred(flat, X)

            for k, t, flat, pk, drest, dfull in _map_k(_grow_r, tree_parallel_hist):
                if second_order and not linear_leaf:
                    F[s, k] += lr * pk  # subsample rows free
                    if drest is not None:
                        F[rest, k] += lr * drest
                else:
                    F[:, k] += lr * dfull
                trees.append((k, t))
                flats.append(flat)
        if trace is not None and adaptive_best_first_pair:
            trace["deployed_pair_growth_schedule"] = deployed_growth_schedule
        result = (
            base,
            lr,
            trees,
            classes,
            flats,
            None,
            linear_leaf,
        )  # refit used all rows -> no leak-safe holdout
        return (*result, trace) if return_trace else result
    finally:
        if _pool is not None:
            _pool.shutdown(wait=False)


def boost_softmax_predict(model, X):
    base, lr, trees, classes = model[0], model[1], model[2], model[3]
    linear = len(model) > 6 and model[6]  # linear-leaf model -> affine tuple prediction
    F = np.tile(base, (len(X), 1))
    if linear:
        for k, t in trees:
            F[:, k] += lr * _affine_tree_pred(t, X)
    else:
        preds = model[4] if len(model) > 4 else [t for _, t in trees]  # flats when available (no re-flatten)
        for (k, t), pr in zip(trees, preds, strict=False):
            F[:, k] += lr * _tree_pred(t if pr is None else pr, X)
    return np.array(classes)[F.argmax(1)], _softmax(F).max(1)
