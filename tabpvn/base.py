"""TabPVN — a Proposer–Verifier Network for proof-carrying tabular prediction.

One principle, three modes, one verifier:

    an untrusted PROPOSER proposes candidate structure — a gradient-boosted ensemble of threshold
    regions (tabular) or τ-gated Horn rules (relational);
    the sound FOLKernel VERIFIES it by closure (modus ponens over the fired clauses), reproducing
    every model answer exactly;
    each answer carries a machine-checkable execution PROOF and an explicit statistical-support scope,
    and ABSTAINS where the configured calibration criterion is not met.

    mode            proposer                             guarantee
    --------------  -----------------------------------  --------------------------------
    regression      additive boosted threshold regions   conformal bounded error
    classification  additive per-class region ensemble   regional precision LB / robustness
    relational      τ-gated relation-chain Horn rules     precision on the graph

`TabPVN().fit(data, y=None, target=...)` auto-detects the mode and self-configures (hyperparameters,
monotone invariants, calibration, class balancing) with no user knobs:
    tabular    -> fit(X, y)  or  fit(dataframe, y)   (raw categoricals / missing values handled)
    events     -> fit(dataframe, y)  (entity/time/value roles discovered and gated automatically)
    relational -> fit([(rel, head, tail), ...], target="relation_to_learn")

Common API: predict / predict_proba, confidence, certificate(X, row) (proof + bound + stability +
sufficient-reason), recourse, certify (soundness), explain, rules; relational adds derive / query.

"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from core.fol_kg_induction import induce
from core.fol_kg_induction import verify as _kg_verify
from core.kernel_fol import FOLKernel
from tabpvn.certified_boost import AdditiveCertifiedClassifier, AdditiveCertifiedRegressor, _fit_sample
from tabpvn.operating_points import fit_binary_thresholds as _fit_binary_thresholds
from tabpvn.preprocessing import (
    _is_classification,
    _onehot_group_metadata,
    _onehot_groups,
    _Preprocessor,
)
from tabpvn.preprocessing import (
    target_encode as target_encode,
)
from tabpvn.proofs import (
    TargetAttestation,
    build_proof_artifact,
    is_proof_artifact,
    is_proof_response,
    machine_payloads,
    proof_response_matches_artifact,
    public_proof_response,
    response_matches_machine,
    verify_structured_payload,
)
from tabpvn.proposers import (
    AffineLogitRead,
    CategoricalPosteriorChallenger,
    NumericIntervalPosteriorChallenger,
    TemporalEvidenceChallenger,
    default_registry,
    gate_report,
)
from tabpvn.relational import _is_relational
from tabpvn.relational import derive_features as derive_features
from tabpvn.trees import _flat_leaf_ids
from tabpvn.validation import FutureValidation

# Config-search parallelism: independent candidate configs can run concurrently when Numba's threading
# backend permits calls from multiple Python threads. The fallback workqueue backend does not, so _pmap
# keeps the outer schedule serial there while the tree kernels retain their own feature-level parallelism.
# All randomness is drawn BEFORE the fits (see _successive_halving), so either schedule is deterministic.
_SEARCH_WORKERS = min(4, (os.cpu_count() or 2))

# Above this many rows the final deploy fit skips its refit-on-all-data pass: refit exists to recover the
# ~30% held-out slice used for early stopping, but at large n the 70% fit slice already has abundant data, so
# refit roughly DOUBLES the deploy-fit cost for a measured ~6e-4 accuracy change. Small n keeps refit (there
# the held-out labels matter). Calibration OOF and the config search never refit.
_REFIT_MAX_N = 50_000

# Exact boosting revisits every fitted row at every stage, so unbounded rows make a nominally in-memory
# estimator scale linearly into hour/day fits. The zero-knob path therefore uses a deterministic bounded
# reservoir (class-proportional for classification). On the 5.6875M-row HIGGS protocol, 2M rows + a 50K
# verifier + 1000 rounds matched
# full-data AUC while fitting 4x faster and using 38% less peak RSS. Explicit configs retain the full-data path.
_BOOST_FIT_MAX_ROWS = 2_000_000
_BOOST_VERIFY_MAX_ROWS = 50_000
_BOOST_SATURATED_CLF_ROUNDS = 800
_BOOST_LARGE_CLF_ROUNDS = 1_000

_RARE_EVENT_MAX_RATE = 0.05
_RARE_CANDIDATE_MAX_RATE = 0.10
_RARE_ARCH_GATE_MAX_ROWS = 6_000
_RARE_ARCH_GATE_MIN_EVENTS = 8
_RARE_ARCH_GATE_MIN_AUC_GAIN = 0.001
_RARE_ARCH_GATE_MIN_AP_GAIN = 0.005
_RARE_ARCH_GATE_MAX_SECONDARY_LOSS = 0.002
_RARE_RESERVOIR_MIN_EVENTS = 20_000
_RARE_VERIFY_MIN_EVENTS = 500
_RARE_TUNE_MIN_EVENTS = 1_000
_RARE_RULE_GATE_MAX_ROWS = 10_000
_RARE_RULE_GATE_MAX_EVENTS = 2_000
_RARE_RULE_GATE_MIN_EVENTS = 16
_RARE_RULE_MIN_FOLD_GAIN = 0.002
_RARE_RULE_MIN_MEAN_GAIN = 0.005

_MULTICLASS_RULE_GATE_MIN_ROWS = 400
_MULTICLASS_RULE_GATE_MAX_ROWS = 4_000
_MULTICLASS_RULE_GATE_MAX_CLASSES = 20
_MULTICLASS_RULE_GATE_MIN_CLASS_ROWS = 8
_MULTICLASS_RULE_GATE_ROUNDS = 160
_MULTICLASS_RULE_MIN_FOLD_GAIN = 0.001
_MULTICLASS_RULE_MIN_MEAN_GAIN = 0.002
_MULTICLASS_HEAD_SCREEN_MIN_FOLD_GAIN = -0.002

# A dominant multiclass prior can bury an OOF-admitted minority rank signal in
# the softmax denominator. A fixed geometric half-step toward a uniform prior
# is a low-capacity challenger: it is admitted only on strong imbalance, needs
# a material pooled macro-OVO lift, and may not lose on an independently fitted
# fold. The certified class is projected back exactly, so this surface can
# refine ranking but never claim decision authority.
_MULTICLASS_PRIOR_RANK_STRENGTH = 0.5
_MULTICLASS_PRIOR_RANK_MIN_DOMINANT_RATE = 0.80
_MULTICLASS_PRIOR_RANK_MIN_GAIN = 0.005
_MULTICLASS_PRIOR_RANK_MIN_FOLD_GAIN = 0.001
_MULTICLASS_NO_SIGNAL_MIN_DOMINANT_RATE = 0.90
_MULTICLASS_NO_SIGNAL_NULL_Z = 1.645
_MULTICLASS_NO_SIGNAL_MIN_LOG_LOSS_GAIN = 0.001
_MULTICLASS_SMOOTH_MIN_RANK_GAIN = 0.001
_MULTICLASS_SMOOTH_MAX_FOLD_LOSS = 0.0
_SMOOTH_FIXED_NEIGHBORS = 15
_MULTICLASS_ADAPTIVE_SMOOTH_MAX_WEIGHT = 0.6
_MULTICLASS_ADAPTIVE_SMOOTH_WEIGHT_MULTIPLIER = 1.5
_BINARY_SMOOTH_MAX_EFFECTIVE_DIMENSIONS = 32.0
_BINARY_SMOOTH_MAX_EFFECTIVE_FRACTION = 0.5


def _large_fit_budget(boost, n_rows, mode, n_classes=None):
    """Resolve the bounded zero-knob budget for a source table."""
    if n_rows <= _BOOST_FIT_MAX_ROWS:
        return boost
    boost.setdefault("fit_cap", _BOOST_FIT_MAX_ROWS)
    boost.setdefault("holdout", _BOOST_VERIFY_MAX_ROWS / _BOOST_FIT_MAX_ROWS)
    if (
        mode == "classification"
        and n_classes == 2
        and int(boost.get("rounds", 0)) >= _BOOST_SATURATED_CLF_ROUNDS
    ):
        # The search reached its ceiling. Give the bounded final fit a modest
        # capacity extension; early stopping still chooses the deployed prefix
        # when the extra stages do not help.
        boost["rounds"] = max(int(boost["rounds"]), _BOOST_LARGE_CLF_ROUNDS)
    return boost


# The certified-confidence layer calibrates a conformal quantile (regression) / precision bound
# (classification) from out-of-fold predictions. That is a STATISTIC over the calibration set — stable on a
# few tens of thousands of rows, not all n. Above this many rows we calibrate on a random subsample, so the
# confidence phase (leak-safe OOF fits, ~a quarter of large-n fit) stops scaling with n while the guarantee
# (empirical coverage ≥ 1−α) is preserved.
_CONF_MAX_N = 50_000

# Affine leaves are an optional representation choice, not the deployed
# predictor itself.  A fixed, stratified audit is statistically stable at this
# scale and prevents the all-fold gate from training several full million-row
# boosters merely to reject the candidate.
_LINEAR_LEAF_GATE_MAX_N = 50_000

# The categorical evidence member is a local, exact-fact memory rather than a
# learned embedding. Its OOF gate is useful in the small-table regime; above
# this cap the quadratic category-overlap read is not a sensible default cost.
_CATEGORY_MEMORY_MAX_N = 2_500

# Count tables are linear in rows and bounded in category-pair families, so the
# posterior challenger can reuse the existing small-table OOF models throughout
# the full regime where those models are already built.
_CATEGORY_POSTERIOR_MAX_N = 10_000
_CATEGORY_POSTERIOR_WEIGHTS = (0.1, 0.25, 0.5, 1.0)
_CATEGORY_POSTERIOR_MIN_RANK_GAIN = 0.003
_CATEGORY_POSTERIOR_MIN_FOLD_RANK_GAIN = 0.001
_CATEGORY_POSTERIOR_MIN_PAIRED_Z = 2.0
_CATEGORY_POSTERIOR_MAX_RANK_REGRESSION = 0.001

# Numeric intervals are an accuracy-only decision challenger. Their probability
# update is never exposed through predict_proba, so Arena ranking remains on the
# independently gated probability stack. A bounded triple can act only when the
# incumbent single/pair update preserves the baseline label. Admission still
# needs broad OOF support: adjacent discounts, majority-fold wins, and tightly
# bounded fold harm.
_NUMERIC_INTERVAL_MAX_N = 10_000
_NUMERIC_INTERVAL_MIN_ACCURACY_GAIN = 0.005
_NUMERIC_INTERVAL_MIN_PAIRED_Z = 2.0
_NUMERIC_INTERVAL_MAX_FOLD_LOSS = 0.0
_NUMERIC_INTERVAL_SUPPORT_MAX_FOLD_LOSS = 0.01
_NUMERIC_INTERVAL_MIN_SUPPORTING_WEIGHTS = 2
_NUMERIC_INTERVAL_SMOOTHING = "hierarchical"
_NUMERIC_INTERVAL_MIN_RANK_GAIN = 0.002
_NUMERIC_INTERVAL_MIN_FOLD_RANK_GAIN = 0.001

# At the current small-table gate a batched dense score matrix is faster than
# per-query postings work.  The two backends share an explicit exact tie rule;
# postings takes over automatically once the dense allocation stops being the
# better serving primitive.
_CATEGORY_MEMORY_DENSE_READ_MAX_N = 4_096

# Proof-path memory is deliberately bounded to the data regime where the
# shared three-fold OOF is already built for the local proposer. It therefore
# adds no extra boosted-model fits to the default path.
_PROOF_PATH_MEMORY_MAX_N = 10_000
_WIDE_SCREEN_MIN_FEATURES = 512
_WIDE_SCREEN_MAX_FEATURES = 256
_WIDE_SCREEN_FINAL_EVIDENCE_ROWS = 20_000
_WIDE_SCREEN_MIN_MEAN_GAIN = 0.005
_RANK_CHECKPOINT_MAX_ROWS = 4_000
_RANK_CHECKPOINT_ROUNDS = 400
_RANK_CHECKPOINT_MIN_GAIN = 0.001
_PROOF_PATH_MEMORY_MAX_TREES = 16
_PROOF_PATH_MEMORY_SUPPORT_MULTIPLIER = 8
_PROOF_PATH_MEMORY_MIN_RANK_GAIN = 0.006
_PROOF_PATH_MEMORY_MIN_FOLD_GAIN = 0.003

# A globally regularized affine logit complements the booster's local regions
# on small, compact tables.  Its prior blend weight tapers with sample size;
# OOF evidence may reject that one prespecified candidate but never tunes it.
_AFFINE_RANK_MAX_N = 10_000
_AFFINE_RANK_MAX_FEATURES = 512
_AFFINE_RANK_INVERSE_REGULARIZATION = 0.03
_AFFINE_RANK_MIN_GAIN = 0.003
_AFFINE_RANK_MIN_FOLD_GAIN = 0.0
_AFFINE_DECISION_MIN_ACCURACY_GAIN = 0.005
_AFFINE_DECISION_MIN_PAIRED_Z = 2.0
_AFFINE_DECISION_MAX_RANK_REGRESSION = 0.001
_AFFINE_MIXED_MIN_STRUCTURED_FEATURES = 32
_AFFINE_MIXED_MAX_TOKEN_FRACTION = 0.5


def _pmap_workers(n_items):
    """Return a safe outer worker count for fits that may enter Numba parallel regions."""
    workers = min(int(n_items), int(_SEARCH_WORKERS))
    if workers <= 1:
        return max(workers, 1)
    from tabpvn.trees import warm_numba

    warm_numba()  # initialize the selected threading layer before querying it
    try:
        from numba import threading_layer

        if threading_layer() == "workqueue":
            return 1
    except (ImportError, RuntimeError, ValueError):
        # An unavailable/uninitialized backend cannot be proven safe for concurrent entry.
        return 1
    return workers


def _pmap(thunks):
    """Evaluate pure keyed thunks in deterministic order using a backend-safe outer schedule."""
    items = list(thunks.items())
    workers = _pmap_workers(len(items))
    if workers <= 1:
        return {k: fn() for k, fn in items}
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [(k, ex.submit(fn)) for k, fn in items]
        return {k: f.result() for k, f in futs}


def _smooth_weight(n):
    """Blend weight for the smooth k-NN proposer as a function of train size — a PRIOR schedule, not a
    per-dataset learned value (learning it is worse on small data: the meta-choice is itself variance-limited).
    Strong at small n (variance-dominated, where a smooth prior helps), linearly tapering to 0 by 10k rows
    (past that the booster dominates and the blend contributes ~nothing — so neither the member nor its OOF
    gate is built there, bounding fit cost), so medium/large data is untouched."""
    return float(np.clip(0.40 * (10000 - n) / 9000.0, 0.0, 0.40))


def _affine_rank_weight(n):
    """Prior blend weight for global affine evidence, tapering to zero by 10k rows."""
    return float(np.clip(0.50 * (10000 - n) / 9000.0, 0.0, 0.50))


def _adaptive_smooth_neighbors(n):
    """Deterministic local-memory width for a fitted row count."""
    rows = max(int(n), 1)
    return min(max(rows - 1, 1), max(2, int(np.ceil(np.sqrt(rows)))))


def _smooth_effective_dimensions(feature_weights):
    """Participation ratio of the weighted Euclidean distance metric."""
    weights = np.asarray(feature_weights, dtype=float)
    contribution = np.square(weights)
    denominator = float(np.square(contribution).sum())
    if weights.ndim != 1 or not len(weights) or denominator <= 0.0:
        return 0.0
    return float(contribution.sum() ** 2 / denominator)


def _binary_smooth_metric_is_diffuse(feature_weights):
    """Whether a binary local metric is too diffuse for stable neighbours."""
    effective = _smooth_effective_dimensions(feature_weights)
    return bool(
        effective > _BINARY_SMOOTH_MAX_EFFECTIVE_DIMENSIONS
        and effective > _BINARY_SMOOTH_MAX_EFFECTIVE_FRACTION * len(feature_weights)
    )


def _preserve_certified_class(base, candidate):
    """Keep an auxiliary probability read inside the certified booster's class region.

    The local and text members may refine probability ranking, but the FOL
    kernel certifies the boosted class. A binary probability that crosses the
    boundary is moved pointwise between that row's incumbent confidence and
    the boundary; already-valid rows remain unchanged. The result is therefore
    identical alone or inside any batch. The multiclass projection pools every
    score that competes with the certified class, then breaks the tie in its
    favour.
    """
    base, candidate = np.asarray(base, float), np.asarray(candidate, float)
    if base.shape != candidate.shape or base.ndim != 2:
        raise ValueError("base and candidate probabilities must have the same 2-D shape")
    chosen = base.argmax(1)
    out = candidate.copy()
    if base.shape[1] == 2:
        positive = chosen == 1
        p = out[:, 1]
        negative_upper = np.nextafter(0.5, 0.0)
        positive_lower = np.nextafter(0.5, 1.0)
        negative_crossing = ~positive & (p >= 0.5)
        if negative_crossing.any():
            anchor = np.minimum(base[negative_crossing, 1], negative_upper)
            fraction = np.clip(2.0 * p[negative_crossing] - 1.0, 0.0, 1.0)
            p[negative_crossing] = anchor + (negative_upper - anchor) * fraction
        positive_crossing = positive & (p <= 0.5)
        if positive_crossing.any():
            anchor = np.maximum(base[positive_crossing, 1], positive_lower)
            fraction = np.clip(2.0 * p[positive_crossing], 0.0, 1.0)
            p[positive_crossing] = positive_lower + (anchor - positive_lower) * fraction
        out[:, 1] = p
        out[:, 0] = 1.0 - p
        return out
    for row, cls in enumerate(chosen):
        pooled = np.flatnonzero(out[row] >= out[row, cls])
        if len(pooled) == 1 and pooled[0] == cls:
            continue
        level = float(out[row, pooled].mean())
        eps = np.spacing(level) if level else np.finfo(float).eps
        out[row, pooled] = level - eps
        out[row, cls] = level + eps * (len(pooled) - 1)
    return out


def _multiclass_prior_rank_projection(base, prior, strength=_MULTICLASS_PRIOR_RANK_STRENGTH):
    """Temper an imbalanced class prior without changing the certified class."""
    from tabpvn.bayes import prior_shift

    base = np.asarray(base, dtype=float)
    prior = np.asarray(prior, dtype=float)
    if base.ndim != 2 or base.shape[1] < 3:
        raise ValueError("multiclass prior ranking requires at least three probability columns")
    if prior.shape != (base.shape[1],):
        raise ValueError("prior must align with the probability columns")
    uniform = np.full(base.shape[1], 1.0 / base.shape[1])
    shifted = prior_shift(base, prior, uniform, strength=strength)
    return _preserve_certified_class(base, shifted)


def _classification_rank_score(yidx, proba):
    """The benchmark-aligned ranking statistic for a probability matrix."""
    from sklearn.metrics import roc_auc_score

    yidx, proba = np.asarray(yidx), np.asarray(proba, float)
    if proba.shape[1] == 2:
        return float(roc_auc_score(yidx, proba[:, 1]))
    return float(roc_auc_score(yidx, proba, multi_class="ovo", average="macro"))


def _global_probability_candidate_evaluation(
    encoded,
    baseline,
    candidate,
    evidence_rows,
    splits,
    *,
    composition,
    decision_eligible,
):
    """Apply the shared rank and decision contract to one global probability read."""
    baseline_prediction = baseline.argmax(1)
    baseline_correct = baseline_prediction == encoded
    baseline_score = _classification_rank_score(encoded[evidence_rows], baseline[evidence_rows])
    baseline_accuracy = float(baseline_correct[evidence_rows].mean())

    def probability_loss(probability):
        return float(
            -np.log(
                np.clip(
                    probability[evidence_rows, encoded[evidence_rows]],
                    1e-300,
                    1.0,
                )
            ).mean()
        )

    baseline_loss = probability_loss(baseline)
    projected = _preserve_certified_class(baseline, candidate)
    if not np.array_equal(projected.argmax(1), baseline_prediction):
        raise ValueError("global probability projection changed the incumbent class")
    projected_score = _classification_rank_score(encoded[evidence_rows], projected[evidence_rows])
    projected_fold_deltas = np.asarray(
        [
            _classification_rank_score(encoded[valid], projected[valid])
            - _classification_rank_score(encoded[valid], baseline[valid])
            for _train, valid in splits
        ],
        dtype=float,
    )
    projected_loss = probability_loss(projected)
    rank_selected = bool(
        projected_score - baseline_score >= _AFFINE_RANK_MIN_GAIN
        and len(projected_fold_deltas)
        and np.all(projected_fold_deltas >= _AFFINE_RANK_MIN_FOLD_GAIN)
    )

    decision_prediction = candidate.argmax(1)
    decision_correct = decision_prediction == encoded
    decision_accuracy = float(decision_correct[evidence_rows].mean())
    fold_net_wins = []
    fold_accuracy_deltas = []
    for _train, valid in splits:
        fold_wins = int(np.sum(decision_correct[valid] & ~baseline_correct[valid]))
        fold_losses = int(np.sum(~decision_correct[valid] & baseline_correct[valid]))
        fold_net_wins.append(fold_wins - fold_losses)
        fold_accuracy_deltas.append((fold_wins - fold_losses) / len(valid))
    wins = int(np.sum(decision_correct[evidence_rows] & ~baseline_correct[evidence_rows]))
    losses = int(np.sum(~decision_correct[evidence_rows] & baseline_correct[evidence_rows]))
    paired_z = (wins - losses) / np.sqrt(max(wins + losses, 1))
    decision_score = _classification_rank_score(encoded[evidence_rows], candidate[evidence_rows])
    decision_fold_deltas = np.asarray(
        [
            _classification_rank_score(encoded[valid], candidate[valid])
            - _classification_rank_score(encoded[valid], baseline[valid])
            for _train, valid in splits
        ],
        dtype=float,
    )
    decision_loss = probability_loss(candidate)
    accuracy_gain = decision_accuracy - baseline_accuracy
    log_loss_tolerance = 1.0 / np.sqrt(max(len(evidence_rows), 1))
    strict_fold_requirement = (len(splits) + 1) // 2
    decision_selected = bool(
        decision_eligible
        and accuracy_gain >= _AFFINE_DECISION_MIN_ACCURACY_GAIN
        and wins > losses
        and paired_z >= _AFFINE_DECISION_MIN_PAIRED_Z
        and all(net >= 0 for net in fold_net_wins)
        and sum(net > 0 for net in fold_net_wins) >= strict_fold_requirement
        and decision_score >= baseline_score - _AFFINE_DECISION_MAX_RANK_REGRESSION
        and decision_loss <= baseline_loss + log_loss_tolerance
    )
    decision_rank_selected = bool(
        decision_score - baseline_score >= _AFFINE_RANK_MIN_GAIN
        and len(decision_fold_deltas)
        and np.all(decision_fold_deltas >= _AFFINE_RANK_MIN_FOLD_GAIN)
    )
    return {
        "composition": composition,
        "candidate": candidate,
        "projected": projected,
        "rank_score": projected_score,
        "rank_fold_deltas": projected_fold_deltas,
        "rank_loss": projected_loss,
        "rank_selected": rank_selected,
        "decision_prediction": decision_prediction,
        "decision_accuracy": decision_accuracy,
        "accuracy_gain": accuracy_gain,
        "fold_net_wins": fold_net_wins,
        "fold_accuracy_deltas": fold_accuracy_deltas,
        "wins": wins,
        "losses": losses,
        "paired_z": float(paired_z),
        "decision_score": decision_score,
        "decision_fold_deltas": decision_fold_deltas,
        "decision_loss": decision_loss,
        "decision_selected": decision_selected,
        "decision_rank_selected": decision_rank_selected,
        "baseline_prediction": baseline_prediction,
        "baseline_score": baseline_score,
        "baseline_accuracy": baseline_accuracy,
        "baseline_loss": baseline_loss,
        "log_loss_tolerance": log_loss_tolerance,
    }


def _multiclass_null_auc_upper_bound(class_counts):
    """One-sided 95% null bound for macro pairwise AUC.

    Under random ranking, a pairwise Mann-Whitney AUC has variance
    ``(n_a + n_b + 1) / (12 n_a n_b)``. Averaging those pair variances gives
    a cheap conservative screen for highly imbalanced multiclass OOF scores;
    it is not used to rank or tune candidate models.
    """
    counts = np.asarray(class_counts, dtype=float)
    if counts.ndim != 1 or len(counts) < 3 or np.any(counts < 2):
        return 1.0
    variances = [
        (counts[a] + counts[b] + 1.0) / (12.0 * counts[a] * counts[b])
        for a in range(len(counts))
        for b in range(a + 1, len(counts))
    ]
    standard_error = float(np.sqrt(np.sum(variances)) / len(variances))
    return float(min(1.0, 0.5 + _MULTICLASS_NO_SIGNAL_NULL_Z * standard_error))


class _SmoothKNN:
    """Distance-weighted k-NN class-probability proposer.

    The probability rule remains our transparent inverse-distance vote. An
    exact cKD index accelerates neighbour lookup when SciPy is available; the
    NumPy brute-force path remains a compatible fallback.
    """

    def __init__(self, X, y, classes, fw=None, k=15):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        # feature weights: the BOOSTER defines the smooth member's metric (weight by split-importance), so the
        # k-NN attends to the features that matter and isn't diluted by irrelevant ones (helps wide data).
        self.fw = np.ones(X.shape[1]) if fw is None else np.asarray(fw, float)
        self.Xn = (X - self.mu) / self.sd * self.fw
        self._squared_norm = np.square(self.Xn).sum(1)
        try:
            from scipy.spatial import cKDTree

            self._search_index = cKDTree(self.Xn)
        except ImportError:  # scikit-learn normally provides SciPy; retain a standalone fallback
            self._search_index = None
        self.classes = list(classes)
        self.yidx = np.array([self.classes.index(v) for v in y])
        self.k = int(min(k, len(X) - 1))
        self.C = len(self.classes)

    def _dense_neighbours(self, query):
        bsq = getattr(self, "_squared_norm", None)
        if bsq is None:  # compatibility with models saved before this cache existed
            bsq = np.square(self.Xn).sum(1)
        distance = np.maximum(
            np.square(query).sum(1)[:, None] + bsq[None, :] - 2.0 * (query @ self.Xn.T),
            0.0,
        )
        return np.argpartition(distance, self.k, axis=1)[:, : self.k]

    def _neighbours(self, query):
        index = getattr(self, "_search_index", None)
        if index is None:
            return self._dense_neighbours(query)

        distance, candidates = index.query(query, k=self.k + 1, workers=-1)
        candidates = np.asarray(candidates, dtype=np.int64)
        distance = np.asarray(distance, dtype=float)
        selected = candidates[:, : self.k].copy()

        # cKDTree may choose either source row at an exact kth-distance tie.
        # Use the dense backend only for those rows so indexed and fallback
        # predictions retain identical tie semantics without a full sort.
        boundary_tie = distance[:, self.k - 1] == distance[:, self.k]
        if boundary_tie.any():
            selected[boundary_tie] = self._dense_neighbours(query[boundary_tie])
        return selected

    def proba(self, X):
        B = (np.asarray(X, float) - self.mu) / self.sd * self.fw
        P = np.zeros((len(B), self.C))
        for i in range(0, len(B), 2000):  # chunk to bound the distance matrix
            a = B[i : i + 2000]
            idx = self._neighbours(a)
            # WEIGHT from distances recomputed DIRECTLY on the k selected neighbours (numerically stable for
            # small distances; O(k·D) with k=15 → no large temp), so proba matches the exact metric.
            dsel = np.sqrt(((a[:, None, :] - self.Xn[idx]) ** 2).sum(-1))
            w = 1.0 / (dsel + 1e-9)
            lab = self.yidx[idx]
            for c in range(self.C):
                P[i : i + 2000, c] = (w * (lab == c)).sum(1)
        return P / (P.sum(1, keepdims=True) + 1e-12)


class _CategoricalEvidenceMemory:
    """Local class evidence from exact overlaps of atomic one-hot categories.

    Every edge is a finite conjunction of raw category facts. Its weight is the
    sum of the facts' information content, so common levels cannot dominate a
    rare but discriminating agreement. A bounded adaptive neighbourhood makes
    the read local; class-prior normalization keeps the vote symmetric for
    macro ranking on imbalanced multiclass data. This is an auxiliary ranker:
    ``_preserve_certified_class`` keeps the additive booster's proof-carrying
    decision unchanged.
    """

    def __init__(self, X, y, classes, groups, seed=0):
        self.groups = tuple(tuple(int(j) for j in group) for group in groups)
        self.classes = list(classes)
        self._class_index = {c: i for i, c in enumerate(self.classes)}
        self.yidx = np.array([self._class_index[v] for v in np.asarray(y)])
        self.codes = self._codes(X)
        self.n, self.C = len(self.yidx), len(self.classes)
        self.k = min(self.n - 1, max(2, int(np.ceil(self.n**0.5))))
        self.idf = []
        for group_idx, group in enumerate(self.groups):
            known = self.codes[:, group_idx]
            counts = np.bincount(known[known >= 0], minlength=len(group)).astype(float)
            self.idf.append(np.log((self.n + 1.0) / (counts + 1.0)) + 1.0)
        # Static inverted index over the raw one-hot facts.  A query only
        # touches rows that agree on at least one category instead of building
        # a query-by-training score matrix.  The index preserves the exact
        # weighted-overlap definition below; it is not an approximate ANN.
        self.postings = tuple(
            tuple(
                np.flatnonzero(self.codes[:, group_idx] == level).astype(np.int32, copy=False)
                for level in range(len(group))
            )
            for group_idx, group in enumerate(self.groups)
        )
        self.prior = np.bincount(self.yidx, minlength=self.C).astype(float) / self.n
        self.temp = self._local_temperature(seed)

    def _codes(self, X):
        X = np.asarray(X, float)
        out = np.full((len(X), len(self.groups)), -1, dtype=np.int16)
        for group_idx, group in enumerate(self.groups):
            block = X[:, group]
            present = block.max(1) > 0.5
            out[present, group_idx] = block[present].argmax(1)
        return out

    def _similarity(self, query_codes):
        """Dense reference implementation for diagnostics and parity tests.

        Production reads use the inverted postings below.  Keeping this small
        reference makes the atomic-fact semantics directly auditable without
        coupling the serving path to an n-by-query allocation.
        """
        query_codes = np.asarray(query_codes, np.int16)
        score = np.zeros((len(query_codes), self.n), dtype=float)
        for group_idx in range(len(self.groups)):
            query = query_codes[:, group_idx]
            known = query[:, None] >= 0
            same = query[:, None] == self.codes[None, :, group_idx]
            score += (known & same) * self.idf[group_idx][np.maximum(query, 0), None]
        return score

    def _matching_scores(self, query_code):
        """Nonzero exact overlap scores, sorted by training-row index."""
        ids, weights = [], []
        for group_idx, level in enumerate(np.asarray(query_code, np.int16)):
            if level < 0 or level >= len(self.postings[group_idx]):
                continue
            posting = self.postings[group_idx][int(level)]
            if len(posting):
                ids.append(posting)
                weights.append(np.full(len(posting), self.idf[group_idx][int(level)], dtype=float))
        if not ids:
            return np.empty(0, np.int32), np.empty(0, float)
        ids = np.concatenate(ids)
        weights = np.concatenate(weights)
        order = np.argsort(ids, kind="stable")
        ids, weights = ids[order], weights[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
        return ids[starts], np.add.reduceat(weights, starts)

    @staticmethod
    def _select_top_k(ids, scores, k):
        """Select the score-top-k with ascending-row ties in linear time.

        The output order is irrelevant to the weighted vote, so a full sort is
        unnecessary.  The cutoff score comes from ``partition``; all strictly
        better rows survive and the remaining tied rows are already ordered by
        their training index.  This is exactly the declared ranking rule.
        """
        if k <= 0:
            return ids[:0], scores[:0]
        if k >= len(ids):
            return ids, scores
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        better = scores > cutoff
        need = k - int(better.sum())
        tied = np.flatnonzero(scores == cutoff)[:need]
        take = np.concatenate((np.flatnonzero(better), tied))
        return ids[take], scores[take]

    def _top_k(self, query_code, k=None, exclude=None):
        """Exact top-k category-overlap neighbours with a stable tie rule.

        Scores are ordered by descending overlap and then ascending training
        row.  Zero-overlap rows are filled in the same row order.  Defining
        this tie rule removes the platform-dependent ``argpartition`` choice
        from both OOF gating and the final deployed read.
        """
        k = self.k if k is None else int(k)
        ids, scores = self._matching_scores(query_code)
        if exclude is not None:
            keep = ids != int(exclude)
            ids, scores = ids[keep], scores[keep]
        if k <= 0:
            return np.empty(0, np.int32), np.empty(0, float)
        chosen_ids, chosen_scores = self._select_top_k(ids, scores, k)
        if len(chosen_ids) >= k:
            return chosen_ids, chosen_scores

        # Every indexed row has a strictly positive score.  Fill the remaining
        # zero-score neighbours without allocating a length-n score vector.
        blocked = {int(row) for row in ids}
        if exclude is not None:
            blocked.add(int(exclude))
        filler = []
        for row in range(self.n):
            if row not in blocked:
                filler.append(row)
                if len(chosen_ids) + len(filler) == k:
                    break
        if filler:
            chosen_ids = np.concatenate((chosen_ids, np.asarray(filler, np.int32)))
            chosen_scores = np.concatenate((chosen_scores, np.zeros(len(filler))))
        return chosen_ids, chosen_scores

    def _local_temperature(self, seed):
        # A bounded anchor set estimates the two-sigma local score width from
        # training facts alone. It scales automatically across category counts.
        rng = np.random.default_rng(seed)
        anchors = np.arange(self.n) if self.n <= 512 else rng.choice(self.n, 512, replace=False)
        nearest = np.empty((len(anchors), self.k), float)
        if self.n <= _CATEGORY_MEMORY_DENSE_READ_MAX_N:
            score = self._similarity(self.codes[anchors])
            rows = np.arange(self.n)
            for pos, anchor in enumerate(anchors):
                local = score[pos].copy()
                local[int(anchor)] = -np.inf
                _indices, nearest[pos] = self._select_top_k(rows, local, self.k)
        else:
            for pos, anchor in enumerate(anchors):
                _indices, nearest[pos] = self._top_k(self.codes[anchor], exclude=int(anchor))
        return max(1e-6, 2.0 * float(np.median(nearest.std(1))))

    def _vote(self, indices, local):
        weight = np.exp((local - local.max()) / self.temp)
        vote = np.bincount(self.yidx[indices], weights=weight, minlength=self.C).astype(float)
        vote /= np.maximum(self.prior, 1e-12)
        return vote / np.maximum(vote.sum(), 1e-12)

    def _dense_proba(self, codes):
        """Exact small-table read, chunked to bound the dense score matrix."""
        out = np.empty((len(codes), self.C))
        rows = np.arange(self.n)
        for start in range(0, len(codes), 256):
            stop = min(len(codes), start + 256)
            score = self._similarity(codes[start:stop])
            for pos, local in enumerate(score):
                indices, selected = self._select_top_k(rows, local, self.k)
                out[start + pos] = self._vote(indices, selected)
        return out

    def _indexed_proba(self, codes):
        """Exact large-table read through the static category postings."""
        out = np.empty((len(codes), self.C))
        for row, code in enumerate(codes):
            indices, local = self._top_k(code)
            out[row] = self._vote(indices, local)
        return out

    def proba(self, X):
        codes = self._codes(X)
        if self.n <= _CATEGORY_MEMORY_DENSE_READ_MAX_N:
            return self._dense_proba(codes)
        return self._indexed_proba(codes)

    def index_report(self):
        """Static evidence-index size for fit reports and profiling."""
        return {
            "groups": len(self.groups),
            "postings": int(sum(len(posting) for group in self.postings for posting in group)),
            "training_rows": int(self.n),
            "read_backend": "dense" if self.n <= _CATEGORY_MEMORY_DENSE_READ_MAX_N else "postings",
        }


def _flat_path_prefixes(flat):
    """Leaf-node to certified path-prefix map for one numeric flat tree.

    Position ``d`` is the node reached after exactly ``d + 1`` split facts.
    A node id is an exact conjunction of its ancestors' threshold decisions;
    it is not a learned embedding or an approximate hash.
    """
    feat, _thr, left, right, _val = flat
    paths, stack, max_depth = {}, [(0, ())], 0
    while stack:
        node, prefix = stack.pop()
        if feat[node] < 0:
            paths[int(node)] = prefix
            max_depth = max(max_depth, len(prefix))
            continue
        stack.append((int(right[node]), prefix + (int(right[node]),)))
        stack.append((int(left[node]), prefix + (int(left[node]),)))
    out = np.full((len(feat), max_depth), -1, dtype=np.int64)
    for leaf, prefix in paths.items():
        out[leaf, : len(prefix)] = prefix
    return out


def _path_partition_information(leaf, yidx, n_classes, label_entropy):
    """Observed label information of one certified leaf partition.

    This only ranks fixed tree regions already proposed by the booster. The
    value is calculated inside each OOF fitting fold, so it cannot inspect a
    row whose correction is being scored.
    """
    _unique, inverse = np.unique(leaf, return_inverse=True)
    counts = np.zeros((int(inverse.max()) + 1, n_classes), dtype=float)
    np.add.at(counts, (inverse, yidx), 1.0)
    support = counts.sum(1)
    conditional = counts / np.maximum(support[:, None], 1.0)
    entropy = -(conditional * np.log(np.maximum(conditional, 1e-12))).sum(1)
    return float(label_entropy - (support / support.sum()) @ entropy)


class _ProofPathTreeIndex:
    """One tree's bounded hierarchy of exact certified path regions."""

    def __init__(self, flat, X, min_support, max_support, train_leaf=None):
        self.flat = flat
        self.path_for_leaf = _flat_path_prefixes(flat)
        feat = flat[0]
        self.leaf_choice_depth = np.full(len(feat), -1, dtype=np.int16)
        train_leaf = _flat_leaf_ids(flat, X) if train_leaf is None else np.asarray(train_leaf, np.int64)
        self.leaf_postings, self.leaf_idf = {}, {}
        unique_leaf, leaf_counts = np.unique(train_leaf, return_counts=True)
        for leaf, count in zip(unique_leaf, leaf_counts, strict=False):
            if count <= max_support:
                leaf = int(leaf)
                self.leaf_postings[leaf] = np.flatnonzero(train_leaf == leaf).astype(np.int32, copy=False)
                self.leaf_idf[leaf] = float(np.log((len(X) + 1.0) / (count + 1.0)) + 1.0)
        self.postings, self.idf = [], []
        self.prefix_fact_count = 0
        if self.path_for_leaf.shape[1] == 0:
            self.fact_count = len(self.leaf_postings)
            return
        for depth in range(self.path_for_leaf.shape[1]):
            nodes = self.path_for_leaf[train_leaf, depth]
            nodes = nodes[nodes >= 0]
            table, weights = {}, {}
            if len(nodes):
                unique, counts = np.unique(nodes, return_counts=True)
                for node, count in zip(unique, counts, strict=False):
                    if min_support <= count <= max_support:
                        node = int(node)
                        table[node] = np.flatnonzero(self.path_for_leaf[train_leaf, depth] == node).astype(
                            np.int32, copy=False
                        )
                        weights[node] = float(np.log((len(X) + 1.0) / (count + 1.0)) + 1.0)
            self.postings.append(table)
            self.idf.append(weights)
            self.prefix_fact_count += len(table)
        for leaf in np.flatnonzero(feat < 0):
            for depth in range(len(self.postings) - 1, -1, -1):
                node = int(self.path_for_leaf[leaf, depth])
                if node >= 0 and node in self.postings[depth]:
                    self.leaf_choice_depth[leaf] = depth
                    break
        self.fact_count = len(self.leaf_postings) + self.prefix_fact_count

    def selected_nodes(self, X):
        """Deepest support-bounded certified prefix for every query row."""
        leaf = _flat_leaf_ids(self.flat, X)
        depth = self.leaf_choice_depth[leaf]
        nodes = np.full(len(leaf), -1, dtype=np.int64)
        selected = np.flatnonzero(depth >= 0)
        if len(selected):
            nodes[selected] = self.path_for_leaf[leaf[selected], depth[selected]]
        return leaf, depth, nodes


class _ProofPathMemory:
    """Exact local class evidence from bounded certified tree paths.

    Each selected anchor tree contributes the deepest route prefix with
    ``sqrt(n)``-scale support. Its posting contains exactly the training rows
    that satisfy the same conjunction of split facts. Query candidates are
    ranked by the sum of information weights from their shared regions, then
    vote with class-prior normalization. The OOF gate decides whether this
    transparent cross-row read earns a probability correction at all.
    """

    def __init__(self, predictor, X, y, classes):
        X, y = np.asarray(X, float), np.asarray(y)
        flats = list(predictor._flats())
        if getattr(predictor, "linear_", False) or not flats or any(flat is None for flat in flats):
            raise ValueError("proof-path memory requires numeric flat certified trees")
        if len(X) < 2:
            raise ValueError("proof-path memory requires at least two training rows")
        self.classes = list(classes)
        class_index = {value: idx for idx, value in enumerate(self.classes)}
        self.yidx = np.array([class_index[value] for value in y])
        self.n, self.C = len(y), len(self.classes)
        self.k = min(self.n, max(2, int(np.ceil(self.n**0.5))))
        self.max_support = min(self.n, _PROOF_PATH_MEMORY_SUPPORT_MULTIPLIER * self.k)
        anchor_count = min(_PROOF_PATH_MEMORY_MAX_TREES, len(flats))
        prior = np.bincount(self.yidx, minlength=self.C).astype(float) / self.n
        label_entropy = float(-(prior * np.log(np.maximum(prior, 1e-12))).sum())
        leaf_routes = [_flat_leaf_ids(flat, X) for flat in flats]
        information = np.array(
            [_path_partition_information(leaf, self.yidx, self.C, label_entropy) for leaf in leaf_routes]
        )
        # Preserve coverage across boosting stages while choosing the most
        # informative proven partition in every deterministic stage band.
        bands = np.array_split(np.arange(len(flats)), anchor_count)
        self.anchor_indices = tuple(int(band[np.argmax(information[band])]) for band in bands if len(band))
        self.anchor_information = [float(information[index]) for index in self.anchor_indices]
        self.indexes = [
            _ProofPathTreeIndex(flats[tree_index], X, self.k, self.max_support, leaf_routes[tree_index])
            for tree_index in self.anchor_indices
        ]
        self.fact_count = int(sum(index.fact_count for index in self.indexes))
        if self.fact_count == 0:
            raise ValueError("proof-path memory found no support-bounded regions")
        self.prior = np.bincount(self.yidx, minlength=self.C).astype(float) / self.n

    @staticmethod
    def _select_top_k(ids, scores, k):
        """Score-top-k with an explicit ascending-row tie break."""
        if k >= len(ids):
            return ids, scores
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        better = scores > cutoff
        need = k - int(better.sum())
        tied = np.flatnonzero(scores == cutoff)[:need]
        take = np.concatenate((np.flatnonzero(better), tied))
        return ids[take], scores[take]

    def _vote(self, ids, scores):
        vote = np.bincount(self.yidx[ids], weights=scores, minlength=self.C).astype(float)
        vote /= np.maximum(self.prior, 1e-12)
        total = vote.sum()
        return vote / total if total > 0 else self.prior

    def proba(self, X):
        routes = [index.selected_nodes(X) for index in self.indexes]
        out = np.tile(self.prior, (len(np.asarray(X)), 1))
        for row in range(len(out)):
            ids, weights = [], []
            for index, (leaves, depth, nodes) in zip(self.indexes, routes, strict=False):
                leaf = int(leaves[row])
                leaf_posting = index.leaf_postings.get(leaf)
                if leaf_posting is not None:
                    ids.append(leaf_posting)
                    weights.append(np.full(len(leaf_posting), index.leaf_idf[leaf], dtype=float))
                d, node = int(depth[row]), int(nodes[row])
                if d < 0 or node == leaf:
                    continue
                posting = index.postings[d][node]
                ids.append(posting)
                weights.append(np.full(len(posting), index.idf[d][node], dtype=float))
            if not ids:
                continue
            candidate_ids = np.concatenate(ids)
            candidate_scores = np.concatenate(weights)
            # Training row IDs form a bounded integer domain. Aggregate their
            # path weights directly instead of sorting every query's postings.
            totals = np.bincount(candidate_ids, weights=candidate_scores, minlength=self.n)
            candidate_ids = np.flatnonzero(totals)
            candidate_scores = totals[candidate_ids]
            candidate_ids, candidate_scores = self._select_top_k(candidate_ids, candidate_scores, self.k)
            out[row] = self._vote(candidate_ids, candidate_scores)
        return out

    def index_report(self):
        return {
            "anchors": len(self.indexes),
            "anchor_tree_indices": list(self.anchor_indices),
            "anchor_information": self.anchor_information,
            "path_facts": self.fact_count,
            "leaf_facts": int(sum(len(index.leaf_postings) for index in self.indexes)),
            "prefix_facts": int(sum(index.prefix_fact_count for index in self.indexes)),
            "training_rows": self.n,
            "neighbourhood": self.k,
            "max_region_support": self.max_support,
        }


class _SDMAttention:
    """Sparse Distributed Memory read = transformer attention over stored training patterns — OUR OWN
    associative-memory primitive (pure numpy, no external model/embedding). Keys are the training docs'
    IDF-weighted, L2-normalized binary token vectors; values are one-hot labels; a query's class proba =
    softmax(beta·cosine(query, keys)) @ values. This is the attention≈SDM read (Bricken & Pehlevan): the
    softmax activates the stored addresses nearest the query and pools their contents. CLASSIFICATION pools
    one-hot labels (`proba`); REGRESSION pools the continuous targets — Nadaraya-Watson attention (`read`).
    Captures 'this text resembles these training texts' — a non-parametric similarity signal that complements
    the token-threshold booster on graded text. Operates ONLY on the bag-of-words token columns (`cols`) so it
    is not confused by mixed-scale tabular features; the cosine + IDF weighting is what beats a raw dot-product."""

    def __init__(self, X, y, classes, cols, beta=10.0, cap=4000, seed=0, regression=False):
        self.cols = np.asarray(cols, int)
        self.beta = float(beta)
        self.regression = regression
        X, y = np.asarray(X, float), np.asarray(y)
        if len(X) > cap:  # SDM stores a sample of hard locations — bound the memory / read cost
            idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
            X, y = X[idx], y[idx]
        M = X[:, self.cols]
        self.idf = np.log((len(M) + 1) / (M.sum(0) + 1)) + 1.0  # down-weight ubiquitous tokens
        K = M * self.idf
        self.K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
        if regression:
            self.vals = y.astype(float)  # stored continuous targets
        else:
            self.classes = list(classes)
            ci = {c: i for i, c in enumerate(self.classes)}
            self.V = np.eye(len(self.classes))[[ci[v] for v in y]]

    def _weights(self, X):  # softmax(beta·cosine) attention weights, query rows × memory
        Q = np.asarray(X, float)[:, self.cols] * self.idf
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        return Q

    def proba(self, X):  # classification: pool one-hot labels
        Q = self._weights(X)
        out = np.zeros((len(Q), len(self.V[0])))
        for i in range(0, len(Q), 512):  # chunk the query×memory attention matrix
            S = Q[i : i + 512] @ self.K.T
            S -= S.max(1, keepdims=True)
            W = np.exp(self.beta * S)
            W /= W.sum(1, keepdims=True)
            out[i : i + 512] = W @ self.V
        return out

    def read(self, X):  # regression: Nadaraya-Watson attention over stored target values
        Q = self._weights(X)
        out = np.zeros(len(Q))
        for i in range(0, len(Q), 512):
            S = Q[i : i + 512] @ self.K.T
            S -= S.max(1, keepdims=True)
            W = np.exp(self.beta * S)
            W /= W.sum(1, keepdims=True)
            out[i : i + 512] = W @ self.vals
        return out

    def refit(self, X, y):  # a fresh instance on new data (for the OOF calibration folds)
        return _SDMAttention(X, y, None, self.cols, beta=self.beta, regression=True, seed=self.seed)


class _LinearReg:
    """Ridge regression over the bag-of-words token columns — OUR OWN numpy primitive (closed-form, no external
    lib). On sparse text REGRESSION a linear model weights the many RARE informative tokens (a product's
    model/year/trim → price) that the tree booster cannot split on — so it closes a gap the booster structurally
    can't. Blended in as an accuracy member (booster carries the interpretable reason; the conformal interval is
    recalibrated on the blend). `read` mirrors `_SDMAttention.read` so both are interchangeable regression
    members."""

    def __init__(self, X, y, cols, lam=1.0, seed=0):
        self.cols = np.asarray(cols, int)
        self.lam = float(lam)
        self.seed = seed
        M = np.asarray(X, float)[:, self.cols]
        yv = np.asarray(y, float)
        self.mu, self.ybar = M.mean(0), float(yv.mean())
        Mc = M - self.mu
        A = Mc.T @ Mc + self.lam * np.eye(Mc.shape[1])  # ridge normal equations (features are few-hundred–1k)
        self.W = np.linalg.solve(A, Mc.T @ (yv - self.ybar))

    def read(self, X):
        return self.ybar + (np.asarray(X, float)[:, self.cols] - self.mu) @ self.W

    def refit(self, X, y):
        return _LinearReg(X, y, self.cols, self.lam, self.seed)


def _booster_importance(pred, D):
    """Per-feature split frequency across the classifier's boosted trees — a cheap importance the smooth k-NN
    uses to weight its metric. `pred.trees_` = list of (class, tuple-tree). Returns sqrt-normalized weights."""
    imp = np.zeros(D)

    def walk(t):
        if t[0] == "node":
            imp[t[1]] += 1.0
            walk(t[3])
            walk(t[4])

    for item in getattr(pred, "trees_", []):
        t = item[1] if isinstance(item, tuple) and len(item) == 2 and not isinstance(item[0], str) else item
        walk(t)
    return np.sqrt(imp / (imp.max() + 1e-9)) + 0.05  # floor 0.05 so unused features aren't fully zeroed


def _cv_splits(n, folds, seed):
    """(train, val) index splits for config ranking. folds>=2 -> shuffled K-fold; folds==1 -> a single 75/25
    holdout (the first fold of a 4-way split) — the cheap rungs of successive halving only need one split."""
    from sklearn.model_selection import KFold

    if folds >= 2:
        return list(KFold(folds, shuffle=True, random_state=seed).split(np.arange(n)))
    return [next(iter(KFold(4, shuffle=True, random_state=seed).split(np.arange(n))))]


def _fit_temperature(scores, y, classes, sample_weight=None):
    """Temperature that minimizes multiclass log-loss of softmax(scores/T) on OUT-OF-FOLD scores — de-
    overconfidences the boosting logits so `predict_proba` is calibrated. Dividing all logits by T > 0 leaves
    the argmax (and `predict`) unchanged; only the probability magnitudes move. 1-D grid search."""
    scores = np.asarray(scores, float)
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx.get(v, 0) for v in np.asarray(y)])
    rows = np.arange(len(yi))
    best_t, best_ll = 1.0, np.inf
    # search T ≥ 1 only: boosted logits are OVER-confident, so calibration should soften, never sharpen
    # (sharpening, T<1, would risk a log-loss regression from OOF noise). T=1 is the safe no-op default.
    for t in np.geomspace(1.0, 8.0, 40):
        z = scores / t
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        loss = -logp[rows, yi]
        ll = loss.mean() if sample_weight is None else np.average(loss, weights=sample_weight)
        if ll < best_ll:
            best_ll, best_t = ll, float(t)
    return best_t


def _wilson_lb(k, n, z=1.64):
    """One-sided Wilson lower bound on a binomial proportion k/n (z≈1.64 => ~95%). Support-adjusts a rule's
    in-sample precision so a high-precision-but-tiny-support rule gets an honestly conservative confidence."""
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0.0, (center - half) / denom)


class TabPVN:
    """Proposer proposes, FOLKernel verifies, answers carry proofs + a guarantee or abstain — over tabular
    classification, tabular regression, and relational rule induction, under one object."""

    _threshold_predicates = True
    _affine_rank_evidence = True
    _multiclass_stratified_verifier = True
    _parameter_names = ("seed", "alpha", "boost", "tau", "additive", "oblique", "task")

    def __init__(
        self,
        seed=0,
        alpha=0.1,
        boost=None,
        tau=None,
        additive=True,
        oblique=True,
        task="auto",
    ):
        # NO HYPERPARAMETERS — the surface is just fit/predict. With no `boost`, TabPVN fully self-configures:
        # it SELECTS the booster config (depth/lr/leaf by internal-holdout RMSE), DISCOVERS+enforces monotone
        # invariants, fits piecewise-LINEAR leaves, and blends a small-n SMOOTH k-NN proposer — each a
        # proposer/verifier gate applied ONLY if it beats an internal out-of-fold check (zero-downside), so
        # none needs a user knob. It also BUILDS the certified-confidence layer (conformal error bound /
        # region-precision bound; alpha = the guarantee miss rate, 0.1 => ~90% coverage) so confidence(X) /
        # proof(X,row) / certificate(X,row) work with no extra calls. Passing an explicit `boost` dict is the
        # only override — it PINS that config and skips the self-configuration gates (for tests / power users).
        # tau/oblique configure relational induction. ``additive`` remains in the pre-1.0 constructor for
        # compatibility but must be true; non-additive prototypes are not production paths.
        if task not in {"auto", "classification", "regression"}:
            raise ValueError("task must be 'auto', 'classification', or 'regression'")
        if not additive:
            raise ValueError(
                "additive=False is an archived research path and is not supported by the production runtime"
            )
        self.seed, self.alpha, self.boost = seed, alpha, boost
        self.tau, self.additive, self.oblique = tau, additive, oblique
        self.task = task
        self.monotone_ = {}
        self.boost_ = {}
        self.pruned_ = []
        self._interactions = None  # selected deterministic predicate map (classification only)
        self._cat_groups = ()  # raw one-hot blocks eligible for native category-in-set tree nodes
        # Private architecture switches. Native category trees remain research
        # paths; category evidence is a deployed OOF-gated member rather than a
        # user-facing tuning knob.
        self._native_categorical = False
        self._honest_categorical = False
        self._categorical_evidence = True
        self._categorical_posterior_evidence = True
        self._numeric_interval_evidence = True
        self._proof_path_evidence = True
        self.interaction_features_ = []
        self.candidate_report_ = []
        self.booster_selection_report_ = []
        self.category_memory_report_ = []
        self.category_posterior_report_ = []
        self.numeric_interval_report_ = []
        self.affine_rank_report_ = []
        self._category_posterior_permission = None
        self._category_posterior_aggregation = None
        self._category_posterior_smoothing = None
        self._numeric_interval_permission = None
        self._affine_rank = None
        self._affine_rank_weight = 0.0
        self._affine_rank_permission = None
        self._affine_composition = "arithmetic"
        self._affine_decision_oof_labels = None
        self.proof_path_memory_report_ = []
        self.smooth_memory_report_ = []
        self.multiclass_signal_report_ = []
        self._no_signal_prior = None
        self.prior_rank_report_ = []
        self._prior_rank_strength = 0.0
        self.class_weight_report_ = []
        self.rare_architecture_report_ = []
        self.regression_loss_report_ = []
        self.feature_screen_report_ = []
        self.rank_checkpoint_report_ = []
        self.compression_evidence_report_ = []
        self.target_encoding_report_ = []
        self.target_encoding_variant_ = "none"
        self.temporal_evidence_report_ = []
        self.event_discovery_report_ = []
        self.temporal_selected_ = False
        self.event_schema_ = None
        self._event_mode = False
        self._temporal_map = None
        self._pending_validation = None
        self._fit_validation = None
        self.validation_report_ = {"mode": "exchangeable"}
        self.proposer_registry_ = default_registry().names()
        self.fit_pipeline_ = None
        self.fit_stages_ = []
        self._policy = None  # internal decision policy: when set, predict() auto-applies certified_decision
        self._last_decision = None  # the bundle from the most recent policy-driven predict()

    def get_params(self, deep=True):
        """Return constructor parameters using the scikit-learn estimator contract."""
        del deep  # No nested estimators; retained for sklearn's required signature.
        return {name: getattr(self, name) for name in self._parameter_names}

    def set_params(self, **params):
        """Set constructor parameters using the scikit-learn estimator contract."""
        valid = self.get_params(deep=False)
        unknown = sorted(set(params) - set(valid))
        if unknown:
            raise ValueError(f"invalid parameter(s) for TabPVN: {unknown}")
        if params.get("additive") is False:
            raise ValueError(
                "additive=False is an archived research path and is not supported by the production runtime"
            )
        task = params.get("task", self.task)
        if task not in {"auto", "classification", "regression"}:
            raise ValueError("task must be 'auto', 'classification', or 'regression'")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "mode") and (
            hasattr(self, "_pred") or (self.mode == "relational" and hasattr(self, "_rules"))
        )

    def _require_fitted(self, modes=None, additive=False):
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("model is not fitted; call fit(X, y) first")
        if modes is not None and self.mode not in modes:
            expected = ", ".join(sorted(modes))
            raise ValueError(f"operation requires mode {expected}; fitted mode is {self.mode}")
        if additive and not self.additive:
            raise ValueError("operation requires the additive certified predictor")

    def save(self, path):
        """Atomically persist this fitted model to ``path``."""
        from tabpvn.model_io import save_model

        return save_model(self, path)

    @classmethod
    def load(cls, path):
        """Load a trusted model file written by :meth:`save`."""
        from tabpvn.model_io import load_model

        model = load_model(path)
        if not isinstance(model, cls):
            raise ValueError(f"model file contains {type(model).__name__}, not {cls.__name__}")
        return model

    def score(self, X, y):
        """Return accuracy for classification or R-squared for regression."""
        self._require_fitted(modes={"classification", "regression"})
        truth = np.asarray(y)
        prediction = np.asarray(self.predict(X))
        if truth.ndim != 1 or len(truth) != len(prediction):
            raise ValueError("y must be one-dimensional and aligned with X")
        if self.mode == "classification":
            return float(np.mean(prediction == truth))
        truth = truth.astype(float)
        prediction = prediction.astype(float)
        residual = float(np.square(truth - prediction).sum())
        total = float(np.square(truth - truth.mean()).sum())
        return float(residual == 0.0) if total == 0.0 else 1.0 - residual / total

    def _classifier(self, **kwargs):
        """Build the same certified classifier used by the final deployment path.

        Internal tuning, calibration and challenger checks must see native
        categorical partitions too; otherwise they would choose a configuration
        for a different learner than the one ultimately served.
        """
        return AdditiveCertifiedClassifier(
            seed=self.seed,
            categorical_groups=getattr(self, "_cat_groups", ()),
            honest_categorical=getattr(self, "_honest_categorical", False),
            **kwargs,
        )

    def _validation_groups(self, rows=None):
        """Return timestamp groups aligned to full or selected fitting rows."""
        validation = getattr(self, "_fit_validation", None)
        if validation is None:
            return None
        return validation.groups if rows is None else validation.groups[np.asarray(rows, dtype=np.int64)]

    def _bounded_evidence_rows(
        self,
        y,
        cap,
        seed,
        *,
        stratified,
        min_class_rows=0,
    ):
        """Bound gate evidence while preserving temporal order when declared."""
        validation = getattr(self, "_fit_validation", None)
        if validation is not None:
            rows = validation.bounded_rows(int(cap)) if len(y) > int(cap) else np.arange(len(y))
            return (None if len(rows) == len(y) else rows), None
        return _fit_sample(
            y,
            cap,
            seed,
            stratified=stratified,
            min_class_rows=min_class_rows,
        )

    def _validation_splits(
        self,
        y,
        *,
        folds,
        classification,
        groups=None,
        single=False,
        holdout=0.25,
    ):
        """Return shuffled folds normally and one strict future holdout for events."""
        target = np.asarray(y)
        validation = getattr(self, "_fit_validation", None)
        if validation is not None:
            if groups is None:
                if len(target) != len(validation.groups):
                    raise ValueError("temporal validation groups must be supplied for subset evidence")
                local = validation
            elif groups is validation.groups:
                local = validation
            else:
                local = FutureValidation(np.asarray(groups, dtype=np.int64))
            return [
                local.split(
                    target if classification else None,
                    holdout=holdout,
                    min_train=max(2, int(0.25 * len(target))),
                    min_valid=max(1, int(0.05 * len(target))),
                    require_class_coverage=classification,
                )
            ]

        if classification:
            from sklearn.model_selection import StratifiedKFold

            requested = max(2, int(folds))
            splits = list(
                StratifiedKFold(requested, shuffle=True, random_state=self.seed).split(
                    np.arange(len(target)), target
                )
            )
        else:
            from sklearn.model_selection import KFold

            requested = 4 if single else max(2, int(folds))
            splits = list(
                KFold(requested, shuffle=True, random_state=self.seed).split(np.arange(len(target)))
            )
        return splits[:1] if single else splits

    def _fit_certified(self, model, X, y, *, groups=None, sample_weight=None):
        """Fit a certified model under the active validation geometry."""
        kwargs = {}
        validation = getattr(self, "_fit_validation", None)
        if validation is not None:
            if groups is None:
                if len(y) != len(validation.groups):
                    raise ValueError("temporal validation groups must align with candidate rows")
                groups = validation.groups
            kwargs["validation_groups"] = np.asarray(groups, dtype=np.int64)
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        return model.fit(X, y, **kwargs)

    def fit(
        self,
        data,
        y=None,
        target=None,
        *,
        entity=None,
        timestamp=None,
        value_columns=None,
    ):
        """Fit an ordinary table or a causally ordered entity-event table.

        DataFrames are inspected for plausible event semantics without labels.
        A bounded multi-window future gate selects the entity, timestamp, marked-value
        channels, and event path only when causal history materially improves the
        full raw schema. Otherwise fitting remains exchangeable. Explicit
        ``entity`` and ``timestamp`` metadata are retained only as an override for
        unusual schemas and require both columns.
        """
        event_requested = any(value is not None for value in (entity, timestamp, value_columns))
        if not event_requested:
            return self._fit_automatic(data, y=y, target=target)
        if entity is None or timestamp is None:
            raise ValueError("event-aware fit requires both entity= and timestamp= schema columns")
        if target is not None:
            raise ValueError("event-aware fit accepts y, not the relational target= argument")
        if y is None:
            raise ValueError("event-aware fit requires y (labels/targets); got None")
        return self._fit_event_table(
            data,
            y,
            entity=entity,
            timestamp=timestamp,
            value_columns=value_columns,
        )

    @staticmethod
    def _automatic_event_margin(report):
        if report.get("metric") == "neg_rmse":
            gain = float(report.get("relative_gain", -np.inf))
            confidence = report.get("confidence_lower_relative_gain")
            if report.get("power_aware"):
                legacy = float(report.get("legacy_minimum_relative_gain", np.inf))
                return max(
                    float(confidence) if confidence is not None else -np.inf,
                    gain - legacy,
                )
            return gain - float(report.get("minimum_relative_gain", np.inf))
        gain = float(report.get("gain", -np.inf))
        confidence = report.get("confidence_lower_gain")
        if report.get("power_aware"):
            legacy = float(report.get("legacy_minimum_gain", np.inf))
            return max(
                float(confidence) if confidence is not None else -np.inf,
                gain - legacy,
            )
        return gain - float(report.get("minimum_gain", np.inf))

    @staticmethod
    def _automatic_event_report(candidate, evidence, *, selected, reason=None):
        report = gate_report(
            "automatic_event_schema",
            selected,
            stage="schema",
            metric=evidence.get("metric"),
            mean_score=evidence.get("mean_score"),
            reason=reason,
        )
        report.update(candidate.asdict())
        report["selection_margin"] = TabPVN._automatic_event_margin(evidence)
        report["evidence"] = dict(evidence)
        return report

    def _fit_automatic(self, data, y=None, target=None):
        """Discover and gate event semantics, otherwise preserve ordinary fit."""
        import pandas as pd

        from tabpvn.event_schema import bounded_event_gate, discover_event_schemas

        candidates = ()
        target_values = None
        if target is None and y is not None and isinstance(data, pd.DataFrame):
            candidate_target = np.asarray(y)
            if candidate_target.ndim == 1 and len(candidate_target) == len(data):
                target_values = candidate_target
                candidates = discover_event_schemas(data)

        reports = []
        selected = None
        selected_evidence = None
        if candidates and target_values is not None:
            mode = (
                ("classification" if _is_classification(target_values) else "regression")
                if self.task == "auto"
                else self.task
            )
            challenger = TemporalEvidenceChallenger(seed=self.seed)
            gate_samples: dict[Any, tuple[Any, np.ndarray, dict[str, Any]]] = {}
            for pair_rank in sorted({candidate.pair_rank for candidate in candidates}):
                pair_candidates = tuple(item for item in candidates if item.pair_rank == pair_rank)
                gate_timestamp = pair_candidates[0].timestamp
                if gate_timestamp not in gate_samples:
                    gate_data, gate_target = bounded_event_gate(
                        data,
                        target_values,
                        timestamp=gate_timestamp,
                    )
                    gate_samples[gate_timestamp] = (gate_data, gate_target, {})
                gate_data, gate_target, baseline_cache = gate_samples[gate_timestamp]
                evaluated = []
                for candidate in pair_candidates:
                    try:
                        evidence = challenger.evaluate(
                            gate_data,
                            gate_target,
                            entity=candidate.entity,
                            timestamp=candidate.timestamp,
                            value_columns=candidate.value_columns,
                            task=mode,
                            drop_entity=False,
                            _baseline_cache=baseline_cache,
                        )
                        evidence["automatic_source_rows"] = int(len(data))
                        evidence["automatic_gate_source_rows"] = int(len(gate_data))
                    except Exception as error:  # an automatic proposer must fail closed
                        evidence = gate_report(
                            "temporal_laplace_evidence",
                            False,
                            stage="schema",
                            reason=f"automatic_schema_error:{type(error).__name__}:{error}",
                        )
                    margin = self._automatic_event_margin(evidence)
                    eligible = bool(evidence.get("selected") and margin >= 0.0)
                    evaluated.append((candidate, evidence, margin, eligible))
                promoted = [item for item in evaluated if item[3]]
                if promoted:
                    selected, selected_evidence, _, _ = max(
                        promoted,
                        key=lambda item: (
                            item[2],
                            item[0].structural_score,
                            -len(item[0].value_columns),
                        ),
                    )
                for candidate, evidence, _, eligible in evaluated:
                    is_selected = candidate == selected
                    if is_selected:
                        reason = None
                    elif eligible:
                        reason = "lower_scoring_schema_candidate"
                    elif evidence.get("selected"):
                        reason = "automatic_schema_search_margin"
                    else:
                        reason = evidence.get("reason", "future_holdout_gate_rejected")
                    reports.append(
                        self._automatic_event_report(
                            candidate,
                            evidence,
                            selected=is_selected,
                            reason=reason,
                        )
                    )
                if selected is not None:
                    break

        if selected is not None:
            return self._fit_event_table(
                data,
                target_values,
                entity=selected.entity,
                timestamp=selected.timestamp,
                value_columns=selected.value_columns,
                evidence_report=selected_evidence,
                selection_source="automatic",
                discovery_reports=reports,
                drop_entity=False,
            )

        fitted = self._fit_pipeline(data, y=y, target=target)
        if isinstance(data, pd.DataFrame) and target is None:
            if not reports:
                reports = [
                    gate_report(
                        "automatic_event_schema",
                        False,
                        stage="schema",
                        reason="no_structurally_eligible_event_schema",
                        source_rows=int(len(data)),
                    )
                ]
            self.event_discovery_report_ = reports
            self.candidate_report_.extend(reports)
        return fitted

    def _fit_pipeline(self, data, y=None, target=None):
        """Fit the shared certified predictor without re-entering public routing."""
        from tabpvn.pipeline import FitPipeline

        return FitPipeline().fit(self, data, y=y, target=target)

    def _fit_event_table(
        self,
        data,
        y,
        *,
        entity,
        timestamp,
        value_columns=None,
        evidence_report=None,
        selection_source="explicit",
        discovery_reports=None,
        drop_entity=True,
    ):
        """Fit an event table with OOF-free, causally gated temporal evidence.

        ``entity`` and ``timestamp`` declare schema semantics rather than model
        hyperparameters. Optional ``value_columns`` identify at most two numeric
        event marks such as amount or failed-attempt count. The representation's
        scales, robust mark normalization, and deployment decision are automatic.
        Prediction accepts the original event schema and requires future rows for
        entities present during fitting.
        """
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("event-aware fit requires a pandas DataFrame")
        if data.columns.duplicated().any():
            duplicates = list(data.columns[data.columns.duplicated()])
            raise ValueError(f"training event DataFrame has duplicate columns: {duplicates[:5]}")
        missing_semantics = [column for column in (entity, timestamp) if column not in data.columns]
        if missing_semantics:
            raise ValueError(f"event DataFrame is missing semantic columns: {missing_semantics}")
        target_values = np.asarray(y)
        if target_values.ndim != 1:
            raise ValueError(f"target y must be one-dimensional; got shape {target_values.shape}")
        if len(target_values) != len(data):
            raise ValueError(f"event table has {len(data)} rows but y has {len(target_values)} values")
        if len(target_values) < 2:
            raise ValueError("event table and y must contain at least two rows")
        if pd.isna(target_values).any():
            raise ValueError("target y contains missing values")
        if target_values.dtype.kind in "fc" and not np.isfinite(target_values).all():
            raise ValueError("target y contains NaN or infinite values")
        if pd.Series(target_values).nunique(dropna=True) < 2:
            raise ValueError("target y has fewer than 2 distinct non-missing values")
        mode = (
            ("classification" if _is_classification(target_values) else "regression")
            if self.task == "auto"
            else self.task
        )
        if isinstance(value_columns, (str, bytes)):
            raise TypeError("value_columns must be a sequence of DataFrame column labels, not a string")
        values = () if value_columns is None else tuple(value_columns)
        validation, order = FutureValidation.from_timestamps(data[timestamp]).sorted()
        report = evidence_report
        if report is None:
            report = TemporalEvidenceChallenger(seed=self.seed).evaluate(
                data,
                target_values,
                entity=entity,
                timestamp=timestamp,
                value_columns=values,
                task=mode,
            )
        selected = bool(report["selected"])
        temporal_map = None
        ordered_data = data.iloc[order].reset_index(drop=True)
        ordered_target = target_values[order]
        model_data = ordered_data.copy(deep=False)
        if selected:
            from tabpvn.temporal import TemporalLaplaceMap

            temporal_map = TemporalLaplaceMap(
                entity,
                timestamp,
                value_columns=values,
            )
            model_data = temporal_map.fit_augment(ordered_data)
            report.update(
                {
                    "deployed_features": int(len(temporal_map.feature_names_)),
                    "deployed_scales_seconds": [float(scale) for scale in temporal_map.scales_seconds_],
                    "deployed_channels": [str(channel) for channel in temporal_map.channel_names_],
                }
            )
        else:
            report["deployed_features"] = 0
        if drop_entity:
            model_data = model_data.drop(columns=[entity])

        # Ordinary fit owns the final predictor lifecycle. Event replay state is
        # attached only after that fit succeeds, so failures cannot leave a
        # partially fitted temporal model behind.
        self._pending_validation = validation
        try:
            self._fit_pipeline(model_data, ordered_target)
        finally:
            self._pending_validation = None
            self._fit_validation = None  # fit-only timestamps must not inflate the serving model
        self._event_mode = True
        self._temporal_map = temporal_map
        self.temporal_selected_ = selected
        self.event_discovery_report_ = list(discovery_reports or [])
        report["downstream_validation"] = dict(self.validation_report_)
        self.temporal_evidence_report_ = [report]
        self.event_schema_ = {
            "entity": entity,
            "timestamp": timestamp,
            "value_columns": values,
            "input_columns": tuple(data.columns),
            "training_order": "timestamp_ascending",
            "selection": selection_source,
            "drop_entity": bool(drop_entity),
        }
        if self.event_discovery_report_:
            self.candidate_report_.extend(self.event_discovery_report_)
        else:
            self.candidate_report_.append(report)
        self.n_features_in_ = int(data.shape[1])
        if all(isinstance(column, str) for column in data.columns):
            self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self

    # ---- shared predictor: detect mode, fit, and build the confidence layer ----
    def _fit_predictor(self, data, y=None, target=None):  # noqa: C901 - staged orchestration
        # A refit is a new model. Decision economics, if configured, belong to the prior fitted model.
        pending_validation = getattr(self, "_pending_validation", None)
        self._pending_validation = None
        self._fit_validation = pending_validation
        self.validation_report_ = (
            {"mode": "exchangeable"}
            if pending_validation is None
            else {
                **pending_validation.report(),
                "candidate_geometry": "single_future_holdout",
                "booster_early_stopping": "future_holdout",
                "confidence_calibration": "future_holdout",
            }
        )
        self._event_mode = False
        self._temporal_map = None
        self.temporal_selected_ = False
        self.temporal_evidence_report_ = []
        self.event_discovery_report_ = []
        self.event_schema_ = None
        self._policy = None
        self._last_decision = None
        self._interactions = None
        self._cat_groups = ()
        self.interaction_features_ = []
        self.candidate_report_ = [gate_report("certified_boost", True, stage="predictor")]
        self.booster_selection_report_ = [gate_report("auto_boost", True, stage="predictor")]
        self.rare_architecture_report_ = []
        self.regression_loss_report_ = []
        self.feature_screen_report_ = []
        self.rank_checkpoint_report_ = []
        self.compression_evidence_report_ = []
        self.target_encoding_report_ = []
        self.target_encoding_variant_ = "none"
        self.category_memory_report_ = []
        self.category_posterior_report_ = []
        self.numeric_interval_report_ = []
        self.affine_rank_report_ = []
        self._category_posterior_permission = None
        self._category_posterior_aggregation = None
        self._category_posterior_smoothing = None
        self.proof_path_memory_report_ = []
        self.multiclass_signal_report_ = []
        self._no_signal_prior = None
        self.prior_rank_report_ = []
        self._prior_rank_strength = 0.0
        self._prior_rank_oof_proba = None
        self._numeric_interval_permission = None
        self._numeric_interval_oof_proba = None
        self._affine_rank = None
        self._affine_rank_weight = 0.0
        self._affine_rank_permission = None
        self._affine_composition = "arithmetic"
        self._affine_rank_oof_proba = None
        self._affine_decision_oof_labels = None
        if _is_relational(data):
            self.mode = "relational"
            self._fit_relational(list(data), target)
            return self
        self._prep = None
        self.feature_names_ = None
        import pandas as pd

        if y is None:
            raise ValueError("tabular fit(X, y) requires y (labels/targets); got None.")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(
                f"target y must be one-dimensional; got shape {y.shape}. "
                "Use TabPVNMultiOutput for multiple targets."
            )
        if pd.isna(y).any():
            raise ValueError("target y contains missing values")
        if y.dtype.kind in "fc" and not np.isfinite(y).all():
            raise ValueError("target y contains NaN or infinite values")
        is_frame = isinstance(data, pd.DataFrame)
        if is_frame:
            if data.columns.duplicated().any():
                duplicates = list(data.columns[data.columns.duplicated()])
                raise ValueError(f"training DataFrame has duplicate columns: {duplicates[:5]}")
            n_rows, n_features = data.shape
            raw_data = None
        else:
            raw_data = np.asarray(data)
            if raw_data.ndim != 2:
                raise ValueError(f"X must be a 2-D table; got shape {raw_data.shape}")
            n_rows, n_features = raw_data.shape
        if n_rows != len(y):
            raise ValueError(f"X has {n_rows} rows but y has {len(y)} values")
        if n_rows < 2:
            raise ValueError("X and y must contain at least two rows")
        if n_features < 1:
            raise ValueError("X must contain at least one feature column")
        self.n_features_in_ = int(n_features)
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if is_frame and all(isinstance(column, str) for column in data.columns):
            self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        if (
            pd.Series(y.ravel()).dropna().nunique() < 2
        ):  # all-NaN / single-value target -> clear error, not a crash
            raise ValueError("target y has fewer than 2 distinct non-missing values — nothing to learn.")
        self.mode = (
            ("classification" if _is_classification(y) else "regression")
            if self.task == "auto"
            else self.task
        )
        self.rare_event_ = False
        self.rare_class_ = None
        self._rare_candidate = False
        self.rare_event_report_ = None
        self.multiclass_architecture_report_ = None
        if self.mode == "classification":
            target_classes, target_counts = np.unique(y, return_counts=True)
            if len(target_classes) == 2:
                rare_index = int(np.argmin(target_counts))
                rare_rate = float(target_counts[rare_index] / target_counts.sum())
                self._rare_candidate = rare_rate <= _RARE_CANDIDATE_MAX_RATE
                if self._rare_candidate:
                    self.rare_class_ = target_classes[rare_index]
                    self.rare_event_report_ = {
                        "active": False,
                        "candidate": True,
                        "class": self.rare_class_,
                        "source_events": int(target_counts[rare_index]),
                        "source_rate": rare_rate,
                    }
                    # Explicit booster configurations retain the historical behavior. The evidence gate is a
                    # zero-knob architecture selector and must not override a caller-specified fit contract.
                    if self.boost:
                        self.rare_event_ = rare_rate <= _RARE_EVENT_MAX_RATE
                        self.rare_event_report_["active"] = self.rare_event_
                        self.rare_event_report_["selection"] = "explicit_config_legacy_boundary"
        if is_frame or raw_data.dtype == object:  # raw business data -> encode
            self._prep = _Preprocessor(
                target_encoding=getattr(self, "_target_encoding", True),
                task=self.mode,
                compression_evidence=self._fit_validation is None,
            )
            X = self._prep.fit_transform(
                data,
                y,
                validation_groups=self._validation_groups(),
            )  # leak-safe target and sequence evidence
            self.feature_names_ = self._prep.names
            self.compression_evidence_report_ = list(self._prep.compression_report)
        else:
            X = np.asarray(raw_data, float)
            if not np.isfinite(X).all():
                raise ValueError(
                    "numeric-array X contains NaN or infinite values; impute first or pass a DataFrame"
                )
        boost = dict(self.boost or {})
        auto_cfg = self.additive and not boost  # no explicit `boost` -> self-configure everything
        self.target_encoding_selected_ = False
        if not auto_cfg and self._prep is not None and any(self._prep.target_enabled.values()):
            self.target_encoding_variant_ = "mean"
        if auto_cfg and self._prep is not None and any(self._prep.target_enabled.values()):
            self.target_encoding_selected_ = self._auto_target_encoding(data, y)
            if self.target_encoding_selected_ and self.target_encoding_variant_ == "gaussian":
                self._prep = _Preprocessor(
                    target_encoding=True,
                    task=self.mode,
                    compression_evidence=self._fit_validation is None,
                    gaussian_target_statistics=True,
                )
                X = self._prep.fit_transform(
                    data,
                    y,
                    validation_groups=self._validation_groups(),
                )
                self.feature_names_ = self._prep.names
                self.compression_evidence_report_ = list(self._prep.compression_report)
            elif not self.target_encoding_selected_:
                # Rebuild, rather than retaining constant target columns: a
                # constant can still consume a sampled tree feature and make the
                # fallback differ from the true legacy frequency-only model.
                self._prep = _Preprocessor(
                    target_encoding=False,
                    task=self.mode,
                    compression_evidence=self._fit_validation is None,
                )
                X = self._prep.fit_transform(
                    data,
                    y,
                    validation_groups=self._validation_groups(),
                )
                self.feature_names_ = self._prep.names
                self.compression_evidence_report_ = list(self._prep.compression_report)
        if (
            self.mode == "classification"
            and self._prep is not None
            and getattr(self, "_native_categorical", False)
        ):
            # Research-only: retain the original one-hot blocks as atomic
            # finite-category facts while leaving their numeric columns in X.
            self._cat_groups = _onehot_groups(self._prep)
        if self.mode == "classification" and auto_cfg and self._rare_candidate:
            self.rare_event_ = self._auto_rare_architecture(X, y)
            self.rare_event_report_["active"] = self.rare_event_
        if auto_cfg:
            boost = self._auto_tune(X, y)  # architecture selects the config; user sets none
        if self.mode == "classification" and auto_cfg and not self.rare_event_:
            shallow_boost = self._auto_shallow_boost(X, y, boost)
            if shallow_boost is not None:
                boost = shallow_boost
        if (
            self.mode == "classification"
            and auto_cfg
            and not self.rare_event_
            and "class_weight" not in boost
        ):
            cw = self._auto_class_weight(X, y, boost)  # balance minority ONLY if it helps (gated)
            if cw is not None:
                boost["class_weight"] = cw
        if (
            self.mode == "classification"
            and auto_cfg
            and not self.rare_event_
            and self._auto_linear_leaf_clf(X, y, boost)
        ):
            boost["linear_leaf"] = True  # logit-linear leaves ONLY where they clearly help (gated)
        if self.mode == "classification" and auto_cfg and not self.rare_event_:
            joint_boost = self._auto_joint_rank_regions(X, y, boost)
            if joint_boost is not None:
                boost = joint_boost
        if (
            self.mode == "classification"
            and auto_cfg
            and not self.rare_event_
            and np.unique(y).size == 2
            and self._auto_rank_checkpoint_clf(X, y, boost)
        ):
            boost["validation_metric"] = "auc"
        self.n_input_features_ = X.shape[1]
        if self.mode == "classification" and auto_cfg:
            n_classes = np.unique(y).size
            if self.rare_event_:
                interactions = self._auto_rare_interactions(X, y, boost)
            elif n_classes > 2:
                interactions = self._auto_multiclass_interactions(X, y, boost)
                # The multiclass architecture gate already compares prefixes
                # on class-stratified verifier rows. Keep that same class-cover
                # contract for compact numeric schemas, where an unstratified
                # final split can stop much earlier despite an unchanged
                # configuration. Category-rich sentinels do not transfer this
                # gain, so they retain the incumbent verifier geometry.
                prep = getattr(self, "_prep", None)
                numeric_schema = prep is None or not (
                    getattr(prep, "cat_cols", ()) or getattr(prep, "text_cols", ())
                )
                stratified_verifier = bool(
                    self._multiclass_stratified_verifier
                    and numeric_schema
                    and _MULTICLASS_RULE_GATE_MIN_ROWS <= len(y) <= _MULTICLASS_RULE_GATE_MAX_ROWS
                    and X.shape[1] <= 32
                )
                if stratified_verifier:
                    boost["stratified_holdout"] = True
                if isinstance(self.multiclass_architecture_report_, dict):
                    self.multiclass_architecture_report_["stratified_verifier"] = stratified_verifier
            else:
                interactions = self._auto_interactions(X, y, boost)
            if interactions is not None:
                self._interactions = interactions
                source_names = self.feature_names_
                if source_names is None:
                    source_names = [f"feature[{j}]" for j in range(X.shape[1])]
                self.interaction_features_ = interactions.names(source_names)
                self.feature_names_ = list(source_names) + self.interaction_features_
                base_width = X.shape[1]
                X = interactions.transform(X)
                if "allowed" in boost and "base_feature_count" not in boost:
                    boost["allowed"] = tuple(boost["allowed"]) + tuple(range(base_width, X.shape[1]))
        if self.mode == "regression" and auto_cfg:
            self.monotone_ = self._auto_monotone(X, y, boost)  # discover + verify monotone invariants
            if self.monotone_:
                boost["mono"] = self.monotone_
            if self._auto_linear_leaf(X, y, boost):  # piecewise-linear leaves ONLY where they clearly help
                boost["linear_leaf"] = True
        if auto_cfg and "refit" not in boost and len(y) >= _REFIT_MAX_N:
            boost["refit"] = (
                False  # large n: skip the full-data refit (≈2x deploy-fit cost, ~6e-4 acc change)
            )
        if auto_cfg:
            n_classes = np.unique(y).size if self.mode == "classification" else None
            boost = _large_fit_budget(boost, len(y), self.mode, n_classes=n_classes)
            if self.rare_event_:
                boost.update(
                    rare_event=True,
                    rare_min_events=_RARE_RESERVOIR_MIN_EVENTS,
                    min_verifier_events=_RARE_VERIFY_MIN_EVENTS,
                )
        if self._fit_validation is not None and "holdout" not in boost:
            effective_rows = min(len(y), int(boost.get("fit_cap", len(y))))
            verifier_rows = int(np.clip(0.10 * effective_rows, 200, _BOOST_VERIFY_MAX_ROWS))
            boost["holdout"] = min(0.25, verifier_rows / effective_rows)
            self.validation_report_["booster_holdout_rows"] = int(round(boost["holdout"] * effective_rows))
            self.validation_report_["booster_holdout_fraction"] = float(boost["holdout"])
        self.boost_ = self._cfg = boost  # the resolved, self-selected configuration
        self._smooth = None  # smooth proposer (set below for small-n classification)
        self._smooth_k = _SMOOTH_FIXED_NEIGHBORS
        self._smooth_geometry = "fixed"
        self._smooth_oof_proba = None  # transient projected OOF blend for fair-price calibration
        self._no_signal_prior = None  # dominant multiclass prior fallback admitted by shared OOF evidence
        self.multiclass_signal_report_ = [
            {
                "name": "multiclass_prior_fallback",
                "selected": False,
                "reason": "shared_oof_unavailable",
            }
        ]
        self._prior_rank_oof_proba = None  # transient selected OOF rank surface for downstream gates
        self._proof_path_memory = None  # certified path-evidence member (OOF-gated)
        self._proof_path_oof_proba = None  # transient projected OOF blend for downstream gates
        self._validation_evidence_rows = None  # strict future rows for temporal challenger calibration
        self._category_memory = None  # explicit category-evidence member (OOF-gated)
        self._category_memory_oof_proba = None  # transient projected OOF blend for the path challenger
        self._category_posterior = None  # OOF-gated finite category posterior
        self._category_posterior_w = 0.0
        self._category_posterior_permission = None  # class_change or rank_only
        self._category_posterior_aggregation = None  # strongest or disjoint_pool
        self._category_posterior_smoothing = None  # global or hierarchical
        self._category_posterior_oof_proba = None  # transient deployed OOF posterior update
        self._numeric_interval = None  # OOF-gated accuracy-only interval decision head
        self._numeric_interval_w = 0.0
        self._numeric_interval_permission = None
        self._numeric_interval_aggregation = None
        self._numeric_interval_smoothing = None
        self._numeric_interval_oof_labels = None  # transient deployed OOF decisions
        self._numeric_interval_oof_proba = None  # transient independently admitted OOF rank surface
        self._affine_rank = None  # explicit global affine probability evidence (OOF-gated)
        self._affine_rank_weight = 0.0
        self._affine_rank_permission = None  # rank_only, decision_only, or decision_and_rank
        self._affine_composition = "arithmetic"
        self._affine_rank_oof_proba = None  # transient selected OOF rank surface
        self._affine_decision_oof_labels = None  # transient selected OOF top-1 decisions
        self._sdm = None  # SDM-attention associative-memory member (set below for text classification)
        if self.additive:  # full-coverage proof-carrying ensemble
            predictor_config = boost
            if self.mode == "classification" and boost.get("adaptive_best_first_pair", False):
                predictor_config = dict(boost, track_residual_dynamics=True)
            predictor = (
                self._classifier(**predictor_config)
                if self.mode == "classification"
                else AdditiveCertifiedRegressor(seed=self.seed, **boost)
            )
            self._pred = self._fit_certified(predictor, X, y)
            if self._fit_validation is not None:
                self.validation_report_["fit_sampling"] = dict(self._pred.fit_sampling_)
            if self.rare_event_ and self.rare_event_report_ is not None:
                sample_classes = np.asarray(self._pred.sample_classes_)
                rare_sample_index = int(np.flatnonzero(sample_classes == self.rare_class_)[0])
                fit_weight = getattr(self._pred, "fit_sample_weight_", None)
                fitted_y = y if self._pred.fit_rows_ is None else y[self._pred.fit_rows_]
                weights = (
                    np.ones(len(fitted_y), dtype=float)
                    if fit_weight is None
                    else np.asarray(fit_weight, dtype=float)
                )
                self.rare_event_report_.update(
                    {
                        "reservoir_rows": int(len(fitted_y)),
                        "reservoir_events": int(self._pred.sample_class_counts_[rare_sample_index]),
                        "weighted_reservoir_rate": float(
                            np.average(fitted_y == self.rare_class_, weights=weights)
                        ),
                        "reservoir_effective_rows": float(
                            weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12)
                        ),
                    }
                )
            # REGRESSION SDM-attention member: built BEFORE the confidence layer so the conformal bound is
            # calibrated on the BLENDED predictor (essential — the guaranteed interval must cover what
            # predict() returns). Nadaraya-Watson attention over the token vectors; blended if it CLEARLY helps.
            if auto_cfg and self.mode == "regression" and self._has_text():
                kind, w = self._sdm_gate_reg(X, y)
                if kind is not None:
                    cols = [j for j, nm in enumerate(self.feature_names_) if "~" in str(nm)]
                    self._sdm = self._reg_member(X, y, cols)[kind](kind)  # SDM attention or ridge linear read
                    self._sdm_w = w
            # SHARED OOF: when the smooth gate will run (small-n classification, no LEVER-A holdout), the
            # confidence layer and the smooth gate would each fit their own deployed-config K-fold OOF. Build
            # it ONCE and let both reuse it — saves ~2 booster fits per fit, guarantee preserved (verified).
            _oof = None
            _ver = getattr(self._pred, "ver_", None)
            if (
                auto_cfg
                and self.mode == "classification"
                and not self.rare_event_
                and _smooth_weight(len(y)) > 0
                and (_ver is None or len(_ver) < 200)
                and len(y) <= _CONF_MAX_N
            ):
                _oof = self._clf_oof(X, y)
            self._build_confidence(
                X, y, precomp=_oof
            )  # certified bounds/precision (reg: on booster+SDM blend)
            if (
                auto_cfg
                and self.mode == "classification"
                and len(self._pred.classes_) > 2
                and _oof is not None
            ):
                no_signal_probability = self._multiclass_no_signal_gate(y, _oof)
                if no_signal_probability is not None:
                    self._refresh_classification_confidence(X, y, no_signal_probability)
            # smooth proposer: classification, small n only (weight>0), else skip entirely
            if (
                auto_cfg
                and self.mode == "classification"
                and not self.rare_event_
                and _smooth_weight(len(y)) > 0
            ):
                fw = _booster_importance(
                    self._pred, X.shape[1]
                )  # booster-defined metric for the smooth member
                w = self._smooth_gate(X, y, _smooth_weight(len(y)), fw, precomp=_oof)  # OOF safety gate
                if w > 0:
                    self._smooth = _SmoothKNN(
                        X,
                        y,
                        self._pred.classes_,
                        fw=fw,
                        k=self._smooth_k,
                    )
                    self._smooth_w = w
                    if self._smooth_oof_proba is not None:
                        blended = self._smooth_oof_proba
                        labels = np.asarray(self._pred.classes_)[blended.argmax(1)]
                        rows = getattr(self, "_validation_evidence_rows", None)
                        rows = np.arange(len(y)) if rows is None else np.asarray(rows, dtype=int)
                        self._cal_conf = (
                            blended[rows].max(1),
                            (labels[rows] == np.asarray(y)[rows]).astype(float),
                        )
            # Category-evidence member: finite one-hot agreements selected
            # from the same deployed OOF predictions as the local member. It
            # cannot change the certified class.
            if (
                auto_cfg
                and self.mode == "classification"
                and _oof is not None
                and getattr(self, "_categorical_evidence", False)
            ):
                w = self._category_memory_gate(X, y, precomp=_oof)
                if w > 0:
                    self._category_memory = _CategoricalEvidenceMemory(
                        X, y, self._pred.classes_, _onehot_groups(self._prep), seed=self.seed
                    )
                    self._category_memory_w = w
                    if self._category_memory_oof_proba is not None:
                        blended = self._category_memory_oof_proba
                        labels = np.asarray(self._pred.classes_)[blended.argmax(1)]
                        rows = getattr(self, "_validation_evidence_rows", None)
                        rows = np.arange(len(y)) if rows is None else np.asarray(rows, dtype=int)
                        self._cal_conf = (
                            blended[rows].max(1),
                            (labels[rows] == np.asarray(y)[rows]).astype(float),
                        )
            # Proof-path memory reads the residual probability geometry after
            # any selected local/category member. It still reuses the same
            # fold models, so no new boosted-model training is introduced.
            if (
                auto_cfg
                and self.mode == "classification"
                and _oof is not None
                and getattr(self, "_proof_path_evidence", False)
            ):
                w = self._proof_path_memory_gate(X, y, precomp=_oof)
                if w > 0:
                    try:
                        self._proof_path_memory = _ProofPathMemory(self._pred, X, y, self._pred.classes_)
                    except Exception:
                        self._proof_path_oof_proba = None  # final index must fail closed too
                    else:
                        self._proof_path_memory_w = w
                        if self._proof_path_oof_proba is not None:
                            blended = self._proof_path_oof_proba
                            labels = np.asarray(self._pred.classes_)[blended.argmax(1)]
                            rows = getattr(self, "_validation_evidence_rows", None)
                            rows = np.arange(len(y)) if rows is None else np.asarray(rows, dtype=int)
                            self._cal_conf = (
                                blended[rows].max(1),
                                (labels[rows] == np.asarray(y)[rows]).astype(float),
                            )
            # This explicit count posterior runs last. Its gate may grant a
            # projected rank refinement or the stricter authority to correct
            # the boosted class; either result refreshes probability confidence
            # from the exact deployed OOF surface.
            if (
                auto_cfg
                and self.mode == "classification"
                and _oof is not None
                and getattr(self, "_categorical_posterior_evidence", False)
            ):
                weight = self._category_posterior_gate(X, y, precomp=_oof)
                if weight > 0:
                    previous = (
                        self._conf,
                        self._cal_conf,
                        self._bal_thr,
                        self._rare_thr,
                    )
                    groups, metadata = _onehot_group_metadata(self._prep)
                    try:
                        posterior = CategoricalPosteriorChallenger(
                            X,
                            y,
                            self._pred.classes_,
                            groups,
                            metadata=metadata,
                            aggregation=self._category_posterior_aggregation,
                            smoothing=self._category_posterior_smoothing,
                        )
                        self._refresh_classification_confidence(X, y, self._category_posterior_oof_proba)
                    except (TypeError, ValueError, FloatingPointError):
                        self._conf, self._cal_conf, self._bal_thr, self._rare_thr = previous
                        self._category_posterior_permission = None
                        self._category_posterior_aggregation = None
                        self._category_posterior_smoothing = None
                        self._category_posterior_oof_proba = None
                        self.category_posterior_report_[0]["selected"] = True
                        self.category_posterior_report_[-1].update(
                            selected=False,
                            reason="final_posterior_fit_failed_closed",
                        )
                    else:
                        self._category_posterior = posterior
                        self._category_posterior_w = weight
                        self.category_posterior_report_[-1]["final_evidence"] = posterior.report()
            # Numeric intervals challenge only the final class decision. Their
            # hidden posterior may change predict(), but never predict_proba(),
            # preserving the Arena ranking surface while retaining OOF-supported
            # accuracy corrections.
            if (
                auto_cfg
                and self.mode == "classification"
                and _oof is not None
                and getattr(self, "_numeric_interval_evidence", False)
            ):
                weight = self._numeric_interval_gate(X, y, precomp=_oof)
                if weight > 0:
                    previous_confidence = self._conf
                    columns, names = self._numeric_interval_columns(X)
                    try:
                        interval = NumericIntervalPosteriorChallenger(
                            X,
                            y,
                            self._pred.classes_,
                            columns,
                            names=names,
                            aggregation=self._numeric_interval_aggregation,
                            smoothing=self._numeric_interval_smoothing,
                        )
                        self._refresh_classification_decision_confidence(
                            X, y, self._numeric_interval_oof_labels
                        )
                    except (TypeError, ValueError, FloatingPointError):
                        self._conf = previous_confidence
                        self._numeric_interval_aggregation = None
                        self._numeric_interval_smoothing = None
                        self._numeric_interval_permission = None
                        self._numeric_interval_oof_proba = None
                        self._numeric_interval_oof_labels = None
                        self.numeric_interval_report_[-1].update(
                            selected=False,
                            reason="final_interval_fit_failed_closed",
                        )
                    else:
                        self._numeric_interval = interval
                        self._numeric_interval_w = weight
                        self.numeric_interval_report_[-1]["final_evidence"] = interval.report()
            # A strongly regularized global affine read complements local tree
            # regions. It normally remains class-preserving; a separate paired
            # accuracy gate may grant top-1 authority when every OOF fold is
            # non-losing and the aggregate decision lift is material.
            if (
                auto_cfg
                and self.mode == "classification"
                and _oof is not None
                and getattr(self, "_affine_rank_evidence", False)
            ):
                weight = self._global_affine_rank_gate(X, y, precomp=_oof)
                if weight > 0.0:
                    previous_confidence = self._conf
                    try:
                        affine_rank = AffineLogitRead(
                            inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
                            seed=self.seed,
                        ).fit(X, y, classes=self._pred.classes_)
                        if self._affine_rank_permission in {
                            "decision_only",
                            "decision_and_rank",
                        }:
                            labels = self._affine_decision_oof_labels
                            if labels is None:
                                raise ValueError("affine decision permission requires OOF labels")
                            self._refresh_classification_decision_confidence(
                                X,
                                y,
                                labels,
                            )
                    except (TypeError, ValueError, FloatingPointError):
                        self._conf = previous_confidence
                        self._affine_rank_permission = None
                        self._affine_composition = "arithmetic"
                        self._affine_rank_oof_proba = None
                        self._affine_decision_oof_labels = None
                        self.affine_rank_report_[-1].update(
                            selected=False,
                            reason="final_affine_fit_failed_closed",
                        )
                    else:
                        self._affine_rank = affine_rank
                        self._affine_rank_weight = weight
                        self.affine_rank_report_[-1]["final_evidence"] = {
                            **affine_rank.report(),
                            "composition": self._affine_composition,
                        }
            # Prior tempering is a public rank-only view, so its gate must read
            # the exact final OOF probability stack after every admitted
            # smooth/category/path/posterior member. It remains absent from the
            # numeric decision head and calibrated decision economics.
            if (
                auto_cfg
                and self.mode == "classification"
                and len(self._pred.classes_) > 2
                and _oof is not None
            ):
                self._multiclass_prior_rank_gate(y, _oof)
            self._smooth_oof_proba = None
            self._prior_rank_oof_proba = None
            self._category_memory_oof_proba = None
            self._proof_path_oof_proba = None
            self._category_posterior_oof_proba = None
            self._numeric_interval_oof_proba = None
            self._numeric_interval_oof_labels = None
            self._affine_rank_oof_proba = None
            self._affine_decision_oof_labels = None
            # SDM-attention member: TEXT classification, blended if it CLEARLY helps out-of-fold (zero-downside).
            # An associative-memory read over the token vectors — complements the booster on graded text.
            if (
                auto_cfg
                and self.mode == "classification"
                and not self.rare_event_
                and self._has_text()
                and self._affine_rank is None
            ):
                w = self._sdm_gate(X, y)
                if w > 0:
                    cols = [j for j, nm in enumerate(self.feature_names_) if "~" in str(nm)]
                    self._sdm = _SDMAttention(X, y, self._pred.classes_, cols, seed=self.seed)
                    self._sdm_w = w
        else:
            raise RuntimeError("non-additive research predictors are not part of the production runtime")
        return self

    def _auto_target_encoding(self, raw_X, y):
        """Select frequency, posterior-mean, or Gaussian category evidence on held-out rows."""
        from sklearn.metrics import mean_squared_error, roc_auc_score

        self.target_encoding_variant_ = "none"
        self.target_encoding_report_ = []
        y = np.asarray(y)
        n = len(y)
        if n < 200:
            self.target_encoding_report_.append(
                {
                    "selected": False,
                    "variant": "none",
                    "reason": "insufficient_rows",
                    "rows": int(n),
                }
            )
            return False
        gate_idx = np.arange(n)
        # The decision is coarse (keep/drop a representation), so a bounded
        # sample is enough on large tables and keeps the zero-knob fit bounded.
        if n > 6_000:
            selected, _weight = self._bounded_evidence_rows(
                y,
                6_000,
                self.seed + 17,
                stratified=self.mode == "classification",
            )
            if selected is not None:
                gate_idx = selected
        raw = raw_X.iloc[gate_idx] if hasattr(raw_X, "iloc") else np.asarray(raw_X, object)[gate_idx]
        yg = y[gate_idx]
        groups = self._validation_groups(gate_idx)
        try:
            tr, va = self._validation_splits(
                yg,
                folds=4,
                classification=self.mode == "classification",
                groups=groups,
                single=True,
            )[0]

            def take(rows):
                return raw.iloc[rows] if hasattr(raw, "iloc") else raw[rows]

            plain_prep = _Preprocessor(
                target_encoding=False,
                task=self.mode,
                compression_evidence=False,
            )
            target_prep = _Preprocessor(
                target_encoding=True,
                task=self.mode,
                compression_evidence=False,
            )
            train_groups = None if groups is None else groups[tr]
            plain_tr = plain_prep.fit_transform(
                take(tr),
                yg[tr],
                validation_groups=train_groups,
            )
            target_tr = target_prep.fit_transform(
                take(tr),
                yg[tr],
                validation_groups=train_groups,
            )
            # The proposer must earn its place from a feature that actually
            # survived its own OOF signal check on the fitting partition.
            if not any(target_prep.target_enabled.values()):
                self.target_encoding_report_.append(
                    {
                        "selected": False,
                        "variant": "none",
                        "reason": "no_oof_target_signal",
                        "rows": int(len(yg)),
                    }
                )
                return False
            plain_va = plain_prep.transform(take(va))
            target_va = target_prep.transform(take(va))
            leaf = int(np.clip(len(tr) // (150 if self.mode == "classification" else 600), 15, 50))
            if self.mode == "classification":
                gate_cfg = dict(rounds=400, lr=0.05, depth=6, leaf=leaf, patience=40, refit=False)
                base = self._fit_certified(
                    AdditiveCertifiedClassifier(seed=self.seed, **gate_cfg),
                    plain_tr,
                    yg[tr],
                    groups=train_groups,
                )
                encoded = self._fit_certified(
                    AdditiveCertifiedClassifier(seed=self.seed, **gate_cfg),
                    target_tr,
                    yg[tr],
                    groups=train_groups,
                )

                def score(model, X):
                    logits = model._scores(X)
                    probs = np.exp(logits - logits.max(1, keepdims=True))
                    probs /= probs.sum(1, keepdims=True)
                    if probs.shape[1] == 2:
                        return float(roc_auc_score(yg[va], probs[:, 1]))
                    return float(roc_auc_score(yg[va], probs, multi_class="ovo", average="macro"))

                base_score, encoded_score = score(base, plain_va), score(encoded, target_va)
                selected = encoded_score >= base_score + 0.003
                self.target_encoding_variant_ = "mean" if selected else "none"
                self.target_encoding_report_.append(
                    {
                        "selected": bool(selected),
                        "variant": self.target_encoding_variant_,
                        "metric": "roc_auc" if len(np.unique(yg)) == 2 else "macro_ovo_auc",
                        "frequency_score": float(base_score),
                        "mean_score": float(encoded_score),
                        "required_gain": 0.003,
                        "reason": "posterior_mean_win" if selected else "mean_gain_below_gate",
                        "rows": int(len(yg)),
                    }
                )
                return bool(selected)

            gate_cfg = dict(rounds=500, lr=0.03, depth=6, leaf=leaf, patience=40, refit=False)
            base = self._fit_certified(
                AdditiveCertifiedRegressor(seed=self.seed, **gate_cfg),
                plain_tr,
                yg[tr],
                groups=train_groups,
            )
            encoded = self._fit_certified(
                AdditiveCertifiedRegressor(seed=self.seed, **gate_cfg),
                target_tr,
                yg[tr],
                groups=train_groups,
            )
            truth = yg[va]
            base_prediction = base.predict(plain_va)
            mean_prediction = encoded.predict(target_va)
            base_rmse = float(mean_squared_error(truth, base_prediction) ** 0.5)
            mean_rmse = float(mean_squared_error(truth, mean_prediction) ** 0.5)
            mean_required_gain = max(0.003 * base_rmse, 1e-6)
            mean_selected = bool(mean_rmse <= base_rmse - mean_required_gain)
            self.target_encoding_variant_ = "mean" if mean_selected else "none"
            report = {
                "selected": mean_selected,
                "variant": self.target_encoding_variant_,
                "metric": "rmse",
                "frequency_rmse": base_rmse,
                "mean_rmse": mean_rmse,
                "mean_required_gain": mean_required_gain,
                "reason": "posterior_mean_win" if mean_selected else "mean_gain_below_gate",
                "rows": int(len(yg)),
            }
            self.target_encoding_report_.append(report)

            # Both target-derived representations must earn their place against
            # frequency. If the mean already wins, Gaussian evidence must then
            # beat that stronger reference rather than merely beat frequency.
            try:
                gaussian_prep = _Preprocessor(
                    target_encoding=True,
                    task="regression",
                    compression_evidence=False,
                    gaussian_target_statistics=True,
                )
                gaussian_tr = gaussian_prep.fit_transform(
                    take(tr),
                    yg[tr],
                    validation_groups=train_groups,
                )
                if not any(gaussian_prep.target_enabled.values()):
                    report["reason"] = "gaussian_no_oof_target_signal"
                    return mean_selected
                gaussian_va = gaussian_prep.transform(take(va))
                gaussian = self._fit_certified(
                    AdditiveCertifiedRegressor(seed=self.seed, **gate_cfg),
                    gaussian_tr,
                    yg[tr],
                    groups=train_groups,
                )
                gaussian_prediction = gaussian.predict(gaussian_va)
                gaussian_rmse = float(mean_squared_error(truth, gaussian_prediction) ** 0.5)
                reference_name = "mean" if mean_selected else "frequency"
                reference_prediction = mean_prediction if mean_selected else base_prediction
                reference_rmse = mean_rmse if mean_selected else base_rmse
                required_fraction = 0.0015 if mean_selected else 0.003
                gaussian_required_gain = max(required_fraction * reference_rmse, 1e-6)

                window_gains = []
                window_consistent = True
                for window in np.array_split(np.arange(len(truth)), 2):
                    if len(window) == 0:
                        continue
                    reference_window_rmse = float(
                        mean_squared_error(truth[window], reference_prediction[window]) ** 0.5
                    )
                    gaussian_window_rmse = float(
                        mean_squared_error(truth[window], gaussian_prediction[window]) ** 0.5
                    )
                    window_gains.append(reference_window_rmse - gaussian_window_rmse)
                    window_consistent &= gaussian_window_rmse <= reference_window_rmse + max(
                        0.001 * reference_window_rmse,
                        1e-6,
                    )

                gaussian_selected = bool(
                    gaussian_rmse <= reference_rmse - gaussian_required_gain and window_consistent
                )
                if gaussian_selected:
                    self.target_encoding_variant_ = "gaussian"
                target_selected = bool(mean_selected or gaussian_selected)
                report.update(
                    {
                        "selected": target_selected,
                        "variant": self.target_encoding_variant_,
                        "gaussian_rmse": gaussian_rmse,
                        "gaussian_reference": reference_name,
                        "gaussian_required_gain": gaussian_required_gain,
                        "gaussian_window_gains": [float(gain) for gain in window_gains],
                        "gaussian_window_consistent": bool(window_consistent),
                        "reason": (
                            "gaussian_challenger_win"
                            if gaussian_selected
                            else (
                                "gaussian_window_regression"
                                if not window_consistent
                                else "gaussian_gain_below_gate"
                            )
                        ),
                    }
                )
                return target_selected
            except Exception as error:
                report["reason"] = f"gaussian_failed:{type(error).__name__}"
                return mean_selected
        except Exception as error:
            # The base representation is always valid; a failed optional
            # proposer must never make fitting fail or silently stay enabled.
            self.target_encoding_variant_ = "none"
            self.target_encoding_report_.append(
                {
                    "selected": False,
                    "variant": "none",
                    "reason": f"gate_failed:{type(error).__name__}",
                    "rows": int(len(yg)),
                }
            )
            return False

    def _multiclass_no_signal_gate(self, y, precomp):
        """Fall back to the class prior when dominant multiclass OOF is indistinguishable from noise.

        This gate is deliberately conjunctive. The rank score must remain
        inside a one-sided random-ranking bound, top-1 accuracy may not beat
        the majority prior, and log-loss must lack a material improvement.
        The prior is projected into the certified class, so the fallback never
        invalidates the booster's proof-carrying decision. Later OOF-gated
        probability members may still challenge this conservative surface.
        """
        self._no_signal_prior = None
        self.multiclass_signal_report_ = [
            {
                "name": "multiclass_prior_fallback",
                "selected": False,
                "reason": "not_evaluated",
            }
        ]
        if precomp is None or self._fit_validation is not None:
            self.multiclass_signal_report_[0]["reason"] = "shared_exchangeable_oof_unavailable"
            return None

        classes = np.asarray(self._pred.classes_)
        labels = np.asarray(y)
        if len(classes) < 3:
            self.multiclass_signal_report_[0]["reason"] = "not_multiclass"
            return None
        class_index = {value: index for index, value in enumerate(classes)}
        try:
            yi = np.asarray([class_index[value] for value in labels], dtype=np.int32)
            rows = np.asarray(precomp.get("evidence_rows", np.arange(len(labels))), dtype=np.int64)
            probability = self._oof_probability_base(precomp)
            counts = np.bincount(yi[rows], minlength=len(classes)).astype(float)
            if probability.shape != (len(labels), len(classes)) or not len(rows) or np.any(counts < 2):
                raise ValueError("invalid multiclass OOF evidence")
        except (KeyError, TypeError, ValueError, FloatingPointError):
            self.multiclass_signal_report_[0]["reason"] = "oof_signal_check_failed_closed"
            return None

        prior = counts / counts.sum()
        dominant_rate = float(prior.max())
        if dominant_rate < _MULTICLASS_NO_SIGNAL_MIN_DOMINANT_RATE:
            self.multiclass_signal_report_[0].update(
                reason="class_prior_not_dominant",
                dominant_rate=dominant_rate,
            )
            return None

        rank_auc = _classification_rank_score(yi[rows], probability[rows])
        null_upper_bound = _multiclass_null_auc_upper_bound(counts)
        accuracy = float(np.mean(probability[rows].argmax(1) == yi[rows]))
        majority_accuracy = dominant_rate
        base_log_loss = float(-np.log(np.clip(probability[rows, yi[rows]], 1e-300, 1.0)).mean())
        prior_probability = np.tile(prior, (len(rows), 1))
        prior_log_loss = float(
            -np.log(np.clip(prior_probability[np.arange(len(rows)), yi[rows]], 1e-300, 1.0)).mean()
        )
        log_loss_gain = prior_log_loss - base_log_loss
        selected = bool(
            rank_auc <= null_upper_bound
            and accuracy <= majority_accuracy + 1e-12
            and log_loss_gain <= _MULTICLASS_NO_SIGNAL_MIN_LOG_LOSS_GAIN
        )
        self.multiclass_signal_report_ = [
            {
                "name": "selected_probability_stack",
                "selected": not selected,
                "stage": "probability_signal",
                "oof_rank_auc": float(rank_auc),
                "oof_accuracy": accuracy,
                "oof_log_loss": base_log_loss,
            },
            {
                "name": "multiclass_prior_fallback",
                "selected": selected,
                "stage": "probability_signal",
                "reason": (
                    "oof_indistinguishable_from_prior" if selected else "oof_predictive_signal_supported"
                ),
                "dominant_rate": dominant_rate,
                "null_rank_auc_upper_bound": float(null_upper_bound),
                "oof_rank_auc": float(rank_auc),
                "oof_accuracy": accuracy,
                "majority_accuracy": majority_accuracy,
                "oof_log_loss": base_log_loss,
                "prior_log_loss": prior_log_loss,
                "log_loss_gain": float(log_loss_gain),
                "minimum_log_loss_gain": _MULTICLASS_NO_SIGNAL_MIN_LOG_LOSS_GAIN,
            },
        ]
        if not selected:
            return None
        deployment_prior = np.asarray(getattr(self, "_prior_train", ()), dtype=float)
        if (
            deployment_prior.shape != prior.shape
            or not np.isfinite(deployment_prior).all()
            or deployment_prior.sum() <= 0.0
        ):
            deployment_prior = prior
        else:
            deployment_prior = deployment_prior / deployment_prior.sum()
        self._no_signal_prior = deployment_prior
        self.multiclass_signal_report_[-1]["deployment_prior"] = deployment_prior.tolist()
        return _preserve_certified_class(
            probability,
            np.tile(deployment_prior, (len(probability), 1)),
        )

    def _multiclass_prior_rank_gate(self, y, precomp):
        """Admit a class-preserving half-prior rank surface from shared OOF evidence."""
        self._prior_rank_strength = 0.0
        self._prior_rank_oof_proba = None
        self.prior_rank_report_ = [
            gate_report(
                "multiclass_prior_rank_projection",
                False,
                stage="probability_rank",
                reason="not_evaluated",
            )
        ]
        if precomp is None:
            self.prior_rank_report_[0]["reason"] = "shared_oof_unavailable"
            return False

        classes = np.asarray(self._pred.classes_)
        prior = np.asarray(getattr(self, "_prior_train", ()), dtype=float)
        if len(classes) < 3 or prior.shape != (len(classes),):
            self.prior_rank_report_[0]["reason"] = "not_multiclass"
            return False
        dominant_rate = float(prior.max())
        if dominant_rate < _MULTICLASS_PRIOR_RANK_MIN_DOMINANT_RATE:
            self.prior_rank_report_[0].update(
                reason="class_prior_not_dominant",
                dominant_rate=dominant_rate,
            )
            return False

        y = np.asarray(y)
        class_index = {value: index for index, value in enumerate(classes)}
        try:
            yi = np.asarray([class_index[value] for value in y], dtype=np.int32)
            splits = tuple(precomp["splits"])
            evidence_rows = np.asarray(
                precomp.get("evidence_rows", np.arange(len(y))),
                dtype=np.int64,
            )
            base = self._oof_probability_stack(precomp)
            if base.shape != (len(y), len(classes)) or not len(splits) or not len(evidence_rows):
                raise ValueError("invalid shared OOF geometry")
            if any(np.unique(yi[valid]).size != len(classes) for _train, valid in splits):
                raise ValueError("an OOF fold is missing a class")
            candidate = _multiclass_prior_rank_projection(base, prior)
            base_score = _classification_rank_score(yi[evidence_rows], base[evidence_rows])
            candidate_score = _classification_rank_score(yi[evidence_rows], candidate[evidence_rows])
            fold_deltas = np.asarray(
                [
                    _classification_rank_score(yi[valid], candidate[valid])
                    - _classification_rank_score(yi[valid], base[valid])
                    for _train, valid in splits
                ],
                dtype=float,
            )
            base_log_loss = float(
                -np.log(np.clip(base[evidence_rows, yi[evidence_rows]], 1e-300, 1.0)).mean()
            )
            candidate_log_loss = float(
                -np.log(np.clip(candidate[evidence_rows, yi[evidence_rows]], 1e-300, 1.0)).mean()
            )
            class_preserved = bool(np.array_equal(base.argmax(1), candidate.argmax(1)))
        except (KeyError, TypeError, ValueError, FloatingPointError):
            self.prior_rank_report_[0]["reason"] = "oof_projection_failed_closed"
            return False

        gain = candidate_score - base_score
        selected = bool(
            class_preserved
            and gain >= _MULTICLASS_PRIOR_RANK_MIN_GAIN
            and np.all(fold_deltas >= _MULTICLASS_PRIOR_RANK_MIN_FOLD_GAIN)
        )
        reason = "consistent_oof_rank_gain" if selected else "insufficient_transferable_rank_gain"
        self.prior_rank_report_ = [
            gate_report(
                "selected_probability_stack",
                not selected,
                stage="probability_rank",
                metric="macro_ovo_auc",
                mean_score=base_score,
                oof_log_loss=base_log_loss,
            ),
            gate_report(
                "multiclass_prior_rank_projection",
                selected,
                stage="probability_rank",
                metric="macro_ovo_auc",
                mean_score=candidate_score,
                rank_auc_delta=gain,
                fold_auc_delta=[float(delta) for delta in fold_deltas],
                oof_log_loss=candidate_log_loss,
                log_loss_delta=candidate_log_loss - base_log_loss,
                dominant_rate=dominant_rate,
                strength=_MULTICLASS_PRIOR_RANK_STRENGTH,
                class_preserved=class_preserved,
                minimum_rank_gain=_MULTICLASS_PRIOR_RANK_MIN_GAIN,
                minimum_fold_gain=_MULTICLASS_PRIOR_RANK_MIN_FOLD_GAIN,
                reason=reason,
            ),
        ]
        if selected:
            self._prior_rank_strength = _MULTICLASS_PRIOR_RANK_STRENGTH
            self._prior_rank_oof_proba = candidate
        return selected

    def _oof_probability_base(self, precomp):
        """Return the calibrated booster OOF surface used by probability challengers."""
        scores = np.asarray(precomp["scores"], dtype=float) / getattr(self, "_temp", 1.0)
        probability = np.exp(scores - scores.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        prior = getattr(self, "_no_signal_prior", None)
        if prior is not None and np.shape(prior) == (probability.shape[1],):
            probability = _preserve_certified_class(
                probability,
                np.tile(prior, (len(probability), 1)),
            )
        return probability

    def _oof_probability_stack(self, precomp):
        """Return the exact selected OOF stack before the final prior-rank view."""
        probability = self._oof_probability_base(precomp)
        for attr in (
            "_smooth_oof_proba",
            "_category_memory_oof_proba",
            "_proof_path_oof_proba",
            "_category_posterior_oof_proba",
            "_numeric_interval_oof_proba",
            "_affine_rank_oof_proba",
        ):
            selected = getattr(self, attr, None)
            if selected is not None:
                probability = selected
        return probability

    def _affine_schema_profile(self, n_features):
        """Describe whether a text-bearing table still has enough structured signal."""
        n_features = int(n_features)
        has_text = self._has_text()
        token_features: int = 0
        if has_text and self._prep is not None:
            prep = self._prep
            try:
                token_features = int(sum(len(prep.text_feat[column].vocab) for column in prep.text_cols))
            except (AttributeError, KeyError, TypeError):
                token_features = sum(1 for name in (self.feature_names_ or ()) if "~" in str(name))
        structured_features = max(n_features - token_features, 0)
        token_fraction = token_features / max(n_features, 1)
        eligible = not has_text or (
            token_features > 0
            and structured_features >= _AFFINE_MIXED_MIN_STRUCTURED_FEATURES
            and token_fraction <= _AFFINE_MIXED_MAX_TOKEN_FRACTION
        )
        return {
            "eligible": bool(eligible),
            "features": n_features,
            "schema": "mixed_structured_text" if has_text else "structured",
            "structured_features": int(structured_features),
            "token_features": int(token_features),
            "token_fraction": float(token_fraction),
        }

    def _global_affine_rank_gate(self, X, y, precomp):
        """Admit affine logits for rank, and separately gate top-1 authority."""
        classes = np.asarray(self._pred.classes_)
        labels = np.asarray(y)
        self._affine_rank_permission = None
        self._affine_composition = "arithmetic"
        self._affine_rank_oof_proba = None
        self._affine_decision_oof_labels = None
        self.affine_rank_report_ = [
            gate_report(
                "global_affine_rank",
                False,
                stage="probability_rank",
                reason="not_evaluated",
            )
        ]
        weight = _affine_rank_weight(len(labels))
        if precomp is None:
            self.affine_rank_report_[0]["reason"] = "shared_oof_unavailable"
            return 0.0
        if len(labels) < 200 or len(labels) > _AFFINE_RANK_MAX_N or weight <= 0.0:
            self.affine_rank_report_[0].update(
                reason="outside_small_table_regime",
                rows=int(len(labels)),
            )
            return 0.0
        schema = self._affine_schema_profile(X.shape[1])
        if X.shape[1] > _AFFINE_RANK_MAX_FEATURES or not schema["eligible"]:
            self.affine_rank_report_[0].update(
                reason=(
                    "outside_compact_feature_regime"
                    if X.shape[1] > _AFFINE_RANK_MAX_FEATURES
                    else "text_dominates_mixed_schema"
                ),
                **schema,
            )
            return 0.0

        class_index = {value: index for index, value in enumerate(classes)}
        try:
            encoded = np.asarray([class_index[value] for value in labels], dtype=np.int32)
            splits = tuple(precomp["splits"])
            evidence_rows = np.asarray(
                precomp.get("evidence_rows", np.arange(len(labels))),
                dtype=np.int64,
            )
            baseline = self._oof_probability_stack(precomp)
            full_counts = np.bincount(encoded, minlength=len(classes)).astype(float)
            full_prior = full_counts / full_counts.sum()
            affine_probability = baseline.copy()
            fold_prior_probability = np.tile(full_prior, (len(labels), 1))
            covered = np.zeros(len(labels), dtype=bool)
            fold_reports = []
            for train, valid in splits:
                member = AffineLogitRead(
                    inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
                    seed=self.seed,
                ).fit(X[train], labels[train], classes=classes)
                affine_probability[valid] = member.proba(X[valid])
                fold_counts = np.bincount(encoded[train], minlength=len(classes)).astype(float)
                if (fold_counts <= 0.0).any():
                    raise ValueError("every affine fold must train on every class")
                fold_prior_probability[valid] = fold_counts / fold_counts.sum()
                covered[valid] = True
                fold_reports.append(member.report())
            if not np.all(covered[evidence_rows]):
                raise ValueError("affine evidence rows require fold-local predictions")
            candidates = {
                "arithmetic": AffineLogitRead.combine(
                    baseline,
                    affine_probability,
                    weight,
                ),
                "prior_ratio": AffineLogitRead.combine(
                    baseline,
                    affine_probability,
                    weight,
                    composition="prior_ratio",
                    prior=fold_prior_probability,
                ),
            }
            for candidate in candidates.values():
                candidate[~covered] = baseline[~covered]
            decision_eligible = getattr(self, "_numeric_interval_oof_labels", None) is None
            evaluations = {
                composition: _global_probability_candidate_evaluation(
                    encoded,
                    baseline,
                    candidate,
                    evidence_rows,
                    splits,
                    composition=composition,
                    decision_eligible=decision_eligible,
                )
                for composition, candidate in candidates.items()
            }
        except (KeyError, TypeError, ValueError, FloatingPointError):
            self.affine_rank_report_[0]["reason"] = "affine_oof_failed_closed"
            return 0.0

        arithmetic = evaluations["arithmetic"]
        baseline_prediction = arithmetic["baseline_prediction"]
        baseline_score = float(arithmetic["baseline_score"])
        baseline_accuracy = float(arithmetic["baseline_accuracy"])
        baseline_loss = float(arithmetic["baseline_loss"])
        log_loss_tolerance = float(arithmetic["log_loss_tolerance"])
        decision_winners = [
            evaluation for evaluation in evaluations.values() if evaluation["decision_selected"]
        ]
        decision = (
            max(
                decision_winners,
                key=lambda evaluation: (
                    evaluation["decision_accuracy"],
                    evaluation["paired_z"],
                    evaluation["decision_score"],
                    evaluation["composition"] == "arithmetic",
                ),
            )
            if decision_winners
            else None
        )
        rank_winners = [evaluation for evaluation in evaluations.values() if evaluation["rank_selected"]]
        rank = (
            max(
                rank_winners,
                key=lambda evaluation: (
                    evaluation["rank_score"],
                    evaluation["composition"] == "arithmetic",
                ),
            )
            if rank_winners
            else None
        )
        if decision is not None:
            decision_rank_selected = bool(decision["decision_rank_selected"])
            permission = "decision_and_rank" if decision_rank_selected else "decision_only"
            selected_probability = decision["candidate"] if decision_rank_selected else None
            selected_score = float(decision["decision_score"])
            selected_fold_deltas = decision["decision_fold_deltas"]
            selected_loss = float(decision["decision_loss"])
            representative = decision
            self._affine_composition = str(decision["composition"])
            reason = (
                "consistent_oof_accuracy_and_rank_gain"
                if decision_rank_selected
                else "consistent_oof_accuracy_gain"
            )
        elif rank is not None:
            permission = "rank_only"
            selected_probability = rank["projected"]
            selected_score = float(rank["rank_score"])
            selected_fold_deltas = rank["rank_fold_deltas"]
            selected_loss = float(rank["rank_loss"])
            representative = rank
            self._affine_composition = str(rank["composition"])
            reason = "consistent_oof_rank_gain"
        else:
            permission = None
            selected_probability = None
            selected_score = float(arithmetic["rank_score"])
            selected_fold_deltas = arithmetic["rank_fold_deltas"]
            selected_loss = float(arithmetic["rank_loss"])
            representative = arithmetic
            reason = "insufficient_transferable_rank_gain"
        selected = permission is not None
        rank_representative = rank if rank is not None else arithmetic
        candidate_reports = [
            {
                "composition": evaluation["composition"],
                "decision_selected": bool(evaluation["decision_selected"]),
                "decision_rank_selected": bool(evaluation["decision_rank_selected"]),
                "decision_oof_accuracy": float(evaluation["decision_accuracy"]),
                "accuracy_gain": float(evaluation["accuracy_gain"]),
                "fold_accuracy_delta": [float(delta) for delta in evaluation["fold_accuracy_deltas"]],
                "fold_net_wins": list(evaluation["fold_net_wins"]),
                "wins": int(evaluation["wins"]),
                "losses": int(evaluation["losses"]),
                "paired_z": float(evaluation["paired_z"]),
                "override_rows": int(np.sum(evaluation["decision_prediction"] != baseline_prediction)),
                "decision_rank_auc": float(evaluation["decision_score"]),
                "decision_rank_auc_delta": float(evaluation["decision_score"] - baseline_score),
                "decision_fold_rank_auc_delta": [
                    float(delta) for delta in evaluation["decision_fold_deltas"]
                ],
                "decision_oof_log_loss": float(evaluation["decision_loss"]),
                "decision_log_loss_delta": float(evaluation["decision_loss"] - baseline_loss),
                "rank_only_selected": bool(evaluation["rank_selected"]),
                "rank_only_oof_rank_auc": float(evaluation["rank_score"]),
                "rank_only_rank_auc_delta": float(evaluation["rank_score"] - baseline_score),
                "rank_only_fold_auc_delta": [float(delta) for delta in evaluation["rank_fold_deltas"]],
                "rank_only_oof_log_loss": float(evaluation["rank_loss"]),
                "rank_only_log_loss_delta": float(evaluation["rank_loss"] - baseline_loss),
            }
            for evaluation in evaluations.values()
        ]
        self.affine_rank_report_ = [
            gate_report(
                "selected_probability_stack",
                not selected,
                stage="probability_rank",
                metric="roc_auc" if len(classes) == 2 else "macro_ovo_auc",
                mean_score=baseline_score,
                oof_log_loss=baseline_loss,
            ),
            gate_report(
                "global_affine_rank",
                selected,
                stage="probability_rank",
                metric="roc_auc" if len(classes) == 2 else "macro_ovo_auc",
                mean_score=selected_score,
                rank_auc_delta=selected_score - baseline_score,
                fold_auc_delta=[float(delta) for delta in selected_fold_deltas],
                oof_log_loss=selected_loss,
                log_loss_delta=selected_loss - baseline_loss,
                weight=weight,
                permission=permission,
                composition=self._affine_composition,
                oof_prior_source="fold_training_rows",
                candidate_selection="decision_accuracy_or_best_projected_rank",
                candidates=candidate_reports,
                inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
                minimum_rank_gain=_AFFINE_RANK_MIN_GAIN,
                minimum_fold_gain=_AFFINE_RANK_MIN_FOLD_GAIN,
                **schema,
                class_preserved=permission != "decision_and_rank",
                rank_only_composition=str(rank_representative["composition"]),
                rank_only_oof_rank_auc=float(rank_representative["rank_score"]),
                rank_only_rank_auc_delta=float(rank_representative["rank_score"] - baseline_score),
                rank_only_oof_log_loss=float(rank_representative["rank_loss"]),
                rank_only_log_loss_delta=float(rank_representative["rank_loss"] - baseline_loss),
                decision_oof_accuracy=float(representative["decision_accuracy"]),
                baseline_oof_accuracy=baseline_accuracy,
                accuracy_gain=float(representative["accuracy_gain"]),
                fold_accuracy_delta=[float(delta) for delta in representative["fold_accuracy_deltas"]],
                fold_net_wins=list(representative["fold_net_wins"]),
                wins=int(representative["wins"]),
                losses=int(representative["losses"]),
                paired_z=float(representative["paired_z"]),
                override_rows=int(np.sum(representative["decision_prediction"] != baseline_prediction)),
                decision_rank_auc=float(representative["decision_score"]),
                decision_rank_auc_delta=float(representative["decision_score"] - baseline_score),
                decision_fold_rank_auc_delta=[
                    float(delta) for delta in representative["decision_fold_deltas"]
                ],
                decision_rank_selected=bool(representative["decision_rank_selected"]),
                decision_oof_log_loss=float(representative["decision_loss"]),
                decision_log_loss_delta=float(representative["decision_loss"] - baseline_loss),
                decision_eligible=decision_eligible,
                decision_blocked_by=(None if decision_eligible else "numeric_interval_decision"),
                minimum_accuracy_gain=_AFFINE_DECISION_MIN_ACCURACY_GAIN,
                minimum_paired_z=_AFFINE_DECISION_MIN_PAIRED_Z,
                maximum_rank_regression=_AFFINE_DECISION_MAX_RANK_REGRESSION,
                log_loss_tolerance=log_loss_tolerance,
                fold_evidence=fold_reports,
                reason=reason,
            ),
        ]
        if selected:
            self._affine_rank_permission = permission
            self._affine_rank_oof_proba = selected_probability
            if decision is not None:
                self._affine_decision_oof_labels = classes[decision["decision_prediction"]]
        return weight if selected else 0.0

    def _proof_path_memory_gate(self, X, y, precomp):
        """Admit certified path evidence only after a strict shared-OOF win.

        Unlike a learned stacker, the challenger is a deterministic lookup over
        rows satisfying the same flat-tree path prefixes. Fold models already
        exist in ``precomp`` for the local-member OOF, so the comparison is
        leak-safe without another booster fit. It challenges the probability
        surface after earlier selected members. The correction can never alter
        the FOL-certified boosted class.
        """
        self._proof_path_oof_proba = None
        self.proof_path_memory_report_ = [{"name": "proof_path_memory", "selected": False}]
        if precomp is None or self._has_text() or len(X) < 200 or len(X) > _PROOF_PATH_MEMORY_MAX_N:
            return 0.0
        classes = np.asarray(self._pred.classes_)
        ci = {value: idx for idx, value in enumerate(classes)}
        yi = np.array([ci[value] for value in np.asarray(y)])
        splits = precomp.get("splits", [])
        models = precomp.get("models", [])
        evidence_rows = np.asarray(
            precomp.get("evidence_rows", np.arange(len(yi))),
            dtype=int,
        )
        if (
            not splits
            or len(models) != len(splits)
            or np.unique(yi, return_counts=True)[1].min() < len(splits)
        ):
            return 0.0
        base = self._oof_probability_base(precomp)
        if getattr(self, "_smooth_oof_proba", None) is not None:
            base = self._smooth_oof_proba
        if getattr(self, "_category_memory_oof_proba", None) is not None:
            base = self._category_memory_oof_proba
        try:
            memory = np.zeros_like(base)
            reports = []
            for (tri, vai), fold_model in zip(splits, models, strict=False):
                member = _ProofPathMemory(fold_model, X[tri], y[tri], classes)
                memory[vai] = member.proba(X[vai])
                reports.append(member.index_report())
            base_score = _classification_rank_score(yi[evidence_rows], base[evidence_rows])
            base_fold = np.array([_classification_rank_score(yi[vai], base[vai]) for _tri, vai in splits])
            best = (base_score, 0.0, np.zeros(len(splits)), base)
            for weight in (0.1, 0.2, 0.3, 0.4, 0.5):
                blended = _preserve_certified_class(base, (1.0 - weight) * base + weight * memory)
                score = _classification_rank_score(yi[evidence_rows], blended[evidence_rows])
                fold_delta = np.array(
                    [
                        _classification_rank_score(yi[vai], blended[vai]) - base_fold[i]
                        for i, (_tri, vai) in enumerate(splits)
                    ]
                )
                if score > best[0]:
                    best = (score, weight, fold_delta, blended)
        except Exception:
            return 0.0
        # Path regions are a flexible derived view of the boosted program, so
        # their gate is intentionally stricter than direct category facts:
        # a broad aggregate lift or a single barely-positive fold has already
        # proved insufficient to transfer. Require material evidence globally
        # and in every independently fitted fold.
        selected = bool(
            best[1] > 0.0
            and best[0] >= base_score + _PROOF_PATH_MEMORY_MIN_RANK_GAIN
            and np.all(best[2] >= _PROOF_PATH_MEMORY_MIN_FOLD_GAIN)
        )
        self.proof_path_memory_report_ = [
            {"name": "certified_boost", "oof_rank_auc": float(base_score), "selected": not selected},
            {
                "name": "proof_path_memory",
                "oof_rank_auc": float(best[0]),
                "fold_auc_delta": [float(value) for value in best[2]],
                "weight": float(best[1]),
                "parameters": reports,
                "selected": selected,
            },
        ]
        if selected:
            self._proof_path_oof_proba = best[3]
        return float(best[1]) if selected else 0.0

    def _smooth_gate(self, X, y, w, fw, precomp=None):
        """Leak-safe gate for fixed and adaptive local probability geometry.

        Every OOF memory uses feature importance from its own fold booster; the
        validation labels therefore cannot shape its distance metric. Binary
        behavior retains the established fixed geometry. Multiclass compares
        that incumbent with one prespecified ``ceil(sqrt(n))`` neighborhood and
        admits the best transferable blend under the existing all-fold rank
        guard. Neither candidate requires another booster fit.
        """
        from sklearn.metrics import roc_auc_score

        classes = np.asarray(self._pred.classes_)
        ci = {c: i for i, c in enumerate(classes)}
        yi = np.array([ci[v] for v in y])
        self._smooth_k = _SMOOTH_FIXED_NEIGHBORS
        self._smooth_geometry = "fixed"
        self._smooth_oof_proba = None
        self.smooth_memory_report_ = [{"name": "smooth_consensus", "selected": False}]
        oob = np.zeros((len(y), len(classes)))
        specifications = [
            {
                "geometry": "fixed",
                "neighbors": min(_SMOOTH_FIXED_NEIGHBORS, len(y) - 1),
                "weight": float(w),
            }
        ]
        if len(classes) > 2:
            adaptive_weight = min(
                _MULTICLASS_ADAPTIVE_SMOOTH_MAX_WEIGHT,
                _MULTICLASS_ADAPTIVE_SMOOTH_WEIGHT_MULTIPLIER * float(w),
            )
            specifications.append(
                {
                    "geometry": "adaptive_sqrt",
                    "neighbors": _adaptive_smooth_neighbors(len(y)),
                    "weight": float(adaptive_weight),
                }
            )
        memories = {
            specification["geometry"]: np.zeros((len(y), len(classes))) for specification in specifications
        }

        def local_neighbors(specification, rows):
            if specification["geometry"] == "adaptive_sqrt":
                return _adaptive_smooth_neighbors(rows)
            return min(_SMOOTH_FIXED_NEIGHBORS, max(int(rows) - 1, 1))

        groups = self._validation_groups()
        try:
            if precomp is not None:  # reuse the shared deployed-config OOF (booster scores + folds)
                splits = precomp["splits"]
                oob = self._oof_probability_base(precomp)
                fold_models = tuple(precomp.get("models", ()))
                fold_feature_weights = [
                    _booster_importance(fold_models[index], X.shape[1])
                    if index < len(fold_models)
                    else np.asarray(fw, dtype=float)
                    for index in range(len(splits))
                ]
                if len(classes) == 2 and len(fold_models) >= len(splits):
                    effective_dimensions = [
                        _smooth_effective_dimensions(weights) for weights in fold_feature_weights
                    ]
                    if all(_binary_smooth_metric_is_diffuse(weights) for weights in fold_feature_weights):
                        self.smooth_memory_report_ = [
                            {
                                "name": "smooth_consensus",
                                "selected": False,
                                "reason": "diffuse_distance_metric",
                                "feature_count": int(X.shape[1]),
                                "fold_effective_dimensions": [float(value) for value in effective_dimensions],
                                "maximum_effective_dimensions": (_BINARY_SMOOTH_MAX_EFFECTIVE_DIMENSIONS),
                                "maximum_effective_fraction": (_BINARY_SMOOTH_MAX_EFFECTIVE_FRACTION),
                            }
                        ]
                        return 0.0
                for index, (tri, vai) in enumerate(splits):
                    local_fw = fold_feature_weights[index]
                    for specification in specifications:
                        geometry = specification["geometry"]
                        memories[geometry][vai] = _SmoothKNN(
                            X[tri],
                            y[tri],
                            classes,
                            fw=local_fw,
                            k=local_neighbors(specification, len(tri)),
                        ).proba(X[vai])
            else:
                kw = {
                    k: v
                    for k, v in dict(self._cfg).items()
                    if k
                    in {
                        "rounds",
                        "lr",
                        "depth",
                        "leaf",
                        "holdout",
                        "patience",
                        "linear_leaf",
                        "class_weight",
                        "validation_metric",
                        "base_feature_count",
                        "residual_stumps",
                        "max_leaves",
                        "best_first_pair",
                        "adaptive_best_first_pair",
                        "allowed",
                    }
                    and v is not None
                }
                kw["refit"] = False  # keep the DEPLOYED round budget so the OOF booster reflects the real one
                splits = self._validation_splits(
                    yi,
                    folds=3,
                    classification=True,
                    groups=groups,
                )
                for tri, vai in splits:
                    m = self._fit_certified(
                        self._classifier(**kw),
                        X[tri],
                        y[tri],
                        groups=(None if groups is None else groups[tri]),
                    )
                    F = m._scores(X[vai])
                    F = F / getattr(self, "_temp", 1.0)
                    e = np.exp(F - F.max(1, keepdims=True))
                    pb = e / e.sum(1, keepdims=True)
                    for j, c in enumerate(m.classes_):
                        oob[vai, ci[c]] = pb[:, j]
                    local_fw = _booster_importance(m, X.shape[1])
                    for specification in specifications:
                        geometry = specification["geometry"]
                        memories[geometry][vai] = _SmoothKNN(
                            X[tri],
                            y[tri],
                            classes,
                            fw=local_fw,
                            k=local_neighbors(specification, len(tri)),
                        ).proba(X[vai])
        except Exception:
            return 0.0  # if the OOF can't be built, don't risk the blend
        evidence_rows = (
            np.asarray(precomp.get("evidence_rows"), dtype=int)
            if precomp is not None and precomp.get("evidence_rows") is not None
            else np.unique(np.concatenate([valid for _train, valid in splits]))
        )
        if len(classes) == 2:
            # Tiny rank changes are calibration noise. A material gain is
            # required so the local member cannot silently lower ROC-AUC on
            # wide binary-category tables where the booster is already strong.
            specification = specifications[0]
            base_auc = float(roc_auc_score(yi[evidence_rows], oob[evidence_rows, 1]))
            blend = _preserve_certified_class(
                oob,
                (1.0 - specification["weight"]) * oob
                + specification["weight"] * memories[specification["geometry"]],
            )
            blend_auc = float(roc_auc_score(yi[evidence_rows], blend[evidence_rows, 1]))
            base_fold = np.array([float(roc_auc_score(yi[vai], oob[vai, 1])) for _tri, vai in splits])
            fold_delta = np.array(
                [
                    float(roc_auc_score(yi[vai], blend[vai, 1])) - base_fold[index]
                    for index, (_tri, vai) in enumerate(splits)
                ]
            )
            selected = blend_auc > base_auc + 0.001
            self.smooth_memory_report_ = [
                {"name": "certified_boost", "oof_rank_auc": base_auc, "selected": not selected},
                {
                    "name": "smooth_consensus",
                    "oof_rank_auc": blend_auc,
                    "fold_auc_delta": [float(value) for value in fold_delta],
                    "weight": specification["weight"],
                    "geometry": specification["geometry"],
                    "neighbors": specification["neighbors"],
                    "selected": selected,
                },
            ]
            if selected:
                self._smooth_oof_proba = blend
            return specification["weight"] if selected else 0.0

        base_score = _classification_rank_score(yi[evidence_rows], oob[evidence_rows])
        evaluations = []
        for specification in specifications:
            geometry = specification["geometry"]
            weight = specification["weight"]
            blend = _preserve_certified_class(
                oob,
                (1.0 - weight) * oob + weight * memories[geometry],
            )
            blend_score = _classification_rank_score(yi[evidence_rows], blend[evidence_rows])
            fold_delta = np.asarray(
                [
                    _classification_rank_score(yi[valid], blend[valid])
                    - _classification_rank_score(yi[valid], oob[valid])
                    for _train, valid in splits
                ],
                dtype=float,
            )
            evaluations.append(
                {
                    **specification,
                    "fold_neighbors": [
                        local_neighbors(specification, len(train)) for train, _valid in splits
                    ],
                    "oof_rank_auc": float(blend_score),
                    "rank_auc_delta": float(blend_score - base_score),
                    "fold_auc_delta": [float(value) for value in fold_delta],
                    "accepted": bool(
                        blend_score >= base_score + _MULTICLASS_SMOOTH_MIN_RANK_GAIN
                        and np.all(fold_delta >= -_MULTICLASS_SMOOTH_MAX_FOLD_LOSS)
                    ),
                    "blend": blend,
                }
            )
        admitted = [evaluation for evaluation in evaluations if evaluation["accepted"]]
        pool = admitted or evaluations
        winner = max(
            pool,
            key=lambda evaluation: (
                evaluation["oof_rank_auc"],
                min(evaluation["fold_auc_delta"]),
                evaluation["geometry"] == "fixed",
            ),
        )
        selected = bool(admitted)
        public_evaluations = [
            {key: value for key, value in evaluation.items() if key != "blend"} for evaluation in evaluations
        ]
        self.smooth_memory_report_ = [
            {
                "name": "certified_boost",
                "oof_rank_auc": float(base_score),
                "selected": not selected,
            },
            {
                "name": "smooth_consensus",
                "oof_rank_auc": winner["oof_rank_auc"],
                "fold_auc_delta": winner["fold_auc_delta"],
                "weight": winner["weight"],
                "geometry": winner["geometry"],
                "neighbors": winner["neighbors"],
                "fold_neighbors": winner["fold_neighbors"],
                "candidates": public_evaluations,
                "minimum_rank_gain": _MULTICLASS_SMOOTH_MIN_RANK_GAIN,
                "maximum_fold_loss": _MULTICLASS_SMOOTH_MAX_FOLD_LOSS,
                "selected": selected,
            },
        ]
        if selected:
            self._smooth_k = int(winner["neighbors"])
            self._smooth_geometry = winner["geometry"]
            self._smooth_oof_proba = winner["blend"]
        return float(winner["weight"]) if selected else 0.0

    def _category_memory_gate(self, X, y, precomp):
        """Admit exact categorical evidence only after a strict OOF rank win.

        The member is deliberately limited to small, category-rich tables. It
        is a challenger to the calibrated deployed OOF prediction (including a
        selected smooth member), not a replacement for the certified booster.
        Every candidate blend preserves the booster class before being scored.
        """
        groups = _onehot_groups(self._prep)
        self._category_memory_oof_proba = None
        self.category_memory_report_ = [{"name": "categorical_evidence", "selected": False}]
        if (
            precomp is None
            or self._has_text()
            or len(groups) < 4
            or len(X) < 200
            or len(X) > _CATEGORY_MEMORY_MAX_N
        ):
            return 0.0
        classes = np.asarray(self._pred.classes_)
        ci = {c: i for i, c in enumerate(classes)}
        yi = np.array([ci[v] for v in np.asarray(y)])
        splits = precomp["splits"]
        evidence_rows = np.asarray(
            precomp.get("evidence_rows", np.arange(len(yi))),
            dtype=int,
        )
        if not splits or np.unique(yi, return_counts=True)[1].min() < len(splits):
            return 0.0
        base = self._oof_probability_base(precomp)
        if getattr(self, "_smooth_oof_proba", None) is not None:
            base = self._smooth_oof_proba
        try:
            memory = np.zeros_like(base)
            params = []
            for tri, vai in splits:
                member = _CategoricalEvidenceMemory(X[tri], y[tri], classes, groups, seed=self.seed)
                memory[vai] = member.proba(X[vai])
                params.append({"k": member.k, "temperature": round(member.temp, 6)})
            base_score = _classification_rank_score(yi[evidence_rows], base[evidence_rows])
            base_fold = np.array([_classification_rank_score(yi[vai], base[vai]) for _tri, vai in splits])
            best = (base_score, 0.0, np.zeros(len(splits)), base)
            for weight in (0.1, 0.2, 0.3, 0.4, 0.5):
                blended = _preserve_certified_class(base, (1.0 - weight) * base + weight * memory)
                score = _classification_rank_score(yi[evidence_rows], blended[evidence_rows])
                fold_delta = np.array(
                    [
                        _classification_rank_score(yi[vai], blended[vai]) - base_fold[i]
                        for i, (_tri, vai) in enumerate(splits)
                    ]
                )
                if score > best[0]:
                    best = (score, weight, fold_delta, blended)
        except Exception:
            return 0.0
        selected = bool(best[1] > 0.0 and best[0] > base_score + 0.003 and np.all(best[2] > 0.0))
        self.category_memory_report_ = [
            {"name": "certified_boost", "oof_rank_auc": float(base_score), "selected": not selected},
            {
                "name": "categorical_evidence",
                "oof_rank_auc": float(best[0]),
                "fold_auc_delta": [float(value) for value in best[2]],
                "weight": float(best[1]),
                "parameters": params,
                "selected": selected,
            },
        ]
        if selected:
            self._category_memory_oof_proba = best[3]
        return float(best[1]) if selected else 0.0

    def _category_posterior_gate(self, X, y, precomp):
        """Admit categorical evidence with the authority its OOF result earned.

        Raw posterior updates may change the certified class only under the
        existing paired-accuracy rule.  A raw update that cannot earn that
        authority is also evaluated after projection back into the booster's
        class.  This second path may refine ranking, but only after a material
        rank-AUC gain on every fold and the same log-loss safeguard.  Both paths
        reuse the deployed-config OOF models and the same finite weight search.
        """
        self._category_posterior_oof_proba = None
        self._category_posterior_permission = None
        self._category_posterior_aggregation = None
        self._category_posterior_smoothing = None
        self.category_posterior_report_ = [
            gate_report(
                "categorical_posterior",
                False,
                stage="posterior_challenger",
                reason="not_evaluated",
            )
        ]
        groups, metadata = _onehot_group_metadata(self._prep)
        if precomp is None:
            self.category_posterior_report_[0]["reason"] = "shared_oof_unavailable"
            return 0.0
        if self._has_text() or len(groups) < 2 or len(X) < 200:
            self.category_posterior_report_[0]["reason"] = "ineligible_schema_or_evidence"
            return 0.0
        if len(X) > _CATEGORY_POSTERIOR_MAX_N:
            self.category_posterior_report_[0]["reason"] = "outside_bounded_oof_regime"
            return 0.0

        classes = np.asarray(self._pred.classes_)
        class_index = {value: index for index, value in enumerate(classes)}
        yi = np.asarray([class_index[value] for value in np.asarray(y)], dtype=np.int32)
        splits = precomp.get("splits", [])
        evidence_rows = np.asarray(
            precomp.get("evidence_rows", np.arange(len(yi))),
            dtype=int,
        )
        if not splits or np.unique(yi, return_counts=True)[1].min() < len(splits):
            self.category_posterior_report_[0]["reason"] = "insufficient_fold_class_support"
            return 0.0

        base = self._oof_probability_base(precomp)
        if getattr(self, "_smooth_oof_proba", None) is not None:
            base = self._smooth_oof_proba
        if getattr(self, "_category_memory_oof_proba", None) is not None:
            base = self._category_memory_oof_proba
        if getattr(self, "_proof_path_oof_proba", None) is not None:
            base = self._proof_path_oof_proba

        candidates = {
            (smoothing, aggregation, weight): base.copy()
            for smoothing in CategoricalPosteriorChallenger.SMOOTHING
            for aggregation in CategoricalPosteriorChallenger.AGGREGATIONS
            for weight in _CATEGORY_POSTERIOR_WEIGHTS
        }
        evidence_reports = []
        try:
            for train, valid in splits:
                challenger = CategoricalPosteriorChallenger(
                    X[train],
                    np.asarray(y)[train],
                    classes,
                    groups,
                    metadata=metadata,
                )
                posteriors = challenger.posterior_grid(X[valid])
                for (smoothing, aggregation), posterior in posteriors.items():
                    for weight in _CATEGORY_POSTERIOR_WEIGHTS:
                        candidates[(smoothing, aggregation, weight)][valid] = (
                            challenger.combine_from_posterior(
                                base[valid], posterior, challenger.prior, weight
                            )
                        )
                report = challenger.report()
                report["aggregations_evaluated"] = list(CategoricalPosteriorChallenger.AGGREGATIONS)
                report["smoothing_evaluated"] = (
                    list(CategoricalPosteriorChallenger.SMOOTHING)
                    if challenger.hierarchical_candidate
                    else ["global"]
                )
                evidence_reports.append(report)
        except (TypeError, ValueError, FloatingPointError):
            self.category_posterior_report_[0]["reason"] = "posterior_fit_failed_closed"
            return 0.0

        rows = evidence_rows

        def metrics(probability):
            predicted = probability.argmax(1)
            correct = predicted == yi
            accuracy = float(correct[rows].mean())
            log_loss = float(-np.log(np.clip(probability[rows, yi[rows]], 1e-300, 1.0)).mean())
            rank = _classification_rank_score(yi[rows], probability[rows])
            return predicted, correct, accuracy, log_loss, rank

        base_predicted, base_correct, base_accuracy, base_log_loss, base_rank = metrics(base)
        strict_fold_requirement = (len(splits) + 1) // 2
        log_loss_tolerance = 1.0 / np.sqrt(len(rows))
        evaluations = []
        class_change_accepted = []
        rank_only_accepted = []
        for (smoothing, aggregation, weight), probability in candidates.items():
            predicted, correct, accuracy, log_loss, rank = metrics(probability)
            projected = _preserve_certified_class(base, probability)
            (
                projected_predicted,
                _projected_correct,
                projected_accuracy,
                projected_log_loss,
                projected_rank,
            ) = metrics(projected)
            fold_net_wins = []
            fold_accuracy_delta = []
            fold_rank_delta = []
            for _train, valid in splits:
                gains = int(np.sum(correct[valid] & ~base_correct[valid]))
                losses = int(np.sum(~correct[valid] & base_correct[valid]))
                fold_net_wins.append(gains - losses)
                fold_accuracy_delta.append((gains - losses) / len(valid))
                fold_rank_delta.append(
                    _classification_rank_score(yi[valid], projected[valid])
                    - _classification_rank_score(yi[valid], base[valid])
                )
            wins = int(np.sum(correct[rows] & ~base_correct[rows]))
            losses = int(np.sum(~correct[rows] & base_correct[rows]))
            paired_z = (wins - losses) / np.sqrt(max(wins + losses, 1))
            transferable = bool(
                all(net >= 0 for net in fold_net_wins)
                and sum(net > 0 for net in fold_net_wins) >= strict_fold_requirement
            )
            class_selected = bool(
                transferable
                and wins > losses
                and accuracy > base_accuracy
                and paired_z >= _CATEGORY_POSTERIOR_MIN_PAIRED_Z
                and rank >= base_rank - _CATEGORY_POSTERIOR_MAX_RANK_REGRESSION
                and log_loss <= base_log_loss + log_loss_tolerance
            )
            rank_selected = bool(
                projected_rank >= base_rank + _CATEGORY_POSTERIOR_MIN_RANK_GAIN
                and all(delta >= _CATEGORY_POSTERIOR_MIN_FOLD_RANK_GAIN for delta in fold_rank_delta)
                and projected_log_loss <= base_log_loss + log_loss_tolerance
                and np.array_equal(projected_predicted, base_predicted)
            )
            evaluation = {
                "smoothing": smoothing,
                "aggregation": aggregation,
                "weight": float(weight),
                "oof_accuracy": accuracy,
                "accuracy_gain": accuracy - base_accuracy,
                "fold_accuracy_delta": [float(value) for value in fold_accuracy_delta],
                "fold_net_wins": fold_net_wins,
                "wins": wins,
                "losses": losses,
                "paired_z": float(paired_z),
                "override_rows": int(np.sum(predicted != base_predicted)),
                "oof_log_loss": log_loss,
                "log_loss_delta": log_loss - base_log_loss,
                "oof_rank_auc": rank,
                "rank_auc_delta": rank - base_rank,
                "accepted": class_selected,
                "class_change_accepted": class_selected,
                "rank_only_oof_accuracy": projected_accuracy,
                "rank_only_oof_log_loss": projected_log_loss,
                "rank_only_log_loss_delta": projected_log_loss - base_log_loss,
                "rank_only_oof_rank_auc": projected_rank,
                "rank_only_rank_auc_delta": projected_rank - base_rank,
                "fold_rank_auc_delta": [float(value) for value in fold_rank_delta],
                "rank_only_accepted": rank_selected,
            }
            evaluations.append(evaluation)
            if class_selected:
                class_change_accepted.append(
                    (
                        accuracy,
                        -log_loss,
                        rank,
                        smoothing == "global",
                        aggregation == "strongest",
                        -weight,
                        probability,
                        evaluation,
                    )
                )
            if rank_selected:
                rank_only_accepted.append(
                    (
                        projected_rank,
                        -projected_log_loss,
                        smoothing == "global",
                        aggregation == "strongest",
                        -weight,
                        projected,
                        evaluation,
                    )
                )

        permission = None
        selected_probability = None
        selected_weight = None
        selected_aggregation = None
        selected_smoothing = None
        selected_score = None
        selected_metric = "accuracy_or_rank_auc"
        reason = "no_transferable_accuracy_or_rank_gain"
        if class_change_accepted:
            best = max(class_change_accepted, key=lambda item: item[:6])
            selected_score, selected_weight = best[0], float(-best[5])
            selected_probability = best[6]
            selected_smoothing = best[7]["smoothing"]
            selected_aggregation = best[7]["aggregation"]
            selected_metric = "accuracy"
            permission = "class_change"
            reason = "consistent_oof_accuracy_gain"
        elif rank_only_accepted:
            best = max(rank_only_accepted, key=lambda item: item[:5])
            selected_score, selected_weight = best[0], float(-best[4])
            selected_probability = best[5]
            selected_smoothing = best[6]["smoothing"]
            selected_aggregation = best[6]["aggregation"]
            selected_metric = "rank_auc"
            permission = "rank_only"
            reason = "consistent_oof_rank_gain"
        selected = permission is not None
        self.category_posterior_report_ = [
            gate_report(
                "certified_probability_stack",
                not selected,
                stage="posterior_challenger",
                metric="accuracy",
                mean_score=base_accuracy,
                oof_log_loss=base_log_loss,
                oof_rank_auc=base_rank,
            ),
            gate_report(
                "categorical_posterior",
                selected,
                stage="posterior_challenger",
                metric=selected_metric,
                mean_score=(
                    selected_score
                    if selected_score is not None
                    else max(
                        max(item["oof_accuracy"], item["rank_only_oof_rank_auc"]) for item in evaluations
                    )
                ),
                weight=selected_weight,
                permission=permission,
                aggregation=selected_aggregation,
                smoothing=selected_smoothing,
                candidates=evaluations,
                evidence=evidence_reports,
                log_loss_tolerance=log_loss_tolerance,
                minimum_rank_gain=_CATEGORY_POSTERIOR_MIN_RANK_GAIN,
                minimum_fold_rank_gain=_CATEGORY_POSTERIOR_MIN_FOLD_RANK_GAIN,
                minimum_paired_z=_CATEGORY_POSTERIOR_MIN_PAIRED_Z,
                maximum_rank_regression=_CATEGORY_POSTERIOR_MAX_RANK_REGRESSION,
                reason=reason,
            ),
        ]
        if not selected:
            return 0.0
        self._category_posterior_permission = permission
        self._category_posterior_aggregation = selected_aggregation
        self._category_posterior_smoothing = selected_smoothing
        self._category_posterior_oof_proba = selected_probability
        return selected_weight

    def _numeric_interval_columns(self, X):
        """Continuous source columns eligible for finite interval evidence."""
        prep = getattr(self, "_prep", None)
        if prep is not None:
            datetime_names = [
                name
                for column in getattr(prep, "datetime_cols", [])
                for name in prep.datetime_feat[column].feature_names(column)
            ]
            names = tuple(str(name) for name in list(prep.num_cols) + datetime_names)
            columns = tuple(range(len(names)))
            return columns, names
        width = min(int(getattr(self, "n_input_features_", X.shape[1])), X.shape[1])
        declared = self.feature_names_ or [f"feature[{column}]" for column in range(width)]
        return tuple(range(width)), tuple(str(declared[column]) for column in range(width))

    def _numeric_interval_gate(self, X, y, precomp):
        """Select interval decisions, then independently test their probability rank.

        Accuracy selects exactly one candidate using paired decision evidence
        and adjacent-discount support. Only that preselected candidate may then
        challenge the public ranking surface, which requires a material pooled
        gain and a positive gain on every fold. A failed rank challenge retains
        the existing decision-only behavior.
        """
        self._numeric_interval_permission = None
        self._numeric_interval_oof_proba = None
        self._numeric_interval_oof_labels = None
        self._numeric_interval_aggregation = None
        self._numeric_interval_smoothing = None
        self.numeric_interval_report_ = [
            gate_report(
                "numeric_interval_decision",
                False,
                stage="decision_challenger",
                reason="not_evaluated",
            )
        ]
        columns, names = self._numeric_interval_columns(X)
        if precomp is None:
            self.numeric_interval_report_[0]["reason"] = "shared_oof_unavailable"
            return 0.0
        if self._has_text() or len(columns) < 2 or len(X) < 200:
            self.numeric_interval_report_[0]["reason"] = "ineligible_schema_or_evidence"
            return 0.0
        if len(X) > _NUMERIC_INTERVAL_MAX_N:
            self.numeric_interval_report_[0]["reason"] = "outside_bounded_oof_regime"
            return 0.0

        classes = np.asarray(self._pred.classes_)
        class_index = {value: index for index, value in enumerate(classes)}
        yi = np.asarray([class_index[value] for value in np.asarray(y)], dtype=np.int32)
        splits = precomp.get("splits", [])
        evidence_rows = np.asarray(
            precomp.get("evidence_rows", np.arange(len(yi))),
            dtype=int,
        )
        if not splits or np.unique(yi, return_counts=True)[1].min() < len(splits):
            self.numeric_interval_report_[0]["reason"] = "insufficient_fold_class_support"
            return 0.0

        base = self._oof_probability_base(precomp)
        for attr in (
            "_smooth_oof_proba",
            "_category_memory_oof_proba",
            "_proof_path_oof_proba",
            "_category_posterior_oof_proba",
        ):
            probability = getattr(self, attr, None)
            if probability is not None:
                base = probability

        candidates = {
            (smoothing, aggregation, weight): base.copy()
            for smoothing in (_NUMERIC_INTERVAL_SMOOTHING,)
            for aggregation in NumericIntervalPosteriorChallenger.AGGREGATIONS
            for weight in _CATEGORY_POSTERIOR_WEIGHTS
        }
        evidence_reports = []
        try:
            for train, valid in splits:
                challenger = NumericIntervalPosteriorChallenger(
                    X[train],
                    np.asarray(y)[train],
                    classes,
                    columns,
                    names=names,
                    smoothing=_NUMERIC_INTERVAL_SMOOTHING,
                )
                posteriors, triple_posterior = challenger.posterior_modes_with_triple(X[valid])
                smoothing = challenger.smoothing
                for aggregation in NumericIntervalPosteriorChallenger.AGGREGATIONS:
                    for weight in _CATEGORY_POSTERIOR_WEIGHTS:
                        key = (smoothing, aggregation, weight)
                        if aggregation == challenger.TRIPLE_FALLBACK:
                            candidates[key][valid] = challenger.combine_with_triple_fallback_from_posteriors(
                                base[valid],
                                posteriors["strongest"],
                                triple_posterior,
                                challenger.prior,
                                weight,
                            )
                        else:
                            candidates[key][valid] = challenger.combine_from_posterior(
                                base[valid],
                                posteriors[aggregation],
                                challenger.prior,
                                weight,
                            )
                evidence_reports.append(challenger.report())
        except (TypeError, ValueError, FloatingPointError):
            self.numeric_interval_report_[0]["reason"] = "interval_fit_failed_closed"
            return 0.0

        base_predicted = base.argmax(1)
        base_correct = base_predicted == yi
        base_accuracy = float(base_correct[evidence_rows].mean())
        strict_fold_requirement = (len(splits) + 1) // 2
        evaluations = []
        probability_by_key = {}
        for (smoothing, aggregation, weight), probability in candidates.items():
            predicted = probability.argmax(1)
            correct = predicted == yi
            accuracy = float(correct[evidence_rows].mean())
            fold_net_wins = []
            fold_accuracy_delta = []
            for _train, valid in splits:
                gains = int(np.sum(correct[valid] & ~base_correct[valid]))
                losses = int(np.sum(~correct[valid] & base_correct[valid]))
                fold_net_wins.append(gains - losses)
                fold_accuracy_delta.append((gains - losses) / len(valid))
            wins = int(np.sum(correct[evidence_rows] & ~base_correct[evidence_rows]))
            losses = int(np.sum(~correct[evidence_rows] & base_correct[evidence_rows]))
            paired_z = (wins - losses) / np.sqrt(max(wins + losses, 1))
            majority_wins = sum(net > 0 for net in fold_net_wins) >= strict_fold_requirement
            evaluation = {
                "smoothing": smoothing,
                "aggregation": aggregation,
                "weight": float(weight),
                "oof_accuracy": accuracy,
                "accuracy_gain": accuracy - base_accuracy,
                "fold_accuracy_delta": [float(value) for value in fold_accuracy_delta],
                "fold_net_wins": fold_net_wins,
                "wins": wins,
                "losses": losses,
                "paired_z": float(paired_z),
                "override_rows": int(np.sum(predicted != base_predicted)),
                "majority_fold_wins": majority_wins,
            }
            evaluations.append(evaluation)
            probability_by_key[(smoothing, aggregation, float(weight))] = probability

        accepted = []
        weight_order = list(_CATEGORY_POSTERIOR_WEIGHTS)
        for evaluation in evaluations:
            mode = [
                candidate
                for candidate in evaluations
                if candidate["smoothing"] == evaluation["smoothing"]
                and candidate["aggregation"] == evaluation["aggregation"]
            ]
            broad_support = [
                candidate
                for candidate in mode
                if candidate["accuracy_gain"] >= _NUMERIC_INTERVAL_MIN_ACCURACY_GAIN
                and candidate["majority_fold_wins"]
                and candidate["paired_z"] > 0.0
                and min(candidate["fold_accuracy_delta"]) >= -_NUMERIC_INTERVAL_SUPPORT_MAX_FOLD_LOSS
            ]
            index = weight_order.index(evaluation["weight"])
            adjacent_support = any(
                abs(weight_order.index(candidate["weight"]) - index) == 1
                for candidate in broad_support
                if candidate is not evaluation
            )
            evaluation["supporting_weights"] = [candidate["weight"] for candidate in broad_support]
            selected = bool(
                evaluation["accuracy_gain"] >= _NUMERIC_INTERVAL_MIN_ACCURACY_GAIN
                and evaluation["majority_fold_wins"]
                and min(evaluation["fold_accuracy_delta"]) >= -_NUMERIC_INTERVAL_MAX_FOLD_LOSS
                and evaluation["paired_z"] >= _NUMERIC_INTERVAL_MIN_PAIRED_Z
                and len(broad_support) >= _NUMERIC_INTERVAL_MIN_SUPPORTING_WEIGHTS
                and adjacent_support
            )
            evaluation["accepted"] = selected
            if selected:
                key = (
                    evaluation["smoothing"],
                    evaluation["aggregation"],
                    evaluation["weight"],
                )
                accepted.append((evaluation, probability_by_key[key]))

        selected_evaluation = None
        selected_probability = None
        if accepted:
            selected_evaluation, selected_probability = max(
                accepted,
                key=lambda item: (
                    item[0]["oof_accuracy"],
                    item[0]["paired_z"],
                    min(item[0]["fold_accuracy_delta"]),
                    item[0]["smoothing"] == "global",
                    item[0]["aggregation"] == "strongest",
                    -item[0]["weight"],
                ),
            )
        selected = selected_evaluation is not None
        rank_selected = False
        base_rank = candidate_rank = None
        fold_rank_delta = []
        if selected:
            try:
                base_rank = _classification_rank_score(yi[evidence_rows], base[evidence_rows])
                candidate_rank = _classification_rank_score(
                    yi[evidence_rows],
                    selected_probability[evidence_rows],
                )
                fold_rank_delta = [
                    _classification_rank_score(yi[valid], selected_probability[valid])
                    - _classification_rank_score(yi[valid], base[valid])
                    for _train, valid in splits
                ]
                rank_selected = bool(
                    candidate_rank - base_rank >= _NUMERIC_INTERVAL_MIN_RANK_GAIN
                    and np.all(
                        np.asarray(fold_rank_delta, dtype=float) >= _NUMERIC_INTERVAL_MIN_FOLD_RANK_GAIN
                    )
                )
            except (TypeError, ValueError, FloatingPointError):
                # Decision authority was already established independently;
                # rank scoring must fail closed without revoking that result.
                base_rank = candidate_rank = None
                fold_rank_delta = []
            selected_evaluation.update(
                rank_auc=candidate_rank,
                rank_auc_gain=(None if candidate_rank is None else candidate_rank - base_rank),
                fold_rank_auc_delta=[float(value) for value in fold_rank_delta],
                rank_selected=rank_selected,
            )
        reason = "consistent_oof_decision_gain" if selected else "no_transferable_decision_gain"
        self.numeric_interval_report_ = [
            gate_report(
                "probability_argmax",
                not selected,
                stage="decision_challenger",
                metric="accuracy",
                mean_score=base_accuracy,
            ),
            gate_report(
                "numeric_interval_decision",
                selected,
                stage="decision_challenger",
                metric="accuracy",
                mean_score=(
                    selected_evaluation["oof_accuracy"]
                    if selected
                    else max(candidate["oof_accuracy"] for candidate in evaluations)
                ),
                permission=(
                    "decision_and_rank" if rank_selected else ("decision_only" if selected else None)
                ),
                weight=selected_evaluation["weight"] if selected else None,
                aggregation=selected_evaluation["aggregation"] if selected else None,
                smoothing=selected_evaluation["smoothing"] if selected else None,
                candidates=evaluations,
                evidence=evidence_reports,
                minimum_accuracy_gain=_NUMERIC_INTERVAL_MIN_ACCURACY_GAIN,
                minimum_paired_z=_NUMERIC_INTERVAL_MIN_PAIRED_Z,
                maximum_fold_loss=_NUMERIC_INTERVAL_MAX_FOLD_LOSS,
                minimum_supporting_weights=_NUMERIC_INTERVAL_MIN_SUPPORTING_WEIGHTS,
                base_rank_auc=base_rank,
                rank_auc=candidate_rank,
                rank_auc_delta=(None if candidate_rank is None else candidate_rank - base_rank),
                fold_rank_auc_delta=[float(value) for value in fold_rank_delta],
                minimum_rank_gain=_NUMERIC_INTERVAL_MIN_RANK_GAIN,
                minimum_fold_rank_gain=_NUMERIC_INTERVAL_MIN_FOLD_RANK_GAIN,
                probability_surface=("numeric_interval_rank" if rank_selected else "unchanged"),
                reason=reason,
            ),
        ]
        if not selected:
            return 0.0
        self._numeric_interval_aggregation = selected_evaluation["aggregation"]
        self._numeric_interval_smoothing = selected_evaluation["smoothing"]
        self._numeric_interval_permission = "decision_and_rank" if rank_selected else "decision_only"
        self._numeric_interval_oof_proba = selected_probability if rank_selected else None
        self._numeric_interval_oof_labels = classes[selected_probability.argmax(1)]
        return float(selected_evaluation["weight"])

    def _refresh_classification_decision_confidence(self, X, y, oof_labels):
        """Recalibrate precision regions on decision labels, not hidden probabilities."""
        from tabpvn.certified_confidence import CertifiedClassConfidence

        rows = getattr(self, "_validation_evidence_rows", None)
        rows = np.arange(len(y)) if rows is None else np.asarray(rows, dtype=int)
        self._conf = CertifiedClassConfidence(seed=self.seed).fit(
            X[rows], np.asarray(y)[rows], np.asarray(oof_labels)[rows]
        )

    def _refresh_classification_confidence(self, X, y, oof_probability):
        """Recalibrate guarantees and operating points after a class challenger."""
        from tabpvn.certified_confidence import CertifiedClassConfidence

        probability = np.asarray(oof_probability, float)
        classes = np.asarray(self._pred.classes_)
        labels = classes[probability.argmax(1)]
        rows = getattr(self, "_validation_evidence_rows", None)
        rows = np.arange(len(y)) if rows is None else np.asarray(rows, dtype=int)
        target = np.asarray(y)[rows]
        selected_probability = probability[rows]
        self._conf = CertifiedClassConfidence(seed=self.seed).fit(X[rows], target, labels[rows])
        yi = np.asarray([np.flatnonzero(classes == value)[0] for value in target])
        self._cal_conf = (
            selected_probability.max(1),
            (selected_probability.argmax(1) == yi).astype(float),
        )
        if len(classes) == 2:
            self._bal_thr, self._rare_thr, _rare_report = _fit_binary_thresholds(
                selected_probability,
                target,
                classes,
                rare_class=(self.rare_class_ if self.rare_event_ else None),
                validation_groups=(None if self._fit_validation is None else self._validation_groups(rows)),
            )

    def _sdm_gate(self, X, y):
        """Leak-safe SELECTION+weight gate for the SDM-attention member (text classification): 3-fold OOF of
        the booster vs a booster+attention blend at candidate weights; return the weight that CLEARLY beats the
        booster out-of-fold, else 0 (so keyword-decisive text keeps the pure booster — zero-downside). Rows
        subsampled for gate cost; booster at the deployed config for a fair comparison."""
        cols = [j for j, nm in enumerate(self.feature_names_ or []) if "~" in str(nm)]
        if not cols:
            return 0.0
        classes = np.asarray(self._pred.classes_)
        ci = {c: i for i, c in enumerate(classes)}
        yi = np.array([ci[v] for v in y])
        Xg, yg, yig = X, np.asarray(y), yi
        groups = self._validation_groups()
        if len(yi) > 2500:  # a coarse accuracy statistic doesn't need every row
            selected, _weight = self._bounded_evidence_rows(
                y,
                2500,
                self.seed + 3,
                stratified=False,
            )
            sub = np.arange(len(yi)) if selected is None else selected
            Xg, yg, yig = X[sub], np.asarray(y)[sub], yi[sub]
            groups = None if groups is None else groups[sub]
        kw = {
            k: v
            for k, v in dict(self._cfg).items()
            if k
            in {
                "rounds",
                "lr",
                "depth",
                "leaf",
                "holdout",
                "patience",
                "linear_leaf",
                "class_weight",
                "rare_event",
                "rare_min_events",
                "min_verifier_events",
                "validation_metric",
                "base_feature_count",
                "residual_stumps",
                "max_leaves",
                "best_first_pair",
                "adaptive_best_first_pair",
                "allowed",
            }
            and v is not None
        }
        kw["refit"] = False
        oob = np.zeros((len(yig), len(classes)))
        oos = np.zeros((len(yig), len(classes)))
        try:
            splits = self._validation_splits(
                yig,
                folds=3,
                classification=True,
                groups=groups,
            )
            for tri, vai in splits:
                m = self._fit_certified(
                    self._classifier(**kw),
                    Xg[tri],
                    yg[tri],
                    groups=(None if groups is None else groups[tri]),
                )
                F = m._scores(Xg[vai])
                e = np.exp(F - F.max(1, keepdims=True))
                pb = e / e.sum(1, keepdims=True)
                for j, c in enumerate(m.classes_):
                    oob[vai, ci[c]] = pb[:, j]
                oos[vai] = _SDMAttention(Xg[tri], yg[tri], classes, cols, seed=self.seed).proba(Xg[vai])
        except Exception:
            return 0.0  # if the OOF can't be built, keep the pure booster
        evidence_rows = np.unique(np.concatenate([valid for _train, valid in splits]))
        acc_b = (oob[evidence_rows].argmax(1) == yig[evidence_rows]).mean()
        best_w, best_acc = 0.0, acc_b
        for w in (0.3, 0.5, 0.7):  # pick the blend weight that helps most out-of-fold
            acc = (
                ((1 - w) * oob[evidence_rows] + w * oos[evidence_rows]).argmax(1) == yig[evidence_rows]
            ).mean()
            if acc > best_acc + 1e-9:
                best_acc, best_w = acc, w
        # require a CLEAR OOF gain (≥0.01): marginal gains don't transfer to the test set and can regress a
        # near-ceiling booster (e.g. spam), so keep the pure booster unless the memory member clearly wins.
        return best_w if best_acc > acc_b + 0.01 else 0.0

    def _reg_member(self, X, y, cols):  # build a fresh regression aux member of the given kind
        return {
            "sdm": lambda k: _SDMAttention(X, y, None, cols, beta=20.0, regression=True, seed=self.seed),
            "linear": lambda k: _LinearReg(X, y, cols, seed=self.seed),
        }

    def _sdm_gate_reg(self, X, y):
        """Leak-safe gate for the REGRESSION aux member: 3-fold OOF R² of the booster vs a booster+member blend,
        for BOTH candidate members — SDM Nadaraya-Watson attention AND a ridge linear read (the linear model is
        what closes the big gap on rare-token regression like car→price, which the tree booster structurally
        can't). Returns (kind, weight) with a CLEAR OOF R² gain, else (None, 0) — zero-downside."""
        from sklearn.metrics import r2_score

        cols = [j for j, nm in enumerate(self.feature_names_ or []) if "~" in str(nm)]
        if not cols:
            return None, 0.0
        yv = np.asarray(y, float)
        Xg, yg = X, yv
        groups = self._validation_groups()
        if len(yv) > 2500:
            selected, _weight = self._bounded_evidence_rows(
                yv,
                2500,
                self.seed + 3,
                stratified=False,
            )
            sub = np.arange(len(yv)) if selected is None else selected
            Xg, yg = X[sub], yv[sub]
            groups = None if groups is None else groups[sub]
        accept = {
            "rounds",
            "lr",
            "depth",
            "leaf",
            "subsample",
            "colsample",
            "lam",
            "nbins",
            "huber",
            "holdout",
            "patience",
        }
        kw = {k: v for k, v in dict(self._cfg).items() if k in accept and v is not None}
        kw["refit"] = False
        ob = np.empty(len(yg))
        cand = {"sdm": np.empty(len(yg)), "linear": np.empty(len(yg))}
        try:
            splits = self._validation_splits(
                yg,
                folds=3,
                classification=False,
                groups=groups,
            )
            for trn, val in splits:
                ob[val] = self._fit_certified(
                    AdditiveCertifiedRegressor(seed=self.seed, **kw),
                    Xg[trn],
                    yg[trn],
                    groups=(None if groups is None else groups[trn]),
                ).predict(Xg[val])
                mk = self._reg_member(Xg[trn], yg[trn], cols)
                for kind in cand:
                    cand[kind][val] = mk[kind](kind).read(Xg[val])
        except Exception:
            return None, 0.0
        evidence_rows = np.unique(np.concatenate([valid for _train, valid in splits]))
        r_b = r2_score(yg[evidence_rows], ob[evidence_rows])
        best = (None, 0.0, r_b)
        for kind, candidate_output in cand.items():
            for w in (0.3, 0.5, 0.7):
                r = r2_score(
                    yg[evidence_rows],
                    (1 - w) * ob[evidence_rows] + w * candidate_output[evidence_rows],
                )
                if r > best[2]:
                    best = (kind, w, r)
        return (best[0], best[1]) if best[2] > r_b + 0.005 else (None, 0.0)  # clear OOF R² gain

    def _clf_oof(self, X, y, folds=3):
        """One StratifiedKFold OOF of the DEPLOYED-config classifier, SHARED by the confidence layer and the
        smooth gate (which otherwise each fit their own K-fold boosters). Returns per-fold raw scores
        (n×C, in `classes_` order), argmax predictions, splits, and the corresponding fold models. The
        proof-path gate reuses those models for certified-region routing rather than training another OOF
        ensemble. Leak-safe (every row predicted out-of-fold); deployed config + `refit=False` so it reflects
        the production model. This is exactly the OOF the smooth gate builds today, so its decision is unchanged."""
        classes = np.asarray(self._pred.classes_)
        ci = {c: i for i, c in enumerate(classes)}
        yi = np.array([ci[v] for v in y])
        kw = {
            k: v
            for k, v in dict(self._cfg).items()
            if k
            in {
                "rounds",
                "lr",
                "depth",
                "leaf",
                "holdout",
                "patience",
                "linear_leaf",
                "class_weight",
                "validation_metric",
                "base_feature_count",
                "residual_stumps",
                "max_leaves",
                "best_first_pair",
                "adaptive_best_first_pair",
                "allowed",
            }
            and v is not None
        }
        # A temporal fold has only one past/future boundary. Refit its model on
        # the entire past side after selecting the tree count; otherwise the
        # shared calibration clone can see barely half the available history
        # and no longer resemble the final all-history predictor under drift.
        kw["refit"] = self._fit_validation is not None
        scores = np.zeros((len(y), len(classes)))
        groups = self._validation_groups()
        calibration_holdout = min(0.25, max(0.10, 200.0 / len(y))) if groups is not None else 0.25
        splits = self._validation_splits(
            yi,
            folds=folds,
            classification=True,
            groups=groups,
            holdout=calibration_holdout,
        )
        models = []

        def fold_fit(tri, vai):
            def run():
                model = self._fit_certified(
                    self._classifier(**kw),
                    X[tri],
                    y[tri],
                    groups=(None if groups is None else groups[tri]),
                )
                return model, model._scores(X[vai])

            return run

        thunks = {index: fold_fit(tri, vai) for index, (tri, vai) in enumerate(splits)}
        # Every fold has fixed rows, seed, and configuration. Numeric tree
        # growth releases the GIL, so concurrent folds reduce wall time without
        # changing their models or deterministic output assembly. Affine leaves
        # do substantial Python/BLAS setup, however, and contend under threads;
        # retain the established serial order for that path.
        fitted = (
            {index: run() for index, run in thunks.items()} if kw.get("linear_leaf", False) else _pmap(thunks)
        )
        for index, (_tri, vai) in enumerate(splits):
            m, F = fitted[index]
            models.append(m)
            for j, c in enumerate(m.classes_):  # remap fold-class order -> global classes_ order
                scores[vai, ci[c]] = F[:, j]
        evidence_rows = np.unique(np.concatenate([valid for _train, valid in splits]))
        self._validation_evidence_rows = evidence_rows
        return {
            "splits": splits,
            "scores": scores,
            "models": models,
            "pred": classes[scores.argmax(1)],
            "evidence_rows": evidence_rows,
        }

    def _build_confidence(self, X, y, folds=2, precomp=None):  # noqa: C901 - calibration state machine
        """Auto-calibrate the certified-confidence layer from internal out-of-fold predictions (leak-safe).
        Uses a LIGHTER booster for the OOF (residual/label estimates for calibration don't need the peak
        config) so default fit stays fast. Skipped if certify=False or too few rows to calibrate.
        2-fold OOF is still leak-safe (every row gets an out-of-fold prediction) and empirically preserves
        the conformal coverage / precision guarantee at a third less calibration cost than 3-fold."""
        self._conf = None
        self._temp = 1.0
        self._bal_thr = None  # balanced-accuracy decision threshold (binary), tuned on OOF below
        self._rare_thr = None  # weighted-F1 threshold for the automatically detected rare class
        self._prior_rank_strength = 0.0
        self._prior_rank_oof_proba = None
        self.prior_rank_report_ = [
            gate_report(
                "multiclass_prior_rank_projection",
                False,
                stage="probability_rank",
                reason="calibration_evidence_unavailable",
            )
        ]
        self._cal_conf = (
            None  # leak-safe OOF (max calibrated proba, correct?) pairs for fair-price no-arbitrage
        )
        self._prior_train = (
            None  # training class base rate (over classes_) for Bayesian prior-shift correction
        )
        if self.mode == "classification":
            yy = np.asarray(y)
            self._prior_train = np.array([float((yy == c).mean()) for c in self._pred.classes_])
        if len(y) < 200:  # calibration always on (auto-skipped only when there are too few rows to calibrate)
            return
        if self.mode == "classification" and self.rare_event_:
            rare_count = int(np.sum(np.asarray(y) == self.rare_class_))
            if rare_count < folds:
                self.rare_event_report_.update(
                    {
                        "calibration_source": "insufficient_rare_evidence",
                        "calibration_rows": int(len(y)),
                        "calibration_events": rare_count,
                        "operating_point": None,
                    }
                )
                return

        def calibrate_classification(
            scores,
            labels,
            sample_weight,
            source,
            validation_groups=None,
        ):
            self._temp = _fit_temperature(scores, labels, self._pred.classes_, sample_weight=sample_weight)
            calibrated = scores / self._temp
            exp_scores = np.exp(calibrated - calibrated.max(1, keepdims=True))
            proba = exp_scores / exp_scores.sum(1, keepdims=True)
            if len(self._pred.classes_) == 2:
                self._bal_thr, self._rare_thr, rare_report = _fit_binary_thresholds(
                    proba,
                    labels,
                    self._pred.classes_,
                    rare_class=(self.rare_class_ if self.rare_event_ else None),
                    sample_weight=sample_weight,
                    validation_groups=validation_groups,
                )
                if validation_groups is not None and rare_report is not None:
                    self.validation_report_["threshold_validation"] = rare_report["validation_mode"]
                    self.validation_report_["threshold_validation_rows"] = rare_report["evaluation_rows"]
                if self.rare_event_ and self.rare_event_report_ is not None:
                    weights = (
                        np.ones(len(labels), dtype=float)
                        if sample_weight is None
                        else np.asarray(sample_weight, dtype=float)
                    )
                    effective_rows = float(weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12))
                    self.rare_event_report_.update(
                        {
                            "calibration_source": source,
                            "calibration_rows": int(len(labels)),
                            "calibration_events": int(np.sum(np.asarray(labels) == self.rare_class_)),
                            "weighted_calibration_rate": float(
                                np.average(np.asarray(labels) == self.rare_class_, weights=weights)
                            ),
                            "calibration_effective_rows": effective_rows,
                            "operating_point": rare_report,
                        }
                    )
            return proba

        try:
            # LEVER A: classification with an out-of-fit deploy holdout (refit=False deploy fit, i.e. large n) —
            # calibrate the precision layer on the deploy model's OWN leak-safe holdout residuals instead of
            # fitting a separate K-fold OOF ensemble. Same precision guarantee (measured), drops the OOF phase.
            ver = getattr(self._pred, "ver_", None) if self.mode == "classification" else None
            if ver is not None and len(ver) >= 200:
                from tabpvn.certified_confidence import CertifiedClassConfidence

                yv = np.asarray(y)
                ver = np.asarray(ver, dtype=np.int64)
                ver_weight = getattr(self._pred, "ver_weight_", None)
                if ver_weight is not None:
                    ver_weight = np.asarray(ver_weight, dtype=float)
                if (
                    len(ver) > _CONF_MAX_N
                ):  # a precision statistic needs a stable sample, not all holdout rows
                    if self.rare_event_:
                        positions, inclusion_weight = _fit_sample(
                            yv[ver],
                            _CONF_MAX_N,
                            self.seed + 12345,
                            stratified=True,
                            min_class_rows=_RARE_VERIFY_MIN_EVENTS,
                        )
                        ver = ver[positions]
                        ver_weight = (
                            inclusion_weight
                            if ver_weight is None
                            else ver_weight[positions] * inclusion_weight
                        )
                        ver_weight = ver_weight / np.mean(ver_weight)
                    else:
                        positions = np.random.default_rng(self.seed + 12345).choice(
                            len(ver), _CONF_MAX_N, replace=False
                        )
                        ver = ver[positions]
                        if ver_weight is not None:
                            ver_weight = ver_weight[positions]
                Xh, yh = X[ver], yv[ver]
                scores = self._pred._scores(Xh)
                self._conf = CertifiedClassConfidence(seed=self.seed).fit(
                    Xh, yh, self._pred.predict(Xh), sample_weight=ver_weight
                )
                threshold_groups = None if self._fit_validation is None else self._validation_groups(ver)
                calibrate_classification(
                    scores,
                    yh,
                    ver_weight,
                    "deploy_verifier",
                    threshold_groups,
                )
                if self._fit_validation is not None:
                    self.validation_report_["confidence_rows"] = int(len(yh))
                    self.validation_report_["confidence_source"] = "deploy_future_verifier"
                return
            calibration_weight = None
            calibration_groups = self._validation_groups()
            if len(y) > _CONF_MAX_N:  # OOF path: calibrate on a representative subsample at scale
                if self._fit_validation is not None:
                    sub = self._fit_validation.bounded_rows(_CONF_MAX_N)
                elif self.mode == "classification" and self.rare_event_:
                    sub, calibration_weight = _fit_sample(
                        y,
                        _CONF_MAX_N,
                        self.seed + 12345,
                        stratified=True,
                        min_class_rows=_RARE_VERIFY_MIN_EVENTS,
                    )
                else:
                    sub = np.random.default_rng(self.seed + 12345).choice(len(y), _CONF_MAX_N, replace=False)
                X, y = X[sub], y[sub]  # a statistic, not a per-row fit — stops confidence scaling with n
                if calibration_groups is not None:
                    calibration_groups = calibration_groups[sub]
            oof = np.empty(len(y), dtype=(float if self.mode == "regression" else y.dtype))
            Model = AdditiveCertifiedRegressor if self.mode == "regression" else AdditiveCertifiedClassifier
            # Calibrate on the SAME self-tuned configuration that is deployed —
            # including linear leaves and class weighting — so every gate and
            # certified confidence statistic describes the production model.
            accept = (
                {
                    "rounds",
                    "lr",
                    "depth",
                    "leaf",
                    "subsample",
                    "colsample",
                    "lam",
                    "nbins",
                    "huber",
                    "holdout",
                    "patience",
                }
                if self.mode == "regression"
                else {
                    "rounds",
                    "lr",
                    "depth",
                    "leaf",
                    "holdout",
                    "patience",
                    "linear_leaf",
                    "class_weight",
                    "rare_event",
                    "rare_min_events",
                    "min_verifier_events",
                    "validation_metric",
                    "base_feature_count",
                    "residual_stumps",
                    "max_leaves",
                    "best_first_pair",
                    "adaptive_best_first_pair",
                    "allowed",
                }
            )
            kw = {k: v for k, v in dict(self._cfg).items() if k in accept and v is not None}
            kw["rounds"] = min(kw.get("rounds", 400), 400)
            # One temporal fold is refitted on its complete past side so its
            # future residuals describe the final all-history model. Ordinary
            # multi-fold OOF keeps the established faster no-refit path.
            kw["refit"] = self._fit_validation is not None
            oof_scores = None if self.mode == "regression" else np.zeros((len(y), len(np.unique(y))))
            splits = self._validation_splits(
                y,
                folds=folds,
                classification=self.mode == "classification",
                groups=calibration_groups,
                holdout=(min(0.25, max(0.10, 200.0 / len(y))) if calibration_groups is not None else 0.25),
            )

            def _fold(
                trn, val
            ):  # fit one OOF fold; independent + seeded, so folds run concurrently (bit-identical)
                def run():
                    mdl = self._fit_certified(
                        Model(seed=self.seed, **kw),
                        X[trn],
                        y[trn],
                        groups=(None if calibration_groups is None else calibration_groups[trn]),
                        sample_weight=(
                            calibration_weight[trn]
                            if self.mode == "classification" and calibration_weight is not None
                            else None
                        ),
                    )
                    pred = mdl.predict(X[val])
                    # if a regression SDM member is deployed, calibrate on the BLENDED OOF prediction so the
                    # conformal bound covers what predict() actually returns (soundness under blending).
                    if self.mode == "regression" and getattr(self, "_sdm", None) is not None:
                        sf = self._sdm.refit(X[trn], y[trn]).read(
                            X[val]
                        )  # SDM or linear member, same interface
                        pred = (1.0 - self._sdm_w) * pred + self._sdm_w * sf
                    sc = mdl._scores(X[val]) if self.mode == "classification" else None
                    return val, pred, sc

                return run

            if precomp is not None and self.mode == "classification":  # reuse the shared deployed-config OOF
                oof = precomp["pred"]  # (the smooth gate built it) -> no re-fit
                oof_scores = precomp["scores"]
                evidence_rows = np.asarray(precomp.get("evidence_rows", np.arange(len(y))), dtype=int)
            else:
                for val, pred, sc in _pmap(
                    {i: _fold(trn, va) for i, (trn, va) in enumerate(splits)}
                ).values():
                    oof[val] = pred
                    if sc is not None:
                        oof_scores[val] = sc
                evidence_rows = np.unique(np.concatenate([valid for _train, valid in splits]))
            if self._fit_validation is not None:
                self.validation_report_["confidence_rows"] = int(len(evidence_rows))
                self.validation_report_["confidence_source"] = "refitted_past_future_tail"
            evidence_weight = (
                None if calibration_weight is None else np.asarray(calibration_weight)[evidence_rows]
            )
            if self.mode == "regression":
                from tabpvn.certified_confidence import CertifiedConfidence

                self._conf = CertifiedConfidence(alpha=self.alpha, seed=self.seed).fit(
                    X[evidence_rows], y[evidence_rows].astype(float), oof[evidence_rows]
                )
            else:
                from tabpvn.certified_confidence import CertifiedClassConfidence

                self._conf = CertifiedClassConfidence(seed=self.seed).fit(
                    X[evidence_rows],
                    y[evidence_rows],
                    oof[evidence_rows],
                    sample_weight=evidence_weight,
                )
                Pcal = calibrate_classification(
                    oof_scores[evidence_rows],
                    y[evidence_rows],
                    evidence_weight,
                    "future_holdout" if self._fit_validation is not None else "out_of_fold",
                    (None if calibration_groups is None else calibration_groups[evidence_rows]),
                )
                ci = {c: i for i, c in enumerate(self._pred.classes_)}
                yi = np.array([ci[v] for v in y[evidence_rows]])
                # leak-safe calibration set for the fair-price layer: the deployed OOF calibrated confidence
                # (max proba) and whether that argmax was correct — the (price, outcome) pairs a no-arbitrage
                # check bets against, and what decide() thresholds at the fair strike.
                if calibration_weight is None:
                    self._cal_conf = (Pcal.max(1), (Pcal.argmax(1) == yi).astype(float))
        except Exception:
            self._conf = None  # never let calibration break fit
            self._temp = 1.0
            self._bal_thr = None
            self._rare_thr = None

    def _successive_halving(self, cands, base_idx, score_fn, rungs, maximize):
        """Hyperband-style SUCCESSIVE HALVING over the candidate configs. Evaluate EVERY candidate at a tiny
        budget (few rounds, small subsample, 1 fold), keep the top half, re-evaluate the survivors at a larger
        budget, and repeat — so only the finalists ever pay the full, faithful fit. Cheap rungs screen out the
        obviously-worse configs (the deep/overfitting corners die immediately), which is what lets the FINAL
        rung afford a larger, in-regime subsample without the old big/small special-case. BASE is always
        promoted so the final hysteresis can compare it fairly. Deterministic. Returns (best_idx, final_scores)
        where final_scores holds each finalist's score at the highest rung it reached."""
        survivors = list(range(len(cands)))
        scores = {}
        for ri, rung in enumerate(rungs):
            # score_fn does its (sequential, order-preserving) random draw eagerly and returns a thunk; the
            # expensive fits then run concurrently — bit-identical to serial since the randomness is fixed.
            thunks = {i: score_fn(cands[i], rung) for i in survivors}
            cur = _pmap(thunks)
            scores.update(
                cur
            )  # survivors' scores overwrite with their higher-budget (more faithful) estimate
            if ri == len(rungs) - 1:
                break
            ranked = sorted(survivors, key=lambda i: cur[i], reverse=maximize)
            keep = max(1, (len(ranked) + 1) // 2)
            survivors = ranked[:keep]
            if base_idx not in survivors:
                survivors.append(base_idx)  # never prune BASE — it must reach the final rung
        best = (max if maximize else min)(survivors, key=lambda i: scores[i])
        return best, scores, survivors

    def _auto_rare_architecture(self, X, y):
        """Admit rare-event handling only after a paired rank-metric win.

        Prevalence opens the candidate set; it does not choose the architecture. A standard booster and a
        rare-aware booster fit the same bounded, prior-corrected training rows and are
        compared on one untouched stratified partition. This replaces the old discontinuous 5% switch while
        keeping the extra gate to two compact fits. Extremely small event sets retain rare mode because they
        cannot support an honest architecture comparison and need stratified verifier/reservoir handling.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        X, y = np.asarray(X, float), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2 or self.rare_class_ is None:
            return False
        rare_index = int(np.flatnonzero(classes == self.rare_class_)[0])
        event_count = int(counts[rare_index])
        source_rate = float(event_count / len(y))
        if event_count < _RARE_ARCH_GATE_MIN_EVENTS:
            self.rare_architecture_report_ = [
                gate_report(
                    "rare_aware_boost",
                    True,
                    stage="predictor",
                    reason="insufficient_events_for_paired_gate",
                    events=event_count,
                    source_rate=source_rate,
                )
            ]
            self.rare_event_report_["selection"] = "insufficient_events_for_paired_gate"
            return True

        minimum = min(
            _RARE_TUNE_MIN_EVENTS,
            event_count,
            _RARE_ARCH_GATE_MAX_ROWS // 4,
        )
        try:
            rows, inclusion_weight = self._bounded_evidence_rows(
                y,
                _RARE_ARCH_GATE_MAX_ROWS,
                self.seed + 313,
                stratified=True,
                min_class_rows=minimum,
            )
            Xg = X if rows is None else X[rows]
            yg = y if rows is None else y[rows]
            groups = self._validation_groups(rows)
            weights = (
                np.ones(len(yg), dtype=float)
                if inclusion_weight is None
                else np.asarray(inclusion_weight, dtype=float)
            )
            split_count = min(4, int(np.unique(yg, return_counts=True)[1].min()))
            train, valid = self._validation_splits(
                yg,
                folds=split_count,
                classification=True,
                groups=groups,
                single=True,
            )[0]
            standard_cfg = dict(
                rounds=240,
                lr=0.05,
                depth=6,
                leaf=int(np.clip(len(y) // 150, 20, 50)),
                patience=24,
                refit=False,
                validation_metric="logloss",
            )
            rare_cfg = dict(
                rounds=240,
                lr=0.05,
                depth=6,
                leaf=int(np.clip(event_count // 20, 5, 50)),
                patience=24,
                refit=False,
                validation_metric="logloss",
                rare_event=True,
                rare_min_events=min(_RARE_TUNE_MIN_EVENTS, event_count),
                min_verifier_events=min(
                    _RARE_VERIFY_MIN_EVENTS,
                    max(2, int(np.sum(yg[train] == self.rare_class_) // 4)),
                ),
            )

            def fit(config):
                return lambda: self._fit_certified(
                    self._classifier(**config),
                    Xg[train],
                    yg[train],
                    groups=(None if groups is None else groups[train]),
                    sample_weight=weights[train],
                )

            fitted = _pmap({"standard": fit(standard_cfg), "rare": fit(rare_cfg)})

            def score(model):
                raw = model._scores(Xg[valid])
                exp_score = np.exp(raw - raw.max(1, keepdims=True))
                probability = exp_score / exp_score.sum(1, keepdims=True)
                model_classes = np.asarray(model.classes_)
                column = int(np.flatnonzero(model_classes == self.rare_class_)[0])
                actual = yg[valid] == self.rare_class_
                return (
                    float(
                        roc_auc_score(
                            actual,
                            probability[:, column],
                            sample_weight=weights[valid],
                        )
                    ),
                    float(
                        average_precision_score(
                            actual,
                            probability[:, column],
                            sample_weight=weights[valid],
                        )
                    ),
                )

            standard_auc, standard_ap = score(fitted["standard"])
            rare_auc, rare_ap = score(fitted["rare"])
            auc_delta, ap_delta = rare_auc - standard_auc, rare_ap - standard_ap
            auc_led = (
                auc_delta >= _RARE_ARCH_GATE_MIN_AUC_GAIN and ap_delta >= -_RARE_ARCH_GATE_MAX_SECONDARY_LOSS
            )
            ap_led = (
                ap_delta >= _RARE_ARCH_GATE_MIN_AP_GAIN and auc_delta >= -_RARE_ARCH_GATE_MAX_SECONDARY_LOSS
            )
            selected = bool(auc_led or ap_led)
            self.rare_architecture_report_ = [
                gate_report(
                    "standard_certified_boost",
                    not selected,
                    stage="predictor",
                    holdout_auc=standard_auc,
                    holdout_average_precision=standard_ap,
                ),
                gate_report(
                    "rare_aware_boost",
                    selected,
                    stage="predictor",
                    holdout_auc=rare_auc,
                    holdout_average_precision=rare_ap,
                    auc_delta=auc_delta,
                    average_precision_delta=ap_delta,
                    evidence_rows=int(len(yg)),
                    evidence_events=int(np.sum(yg == self.rare_class_)),
                    source_rate=source_rate,
                ),
            ]
            self.rare_event_report_["selection"] = "paired_rank_holdout"
            self.rare_event_report_["architecture_candidates"] = self.rare_architecture_report_
            return selected
        except Exception as exc:
            self.rare_architecture_report_ = [
                gate_report(
                    "standard_certified_boost",
                    True,
                    stage="predictor",
                    reason="rare_gate_failed_closed",
                    error=type(exc).__name__,
                )
            ]
            self.rare_event_report_["selection"] = "gate_failed_closed"
            self.rare_event_report_["architecture_candidates"] = self.rare_architecture_report_
            return False

    def _wide_feature_pool(self, X, y, limit=_WIDE_SCREEN_MAX_FEATURES):
        """Select a bounded classifier split pool from fit-side class moments.

        Location and scale separation retain ordinary and U-shaped univariate signal. A deterministic quarter
        of the budget explores the remaining columns so interaction-only facts are not categorically excluded.
        The caller is responsible for untouched-fold admission of the resulting pool.
        """
        X, y = np.asarray(X, float), np.asarray(y)
        width = X.shape[1]
        limit = min(int(limit), width)
        if limit >= width:
            return np.arange(width, dtype=np.int64)
        overall_mean = X.mean(0)
        overall_var = X.var(0)
        scale = np.maximum(overall_var, 1e-12)
        location = np.zeros(width, dtype=float)
        spread = np.zeros(width, dtype=float)
        for label in np.unique(y):
            rows = y == label
            mass = float(rows.mean())
            class_mean = X[rows].mean(0)
            class_var = X[rows].var(0)
            location += mass * np.square(class_mean - overall_mean) / scale
            spread += mass * np.square(class_var - overall_var) / np.square(scale)
        score = location + 0.1 * spread
        signal_count = max(1, 3 * limit // 4)
        ranked = np.lexsort((np.arange(width), -score))
        signal = ranked[:signal_count]
        remaining = ranked[signal_count:]
        rng = np.random.default_rng(self.seed + 977)
        exploration = rng.permutation(remaining)[: limit - signal_count]
        return np.sort(np.r_[signal, exploration]).astype(np.int64, copy=False)

    def _auto_tune(self, X, y):
        """SELECT the booster configuration from the data — the user passes no hyperparameters. The number of
        rounds is chosen for free by early stopping, so this searches only the few structural knobs that move
        the needle (tree depth, learning rate, leaf size), each scaled to the dataset, via SUCCESSIVE HALVING
        (screen all configs cheaply, spend the full CV budget only on the finalists). Robust knobs (subsample /
        colsample / L2 / bins) take data-scaled defaults. CV uses refit=False — the full-data refit only
        improves the FINAL model, not the config RANKING, so it is pure waste during the search."""
        X = np.asarray(X, float)
        if self.mode == "classification":  # classification has its own config search
            return self._auto_tune_clf(X, y)
        y = np.asarray(y, float)
        n, d = X.shape
        cs = 0.6 if d > 20 else (0.8 if d > 6 else 1.0)  # thinner column sampling as width grows
        leaf = int(np.clip(n // 600, 15, 80))  # bigger leaves (more regularization) on more data
        common = dict(subsample=0.8, colsample=cs, lam=1.0, nbins=128, patience=40)
        if n < 500:  # too little data to search reliably -> safe default
            return dict(
                rounds=800, lr=0.05, depth=4, leaf=max(8, n // 40), **{**common, "nbins": 64, "patience": 60}
            )
        BASE = 1  # depth6/lr0.03: the conservative default config
        cands = [
            dict(lr=0.03, depth=4, leaf=max(leaf, 30), **common),
            dict(lr=0.03, depth=6, leaf=leaf, **common),
            dict(lr=0.03, depth=8, leaf=leaf * 2, **common),
            dict(lr=0.05, depth=6, leaf=leaf, **common),
            dict(lr=0.02, depth=8, leaf=leaf * 2, **common),  # low-lr/deep corner (tuned-GBDT sweet spot)
        ]
        cap = min(n, 8000 if n >= 15000 else 5000)  # faithful final subsample (in-regime for big data)
        # Rank near the DEPLOY round budget (rounds=2000 below), not a truncated one: a slow-converging config
        # (low-lr / shallow) is still underfit at ~600 rounds, so it gets mis-pruned/misranked and the 2000-round
        # deploy then flips the ranking (measured up to ~10% worse RMSE). Rounds are cheap here (subsamples are
        # ≤8k rows, early-stopped by patience=40), so we hold every rung at a converged budget; the speed still
        # comes from the subsample size, fold count, and successive pruning, not from starving rounds.
        rungs = [
            dict(
                rounds=1200, sub=min(cap, 2500), folds=1
            ),  # screen every candidate (small subsample, 1 split)
            dict(rounds=1200, sub=min(cap, 4000), folds=1),  # semifinal
            dict(
                rounds=1200, sub=cap, folds=2
            ),  # faithful final ranking (SH already screened -> 2 folds suffice)
        ]
        rung_evidence = {}

        def evidence(rung):
            key = (rung["sub"], rung["folds"])
            if key not in rung_evidence:
                selected, _weight = self._bounded_evidence_rows(
                    y,
                    rung["sub"],
                    self.seed + 37 * rung["sub"] + rung["folds"],
                    stratified=False,
                )
                sel = np.arange(n) if selected is None else selected
                Xs, ys = X[sel], y[sel]
                groups = self._validation_groups(sel)
                splits = self._validation_splits(
                    ys,
                    folds=max(2, rung["folds"]),
                    classification=False,
                    groups=groups,
                    single=rung["folds"] == 1,
                )
                rung_evidence[key] = Xs, ys, groups, splits
            return rung_evidence[key]

        def score_fn(cfg, rung):  # returns a thunk -> mean holdout/CV RMSE at this rung's budget
            Xs, ys, groups, splits = evidence(rung)

            def run():
                errs = []
                for tr, va in splits:
                    m = self._fit_certified(
                        AdditiveCertifiedRegressor(seed=self.seed, rounds=rung["rounds"], refit=False, **cfg),
                        Xs[tr],
                        ys[tr],
                        groups=(None if groups is None else groups[tr]),
                    )
                    errs.append(np.sqrt(((m.predict(Xs[va]) - ys[va]) ** 2).mean()))
                return float(np.mean(errs)), tuple(float(err) for err in errs)

            return run

        best, scores, _ = self._successive_halving(cands, BASE, score_fn, rungs, maximize=False)
        pick_idx = best if scores[best][0] < scores[BASE][0] * 0.998 else BASE
        pick = dict(cands[pick_idx])
        huber_score = scores[pick_idx]
        squared_cfg = dict(pick)
        squared_cfg["huber"] = None
        squared_selected = False
        try:
            squared_score = score_fn(squared_cfg, rungs[-1])()
            huber_folds = np.asarray(huber_score[1], dtype=float)
            squared_folds = np.asarray(squared_score[1], dtype=float)
            fold_reduction = (huber_folds - squared_folds) / np.maximum(huber_folds, 1e-12)
            mean_reduction = (huber_score[0] - squared_score[0]) / max(huber_score[0], 1e-12)
            squared_selected = bool(mean_reduction >= 0.002 and np.all(fold_reduction >= -0.001))
            self.regression_loss_report_ = [
                gate_report(
                    "huber_loss",
                    not squared_selected,
                    stage="predictor",
                    mean_rmse=float(huber_score[0]),
                    fold_rmse=[float(value) for value in huber_folds],
                ),
                gate_report(
                    "squared_loss",
                    squared_selected,
                    stage="predictor",
                    mean_rmse=float(squared_score[0]),
                    fold_rmse=[float(value) for value in squared_folds],
                    relative_rmse_reduction=float(mean_reduction),
                    fold_relative_rmse_reduction=[float(value) for value in fold_reduction],
                ),
            ]
        except Exception as exc:
            self.regression_loss_report_ = [
                gate_report(
                    "huber_loss",
                    True,
                    stage="predictor",
                    reason="squared_loss_gate_failed_closed",
                    error=type(exc).__name__,
                )
            ]
        pick["huber"] = None if squared_selected else 0.95
        return dict(rounds=2000, patience=60, **{k: v for k, v in pick.items() if k != "patience"})

    def _auto_tune_clf(self, X, y):  # noqa: C901 - bounded search protocol
        """SELECT the classifier configuration from the data (the classifier's static defaults are weak on many
        datasets) via SUCCESSIVE HALVING on depth / learning-rate / leaf size. Cheap rungs screen every config
        on a small subsample; only the finalists get the full, in-regime CV — so a single code path handles
        small and large n (no big/small special-case) and can afford a larger final subsample precisely because
        few configs reach it. Every candidate in a rung sees the SAME stratified rows and folds, and the primary
        objective is the benchmark-facing probability rank metric (ROC-AUC / macro OVO-AUC), not thresholded
        accuracy. rounds are early-stopped for free; CV uses refit=False (ranking, not final fit). Hysteresis and
        an accuracy guard keep the conservative default unless a rival is clearly and safely better."""
        n, d = X.shape
        rare_event = bool(getattr(self, "rare_event_", False))
        minority_count = int(np.unique(y, return_counts=True)[1].min())
        leaf = int(np.clip(minority_count // 20, 5, 50)) if rare_event else int(np.clip(n // 150, 20, 50))
        fine = max(20, leaf // 2)  # finer leaves for the deeper capacity configs
        if rare_event and minority_count < 8:
            return dict(rounds=600, lr=0.05, depth=4, leaf=leaf, patience=20)
        if n < 400:
            return dict(rounds=600, lr=0.05, depth=4, leaf=leaf, patience=20)
        BASE = 1  # depth6/lr0.05: strong conservative default
        # The depth-10 high-capacity corner only wins with enough data; on small n it overfits and loses to
        # the depth-4/6/8 configs anyway (see comment below), so screening it there just burns a rung-0 fit.
        # GATE it to large n (drop-in-place: the surviving configs' order/indices — and BASE=1 — are unchanged,
        # and at n>=15000 the full 6-config search is identical to before). Verified not to change the winner /
        # held-out accuracy. (Depth 8 is NOT gated: it can be the best config even at small n on some datasets.)
        cands = [
            dict(lr=0.05, depth=4, leaf=leaf),
            dict(lr=0.05, depth=6, leaf=leaf),
            dict(lr=0.10, depth=6, leaf=leaf),
            dict(lr=0.05, depth=8, leaf=fine),
            dict(lr=0.03, depth=6, leaf=leaf),  # lower-lr/deeper corner
            dict(lr=0.06, depth=10, leaf=fine),  # gated: n >= 15000 (wins on large data, dies early on small)
        ]
        keep = [True, True, True, True, True, n >= 15000]
        cands = [c for c, k in zip(cands, keep, strict=False) if k]
        cap = min(n, 12000 if n >= 15000 else 6000)  # faithful final subsample (in-regime for big data)
        rungs = [
            dict(rounds=120, sub=min(cap, 2500), folds=1),  # screen every candidate
            dict(rounds=300, sub=min(cap, 5000), folds=1),  # semifinal
            dict(rounds=800, sub=cap, folds=2),  # faithful final ranking
        ]
        rung_evidence = {}

        def evidence(rung):
            """One paired, stratified evidence bundle shared by every config in a rung."""
            key = (rung["sub"], rung["folds"])
            if key in rung_evidence:
                return rung_evidence[key]
            min_class_rows = min(_RARE_TUNE_MIN_EVENTS, rung["sub"] // 4) if rare_event else 0
            sel, sample_weight = self._bounded_evidence_rows(
                y,
                rung["sub"],
                self.seed + 101 * rung["sub"] + rung["folds"],
                stratified=True,
                min_class_rows=min_class_rows,
            )
            if sel is None:
                sel = np.arange(n)
                sample_weight = np.ones(n) if rare_event else None
            Xs, ys = X[sel], y[sel]
            groups = self._validation_groups(sel)

            requested = rung["folds"] if rung["folds"] >= 2 else 4
            smallest_class = int(np.unique(ys, return_counts=True)[1].min())
            split_count = min(requested, smallest_class)
            # The early returns above guarantee this in normal operation; retaining the check makes the
            # evidence contract explicit if those floors change later.
            if split_count < 2:
                raise ValueError("classifier config search requires at least two rows per class")
            splits = self._validation_splits(
                ys,
                folds=split_count,
                classification=True,
                groups=groups,
                single=rung["folds"] == 1,
            )
            rung_evidence[key] = Xs, ys, sample_weight, groups, splits
            return rung_evidence[key]

        search_allowed = None
        final_allowed = None
        if not rare_event and np.unique(y).size == 2 and d >= _WIDE_SCREEN_MIN_FEATURES:
            try:
                Xs, ys, _sample_weight, groups, splits = evidence(rungs[0])
                tr, va = splits[0]
                candidate_allowed = self._wide_feature_pool(Xs[tr], ys[tr])

                def gate_fit(allowed):
                    def run():
                        kwargs = dict(
                            rounds=rungs[0]["rounds"],
                            patience=30,
                            refit=False,
                            validation_metric="logloss",
                            rare_event=False,
                            **cands[BASE],
                        )
                        if allowed is not None:
                            kwargs["allowed"] = allowed
                        return self._fit_certified(
                            self._classifier(**kwargs),
                            Xs[tr],
                            ys[tr],
                            groups=(None if groups is None else groups[tr]),
                        )

                    return run

                fitted = _pmap(
                    {
                        "all": gate_fit(None),
                        "screened": gate_fit(candidate_allowed),
                    }
                )

                def gate_score(model):
                    raw = model._scores(Xs[va])
                    exp_score = np.exp(raw - raw.max(1, keepdims=True))
                    probability = exp_score / exp_score.sum(1, keepdims=True)
                    model_classes = np.asarray(model.classes_)
                    class_index = {value: index for index, value in enumerate(model_classes)}
                    yidx = np.array([class_index[value] for value in ys[va]], dtype=np.int64)
                    predicted = model_classes[probability.argmax(1)]
                    return (
                        _classification_rank_score(yidx, probability),
                        float(np.mean(predicted == ys[va])),
                    )

                all_rank, all_accuracy = gate_score(fitted["all"])
                screened_rank, screened_accuracy = gate_score(fitted["screened"])
                rank_delta = screened_rank - all_rank
                selected = bool(
                    rank_delta >= _WIDE_SCREEN_MIN_MEAN_GAIN and screened_accuracy >= all_accuracy - 0.002
                )
                if selected:
                    search_allowed = candidate_allowed
                    rows, _ = self._bounded_evidence_rows(
                        y,
                        _WIDE_SCREEN_FINAL_EVIDENCE_ROWS,
                        self.seed + 983,
                        stratified=True,
                    )
                    Xf = X if rows is None else X[rows]
                    yf = y if rows is None else y[rows]
                    final_allowed = self._wide_feature_pool(Xf, yf)
                self.feature_screen_report_ = [
                    gate_report(
                        "all_features",
                        not selected,
                        stage="schema",
                        holdout_rank_auc=float(all_rank),
                        holdout_accuracy=float(all_accuracy),
                        source_features=int(d),
                    ),
                    gate_report(
                        "honest_wide_feature_pool",
                        selected,
                        stage="schema",
                        holdout_rank_auc=float(screened_rank),
                        holdout_accuracy=float(screened_accuracy),
                        auc_delta=float(rank_delta),
                        selected_features=int(len(candidate_allowed)),
                        gate_rounds=int(rungs[0]["rounds"]),
                    ),
                ]
            except Exception as exc:
                self.feature_screen_report_ = [
                    gate_report(
                        "all_features",
                        True,
                        stage="schema",
                        reason="wide_feature_gate_failed_closed",
                        error=type(exc).__name__,
                        source_features=int(d),
                    )
                ]

        def score_fn(cfg, rung):  # returns a thunk -> mean holdout/CV task score at this rung's budget
            Xs, ys, sample_weight, groups, splits = evidence(rung)

            def run():
                rank_scores, rare_scores, accuracies = [], [], []
                for tr, va in splits:
                    classifier_kwargs = dict(
                        rounds=rung["rounds"],
                        patience=30,
                        refit=False,
                        validation_metric="logloss",
                        rare_event=rare_event,
                        rare_min_events=_RARE_TUNE_MIN_EVENTS,
                        min_verifier_events=min(_RARE_VERIFY_MIN_EVENTS, _RARE_TUNE_MIN_EVENTS // 4),
                    )
                    classifier_kwargs.update(cfg)
                    if search_allowed is not None:
                        classifier_kwargs["allowed"] = search_allowed
                    m = self._fit_certified(
                        self._classifier(**classifier_kwargs),
                        Xs[tr],
                        ys[tr],
                        groups=(None if groups is None else groups[tr]),
                        sample_weight=(None if sample_weight is None else sample_weight[tr]),
                    )
                    raw = m._scores(Xs[va])
                    ex = np.exp(raw - raw.max(1, keepdims=True))
                    proba = ex / ex.sum(1, keepdims=True)
                    classes = np.asarray(m.classes_)
                    predicted = classes[proba.argmax(1)]
                    validation_weight = None if sample_weight is None else sample_weight[va]
                    accuracies.append(np.average(predicted == ys[va], weights=validation_weight))
                    class_index = {value: index for index, value in enumerate(classes)}
                    yidx = np.array([class_index[value] for value in ys[va]], dtype=np.int64)
                    rank_scores.append(_classification_rank_score(yidx, proba))
                    if rare_event:
                        from sklearn.metrics import average_precision_score

                        rare_col = list(classes).index(self.rare_class_)
                        rare_scores.append(
                            average_precision_score(
                                ys[va] == self.rare_class_,
                                proba[:, rare_col],
                                sample_weight=validation_weight,
                            )
                        )
                # Tuples compare lexicographically, so successive halving ranks probability quality first and
                # uses rare-event AP / accuracy only as deterministic tie-breaks. Both finalists reach the
                # same last rung, so the comparison remains paired.
                if rare_event:
                    return (
                        float(np.mean(rank_scores)),
                        float(np.mean(rare_scores)),
                        float(np.mean(accuracies)),
                    )
                return float(np.mean(rank_scores)), float(np.mean(accuracies))

            return run

        try:
            best, scores, _ = self._successive_halving(cands, BASE, score_fn, rungs, maximize=True)
        except (ValueError, FloatingPointError) as error:
            if self._fit_validation is None:
                raise
            self.booster_selection_report_.append(
                gate_report(
                    "future_config_search",
                    False,
                    stage="predictor",
                    reason="insufficient_future_metric_support",
                    error=type(error).__name__,
                )
            )
            return dict(rounds=800, patience=20, **cands[BASE])
        if rare_event:
            best_rank, best_ap, best_accuracy = scores[best]
            base_rank, base_ap, base_accuracy = scores[BASE]
            secondary_safe = best_ap >= base_ap - _RARE_ARCH_GATE_MAX_SECONDARY_LOSS
        else:
            best_rank, best_accuracy = scores[best]
            base_rank, base_accuracy = scores[BASE]
            secondary_safe = True
        rank_win = best_rank > base_rank + 0.002
        accuracy_safe = rare_event or best_accuracy >= base_accuracy - 0.002
        pick_idx = best if rank_win and secondary_safe and accuracy_safe else BASE
        pick = dict(cands[pick_idx])
        if final_allowed is not None:
            pick["allowed"] = tuple(int(feature) for feature in final_allowed)
        # patience 30->20: clf holdout log-loss reaches ~99.7% of its final gain by ~round 100, so 20 trailing
        # rounds past the best is ample; 30 just fits ~10 extra trees per class that don't move accuracy.
        return dict(rounds=800, patience=20, **{k: v for k, v in pick.items() if k != "patience"})

    def _auto_rank_checkpoint_clf(self, X, y, boost):
        """Gate an AUC-selected prefix from the same transparent tree trace.

        This candidate is limited to binary schemas where rank checkpointing has a plausible advantage:
        skewed targets, many categorical fields, or very wide feature spaces. Each fold grows one trace while
        tracking both log-loss and AUC prefixes, so the challenger does not require a second fitted model.
        """
        X, y = np.asarray(X, float), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        category_groups = _onehot_groups(getattr(self, "_prep", None))
        eligible = bool(
            len(classes) == 2
            and len(y) >= 400
            and (
                counts.min() / counts.sum() <= _RARE_CANDIDATE_MAX_RATE
                or X.shape[1] >= _WIDE_SCREEN_MIN_FEATURES
                or len(category_groups) >= 4
            )
        )
        if not eligible or str(boost.get("validation_metric", "logloss")) != "logloss":
            return False
        rows, _ = self._bounded_evidence_rows(
            y,
            _RANK_CHECKPOINT_MAX_ROWS,
            self.seed + 619,
            stratified=True,
        )
        Xg = X if rows is None else X[rows]
        yg = y if rows is None else y[rows]
        groups = self._validation_groups(rows)
        gate_cfg = dict(boost)
        gate_cfg.pop("refit", None)
        gate_cfg.pop("validation_metric", None)
        gate_cfg["rounds"] = min(
            int(gate_cfg.get("rounds", _RANK_CHECKPOINT_ROUNDS)),
            _RANK_CHECKPOINT_ROUNDS,
        )
        gate_cfg["refit"] = False
        gate_cfg["validation_metric"] = "logloss"
        gate_cfg["track_validation_metrics"] = ("auc",)

        def fold_fit(train, valid):
            def run():
                model = self._fit_certified(
                    self._classifier(**gate_cfg),
                    Xg[train],
                    yg[train],
                    groups=(None if groups is None else groups[train]),
                )

                def score(raw):
                    exp_score = np.exp(raw - raw.max(1, keepdims=True))
                    probability = exp_score / exp_score.sum(1, keepdims=True)
                    model_classes = np.asarray(model.classes_)
                    class_index = {value: index for index, value in enumerate(model_classes)}
                    yidx = np.array([class_index[value] for value in yg[valid]], dtype=np.int64)
                    prediction = model_classes[probability.argmax(1)]
                    return (
                        _classification_rank_score(yidx, probability),
                        float(np.mean(prediction == yg[valid])),
                    )

                baseline = score(model._scores(Xg[valid]))
                challenger = score(model._scores_at_checkpoint(Xg[valid], "auc"))
                return baseline, challenger

            return run

        try:
            splits = self._validation_splits(
                yg,
                folds=2,
                classification=True,
                groups=groups,
            )
            results = _pmap({index: fold_fit(train, valid) for index, (train, valid) in enumerate(splits)})
            baseline_rank = np.array([results[index][0][0] for index in range(len(splits))], dtype=float)
            challenger_rank = np.array([results[index][1][0] for index in range(len(splits))], dtype=float)
            baseline_accuracy = np.array([results[index][0][1] for index in range(len(splits))], dtype=float)
            challenger_accuracy = np.array(
                [results[index][1][1] for index in range(len(splits))], dtype=float
            )
            delta = challenger_rank - baseline_rank
            selected = bool(
                np.all(delta >= 0.0)
                and float(delta.mean()) >= _RANK_CHECKPOINT_MIN_GAIN
                and float(challenger_accuracy.mean()) >= float(baseline_accuracy.mean()) - 0.002
            )
            self.rank_checkpoint_report_ = [
                gate_report(
                    "logloss_checkpoint",
                    not selected,
                    stage="predictor",
                    fold_rank_auc=[float(value) for value in baseline_rank],
                    mean_accuracy=float(baseline_accuracy.mean()),
                ),
                gate_report(
                    "auc_checkpoint",
                    selected,
                    stage="predictor",
                    fold_rank_auc=[float(value) for value in challenger_rank],
                    fold_auc_delta=[float(value) for value in delta],
                    mean_accuracy=float(challenger_accuracy.mean()),
                    evidence_rows=int(len(yg)),
                    shared_tree_trace=True,
                ),
            ]
            return selected
        except Exception as exc:
            self.rank_checkpoint_report_ = [
                gate_report(
                    "logloss_checkpoint",
                    True,
                    stage="predictor",
                    reason="rank_checkpoint_gate_failed_closed",
                    error=type(exc).__name__,
                )
            ]
            return False

    def _auto_class_weight(self, X, y, boost):
        """Balance classes only when the probability ranking improves.

        A weighted learner can improve a fixed 0.5 decision threshold while
        degrading the ranking exposed by ``predict_proba``. Tabular benchmarks
        and most downstream threshold policies use ROC-AUC, so this gate compares
        stratified OOF AUC rather than balanced accuracy. Mild imbalance stays
        unweighted; callers retain explicit threshold control through calibrated
        probabilities and ``select``.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        y = np.asarray(y)
        cls, counts = np.unique(y, return_counts=True)
        self.class_weight_report_ = [{"name": "unweighted", "selected": True}]
        if len(cls) < 2 or counts.min() / counts.sum() > 0.2:
            return None
        if len(y) > 6000:
            if self._fit_validation is None:
                _, sel = train_test_split(
                    np.arange(len(y)), test_size=6000, random_state=self.seed + 11, stratify=y
                )
            else:
                selected, _weight = self._bounded_evidence_rows(
                    y,
                    6000,
                    self.seed + 11,
                    stratified=True,
                )
                sel = np.arange(len(y)) if selected is None else selected
        else:
            sel = np.arange(len(y))
        Xs, ys = X[sel], y[sel]
        groups = self._validation_groups(sel)
        _, sample_counts = np.unique(ys, return_counts=True)
        folds = min(3, int(sample_counts.min()))
        if folds < 2:
            return None
        light = {k: v for k, v in boost.items() if k != "class_weight"}
        light["rounds"] = min(light.get("rounds", 400), 400)

        def cv_auc(cw):
            splits = self._validation_splits(
                ys,
                folds=folds,
                classification=True,
                groups=groups,
            )

            def mk(tr, va):
                def run():
                    m = self._fit_certified(
                        self._classifier(class_weight=cw, **light),
                        Xs[tr],
                        ys[tr],
                        groups=(None if groups is None else groups[tr]),
                    )
                    scores = m._scores(Xs[va])
                    probs = np.exp(scores - scores.max(1, keepdims=True))
                    probs /= probs.sum(1, keepdims=True)
                    if probs.shape[1] == 2:
                        return float(roc_auc_score(ys[va], probs[:, 1]))
                    return float(roc_auc_score(ys[va], probs, multi_class="ovo", average="macro"))

                return run

            r = list(_pmap({i: mk(tr, va) for i, (tr, va) in enumerate(splits)}).values())
            return float(np.mean(r))

        try:
            auc_w = cv_auc("balanced")
            auc_n = cv_auc(None)
        except (ValueError, FloatingPointError):
            return None
        selected = auc_w >= auc_n + 0.003
        self.class_weight_report_ = [
            {"name": "unweighted", "oof_rank_auc": float(auc_n), "selected": not selected},
            {"name": "balanced", "oof_rank_auc": float(auc_w), "selected": selected},
        ]
        return "balanced" if selected else None

    def _auto_joint_rank_regions(self, X, y, boost):
        """Gate a coupled depth/leaf/checkpoint architecture as one unit.

        Separate depth and affine-leaf gates can miss a combination whose
        pieces are weak alone. The challenger keeps finite threshold regions,
        adds the existing certified affine leaves, and selects its prefix by
        untouched rank AUC. It is limited to near-rare category-rich binary
        tables where that complementary geometry is useful.
        """
        from sklearn.metrics import roc_auc_score

        X, y = np.asarray(X, float), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        if not (
            400 <= len(y) <= 10_000
            and len(classes) == 2
            and counts.min() >= 2
            and counts.min() / counts.sum() > _RARE_EVENT_MAX_RATE
            and counts.min() / counts.sum() <= 0.10
            and len(_onehot_groups(getattr(self, "_prep", None))) >= 4
        ):
            return None
        baseline = dict(boost)
        challenger = dict(boost)
        challenger.update(lr=0.05, depth=4, linear_leaf=True, validation_metric="auc")
        if all(challenger.get(key) == baseline.get(key) for key in challenger):
            return None
        baseline_gate, challenger_gate = dict(baseline), dict(challenger)
        for config in (baseline_gate, challenger_gate):
            config["rounds"] = min(int(config.get("rounds", 400)), 400)
            config["refit"] = False

        baseline_scores, challenger_scores = [], []
        groups = self._validation_groups()
        try:
            splits = self._validation_splits(
                y,
                folds=2,
                classification=True,
                groups=groups,
            )
            for train, validation in splits:
                train_groups = None if groups is None else groups[train]
                baseline_model = self._fit_certified(
                    self._classifier(**baseline_gate),
                    X[train],
                    y[train],
                    groups=train_groups,
                )
                challenger_model = self._fit_certified(
                    self._classifier(**challenger_gate),
                    X[train],
                    y[train],
                    groups=train_groups,
                )

                def score(model, *, validation_rows=validation):
                    raw = model._scores(X[validation_rows])
                    probability = np.exp(raw - raw.max(1, keepdims=True))
                    probability /= probability.sum(1, keepdims=True)
                    positive = model.classes_[1]
                    return float(roc_auc_score(y[validation_rows] == positive, probability[:, 1]))

                baseline_scores.append(score(baseline_model))
                challenger_scores.append(score(challenger_model))
        except Exception:
            return None

        delta = np.asarray(challenger_scores) - np.asarray(baseline_scores)
        selected = bool(np.all(delta > 0.0) and float(delta.mean()) >= 0.003)
        self.booster_selection_report_.extend(
            [
                gate_report(
                    "current_region_architecture",
                    not selected,
                    stage="predictor",
                    mean_auc=float(np.mean(baseline_scores)),
                ),
                gate_report(
                    "joint_rank_regions",
                    selected,
                    stage="predictor",
                    mean_auc=float(np.mean(challenger_scores)),
                    fold_auc_delta=[float(value) for value in delta],
                ),
            ]
        )
        return challenger if selected else None

    def _auto_shallow_boost(self, X, y, boost):
        """Let a compact certified region program challenge the tuned booster.

        Deep regions can overfit small and medium tables even when their
        threshold accuracy wins the initial broad config search. A depth-three
        program is a distinct, readable inductive bias. It replaces the tuned
        program only after a two-fold ROC-AUC win on both untouched folds and a
        deliberately large mean margin, so this extra capacity choice fails
        closed on ordinary tables.
        """
        from sklearn.metrics import roc_auc_score

        X, y = np.asarray(X, float), np.asarray(y)
        self.booster_selection_report_ = [gate_report("auto_boost", True, stage="predictor")]
        classes, counts = np.unique(y, return_counts=True)
        if not (400 <= len(y) <= 10_000 and len(classes) == 2 and counts.min() >= 2):
            return None

        shallow = dict(boost)
        shallow.update(
            lr=0.05,
            depth=3,
            leaf=max(10, min(30, int(boost.get("leaf", 20)))),
        )
        if all(shallow.get(key) == boost.get(key) for key in ("lr", "depth", "leaf")):
            return None
        base_gate, shallow_gate = dict(boost), dict(shallow)
        base_gate["refit"] = False
        shallow_gate["refit"] = False
        groups = self._validation_groups()
        try:
            splits = self._validation_splits(
                y,
                folds=2,
                classification=True,
                groups=groups,
            )

            def fit_auc(config, tr, va):
                def run():
                    model = self._fit_certified(
                        self._classifier(**config),
                        X[tr],
                        y[tr],
                        groups=(None if groups is None else groups[tr]),
                    )
                    scores = model._scores(X[va])
                    probs = np.exp(scores - scores.max(1, keepdims=True))
                    probs /= probs.sum(1, keepdims=True)
                    return float(roc_auc_score(y[va] == model.classes_[1], probs[:, 1]))

                return run

            thunks = {}
            for index, (tr, va) in enumerate(splits):
                thunks[(index, "base")] = fit_auc(base_gate, tr, va)
                thunks[(index, "shallow")] = fit_auc(shallow_gate, tr, va)
            # This gate runs before the linear-leaf challenger in the default
            # fit order. Preserve serial evaluation if a future caller supplies
            # affine leaves, whose Python leaf setup contends under threads.
            fitted = (
                {key: run() for key, run in thunks.items()}
                if base_gate.get("linear_leaf", False) or shallow_gate.get("linear_leaf", False)
                else _pmap(thunks)
            )
            base_scores = [fitted[(index, "base")] for index in range(len(splits))]
            shallow_scores = [fitted[(index, "shallow")] for index in range(len(splits))]
        except Exception:
            return None

        deltas = np.asarray(shallow_scores) - np.asarray(base_scores)
        selected = bool(np.all(deltas > 0.0) and float(deltas.mean()) >= 0.015)
        self.booster_selection_report_ = [
            gate_report("auto_boost", not selected, stage="predictor", mean_auc=float(np.mean(base_scores))),
            gate_report(
                "shallow_certified_boost",
                selected,
                stage="predictor",
                mean_auc=float(np.mean(shallow_scores)),
                fold_auc_delta=[float(delta) for delta in deltas],
            ),
        ]
        return shallow if selected else None

    def _auto_rare_interactions(self, X, y, boost):  # noqa: C901 - coupled evidence gate
        """Select a coupled rank-checkpoint and rare-rule architecture by weighted AP.

        Proposal rows may be case-control sampled for bounded cost, but inverse
        inclusion weights preserve the source prevalence in model fitting and
        every gate score.  The same deterministic folds compare the current
        booster, an AP-checkpointed booster, and the checkpointed booster with
        replayable minority-tail clauses.
        """
        from tabpvn.predicate_compiler import SymbolicPredicateMap
        from tabpvn.proposers import ClassificationEvidenceWorkspace

        X, y = np.asarray(X, float), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        self.candidate_report_ = [gate_report("certified_boost", True, stage="predictor")]
        if len(classes) != 2 or counts.min() < _RARE_RULE_GATE_MIN_EVENTS or not np.isfinite(X).all():
            return None

        minimum = min(
            _RARE_RULE_GATE_MAX_EVENTS,
            int(counts.min()),
            _RARE_RULE_GATE_MAX_ROWS // 4,
        )
        try:
            rows, inclusion_weight = self._bounded_evidence_rows(
                y,
                _RARE_RULE_GATE_MAX_ROWS,
                self.seed + 409,
                stratified=True,
                min_class_rows=minimum,
            )
        except Exception:
            return None
        Xg = X if rows is None else X[rows]
        yg = y if rows is None else y[rows]
        groups = self._validation_groups(rows)
        weights = (
            np.ones(len(yg), dtype=float)
            if inclusion_weight is None
            else np.asarray(inclusion_weight, dtype=float)
        )
        rare_class = self.rare_class_
        exclusive_groups = ()
        if getattr(self, "_prep", None) is not None:
            _, meta, _ = self._reason_groups()
            exclusive_groups = tuple(group["cols"] for group in meta if group["kind"] == "onehot")

        common = dict(boost)
        common["rounds"] = min(common.get("rounds", 400), 400)
        common["refit"] = False
        common["rare_event"] = True
        common["rare_min_events"] = min(_RARE_TUNE_MIN_EVENTS, minimum)
        common["min_verifier_events"] = min(
            _RARE_VERIFY_MIN_EVENTS,
            max(4, int(np.sum(yg == rare_class) // 6)),
        )
        rank_capable = bool(rare_class == classes[-1])
        base_metric = str(common.get("validation_metric", "logloss"))
        trace_cfg = dict(common)
        trace_cfg["validation_metric"] = base_metric
        if rank_capable and base_metric != "average_precision":
            trace_cfg["track_validation_metrics"] = ("average_precision",)

        try:
            evidence_splits = self._validation_splits(
                yg,
                folds=2,
                classification=True,
                groups=groups,
            )
            workspace = ClassificationEvidenceWorkspace(
                yg,
                seed=self.seed,
                folds=2,
                metric="average_precision",
                positive_class=rare_class,
                sample_weight=weights,
                splits=tuple(evidence_splits) if groups is not None else None,
            )
        except ValueError:
            return None

        def checkpoint_scores(model, features, metric):
            scorer = getattr(model, "_scores_at_checkpoint", None)
            if scorer is not None:
                try:
                    return scorer(features, metric)
                except ValueError:
                    pass
            return model._scores(features)

        def probabilities(scores):
            exp_scores = np.exp(scores - scores.max(1, keepdims=True))
            return exp_scores / exp_scores.sum(1, keepdims=True)

        def checkpoint_evaluators(mapper_factory=None):
            cache = {}

            def for_metric(metric):
                def evaluate(train, valid):
                    key = np.asarray(valid, dtype=np.int64).tobytes()
                    if key not in cache:
                        mapper = None if mapper_factory is None else mapper_factory(train, valid)
                        train_features = Xg[train] if mapper is None else mapper.transform(Xg[train])
                        model = self._fit_certified(
                            self._classifier(**trace_cfg),
                            train_features,
                            yg[train],
                            groups=(None if groups is None else groups[train]),
                            sample_weight=weights[train],
                        )
                        cache[key] = (model, mapper)
                    model, mapper = cache[key]
                    valid_features = Xg[valid] if mapper is None else mapper.transform(Xg[valid])
                    score = checkpoint_scores(model, valid_features, metric)
                    return probabilities(score), np.asarray(model.classes_)

                return evaluate

            return cache, for_metric

        def residual_mapper(features, labels, mapper_weights, probability):
            event = np.asarray(labels) == rare_class
            residual = event.astype(float) - probability
            hessian = np.clip(probability * (1.0 - probability), 1e-9, None)
            mapper = SymbolicPredicateMap(
                seed=self.seed,
                exclusive_groups=exclusive_groups,
                rare_rules=True,
                rare_class=rare_class,
            ).fit(
                features,
                labels,
                sample_weight=mapper_weights,
                residual=residual,
                hessian=hessian,
            )
            if not mapper.predicates:
                mapper = SymbolicPredicateMap(
                    seed=self.seed,
                    exclusive_groups=exclusive_groups,
                    rare_rules=True,
                    rare_class=rare_class,
                ).fit(features, labels, sample_weight=mapper_weights)
                mapper.proposal_objective_ = "label_association_fallback"
            else:
                mapper.proposal_objective_ = "booster_residual_newton_gain"
            return mapper

        try:
            raw_cache, raw_evaluators = checkpoint_evaluators()
            baseline = workspace.evaluate(
                "rare_baseline",
                raw_evaluators(base_metric),
            )
            rank = (
                workspace.evaluate(
                    "rare_rank_checkpoint",
                    raw_evaluators("average_precision"),
                )
                if rank_capable
                else baseline
            )
            rank_selected, rank_deltas = (
                workspace.accepts(
                    rank,
                    baseline,
                    min_fold_gain=_RARE_RULE_MIN_FOLD_GAIN,
                    min_mean_gain=_RARE_RULE_MIN_MEAN_GAIN,
                )
                if rank_capable
                else (False, np.zeros(len(workspace.splits), dtype=float))
            )

            rare_column = int(np.flatnonzero(workspace.classes == rare_class)[0])
            full_mapper = residual_mapper(
                Xg,
                yg,
                weights,
                baseline.probabilities[:, rare_column],
            )
            rules_selected = False
            rule_deltas = incremental_deltas = np.asarray([], dtype=float)
            rule_evidence = None
            if full_mapper.predicates:

                def mapper_factory(train, valid):
                    key = np.asarray(valid, dtype=np.int64).tobytes()
                    raw_model, _ = raw_cache[key]
                    proposal = getattr(raw_model, "ver_", None)
                    proposal = (
                        np.arange(len(train), dtype=int)
                        if proposal is None
                        else np.asarray(proposal, dtype=int)
                    )
                    if (
                        len(proposal) < SymbolicPredicateMap.MIN_FIT_ROWS
                        or np.unique(yg[train][proposal]).size < 2
                    ):
                        proposal = np.arange(len(train), dtype=int)
                    proposal_features = Xg[train][proposal]
                    proposal_labels = yg[train][proposal]
                    proposal_weights = weights[train][proposal]
                    raw_score = checkpoint_scores(
                        raw_model,
                        proposal_features,
                        base_metric,
                    )
                    raw_probability = probabilities(raw_score)
                    model_rare_column = int(np.flatnonzero(np.asarray(raw_model.classes_) == rare_class)[0])
                    mapper = residual_mapper(
                        proposal_features,
                        proposal_labels,
                        proposal_weights,
                        raw_probability[:, model_rare_column],
                    )
                    if not mapper.predicates:
                        raise ValueError("residual-rule fold produced no replayable predicates")
                    return mapper

                _rule_cache, rule_evaluators = checkpoint_evaluators(mapper_factory=mapper_factory)
                rule_metric = "average_precision" if rank_capable else base_metric
                try:
                    rule_evidence = workspace.evaluate(
                        "rare_rank_rules",
                        rule_evaluators(rule_metric),
                    )
                    absolute_selected, rule_deltas = workspace.accepts(
                        rule_evidence,
                        baseline,
                        min_fold_gain=_RARE_RULE_MIN_FOLD_GAIN,
                        min_mean_gain=_RARE_RULE_MIN_MEAN_GAIN,
                    )
                    lower = rank if rank_selected else baseline
                    incremental_selected, incremental_deltas = workspace.accepts(
                        rule_evidence,
                        lower,
                        min_fold_gain=_RARE_RULE_MIN_FOLD_GAIN,
                        min_mean_gain=_RARE_RULE_MIN_MEAN_GAIN,
                    )
                    rules_selected = bool(absolute_selected and incremental_selected)
                except Exception:
                    rule_evidence = None
                    rule_deltas = incremental_deltas = np.asarray([], dtype=float)
                    rules_selected = False
        except Exception:
            return None

        deploy_rank = bool(rank_capable and (rank_selected or rules_selected))
        if deploy_rank:
            boost["validation_metric"] = "average_precision"
        selected_mapper = full_mapper if rules_selected else None
        self.candidate_report_ = [
            gate_report(
                "certified_boost",
                not deploy_rank and selected_mapper is None,
                stage="predictor",
                metric="average_precision",
                mean_score=baseline.mean_score,
            ),
            gate_report(
                "rare_rank_checkpoint",
                rank_selected,
                stage="predictor",
                metric="average_precision",
                mean_score=rank.mean_score,
                fold_ap_delta=[float(delta) for delta in rank_deltas],
                deployed=deploy_rank,
                required_by_selected_rules=rules_selected,
            ),
            gate_report(
                "rare_symbolic_predicate_boost",
                rules_selected,
                stage="schema",
                metric="average_precision",
                mean_score=(None if rule_evidence is None else rule_evidence.mean_score),
                fold_ap_delta=[float(delta) for delta in rule_deltas],
                incremental_ap_delta=[float(delta) for delta in incremental_deltas],
                proposal_objective=getattr(full_mapper, "proposal_objective_", None),
                interval_candidates_considered=int(
                    getattr(full_mapper, "interval_candidates_considered_", 0)
                ),
                interval_predicates=int(getattr(full_mapper, "interval_predicates_selected_", 0)),
                interval_union_candidates=int(getattr(full_mapper, "interval_union_candidates_", 0)),
                interval_union_predicates=int(getattr(full_mapper, "interval_union_predicates_selected_", 0)),
                residual_interval_predicates=int(getattr(full_mapper, "residual_interval_predicates_", 0)),
                multi_interval_columns=int(getattr(full_mapper, "multi_interval_columns_", 0)),
                residual_allocator=getattr(full_mapper, "residual_allocator_", None),
                residual_candidate_families=getattr(
                    full_mapper,
                    "residual_candidate_family_counts_",
                    {},
                ),
                residual_selected_families=getattr(
                    full_mapper,
                    "residual_selected_family_counts_",
                    {},
                ),
            ),
        ]
        if self.rare_event_report_ is not None:
            self.rare_event_report_["architecture_gate"] = {
                "rows": int(len(yg)),
                "events": int(np.sum(yg == rare_class)),
                "weighted_rate": float(np.average(yg == rare_class, weights=weights)),
                "rank_checkpoint": deploy_rank,
                "symbolic_rules": rules_selected,
                "proposal_objective": getattr(full_mapper, "proposal_objective_", None),
                "interval_predicates": int(getattr(full_mapper, "interval_predicates_selected_", 0)),
                "interval_union_candidates": int(getattr(full_mapper, "interval_union_candidates_", 0)),
                "interval_union_predicates": int(
                    getattr(full_mapper, "interval_union_predicates_selected_", 0)
                ),
                "residual_interval_predicates": int(getattr(full_mapper, "residual_interval_predicates_", 0)),
                "residual_allocator": getattr(full_mapper, "residual_allocator_", None),
                "residual_selected_families": getattr(
                    full_mapper,
                    "residual_selected_family_counts_",
                    {},
                ),
                "fold_booster_fits": 4 if full_mapper.predicates else 2,
            }
        return selected_mapper

    def _auto_multiclass_interactions(self, X, y, boost):  # noqa: C901 - coupled evidence gate
        """Gate rank checkpointing and class-conditional residual predicates.

        One coupled softmax trajectory supplies both log-loss and macro OVO-AUC
        prefixes.  Residual predicates are proposed one-vs-rest from leak-safe
        probabilities, merged into a finite class-balanced schema, and accepted
        only when the augmented certified booster improves both shared folds.
        """
        from tabpvn.predicate_compiler import (
            MulticlassCrossfitPredicateMap,
            MulticlassResidualPredicateMap,
            SymbolicPredicateMap,
        )
        from tabpvn.proposers import ClassificationEvidenceWorkspace

        X, y = np.asarray(X, float), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        self.candidate_report_ = [gate_report("certified_boost", True, stage="predictor")]
        if not (
            _MULTICLASS_RULE_GATE_MIN_ROWS <= len(y)
            and 3 <= len(classes) <= _MULTICLASS_RULE_GATE_MAX_CLASSES
            and counts.min() >= _MULTICLASS_RULE_GATE_MIN_CLASS_ROWS
            and np.isfinite(X).all()
            and not bool(boost.get("linear_leaf", False))
        ):
            return None

        minimum = min(
            int(counts.min()),
            max(
                _MULTICLASS_RULE_GATE_MIN_CLASS_ROWS,
                _MULTICLASS_RULE_GATE_MAX_ROWS // (2 * len(classes)),
            ),
        )
        try:
            rows, inclusion_weight = self._bounded_evidence_rows(
                y,
                _MULTICLASS_RULE_GATE_MAX_ROWS,
                self.seed + 613,
                stratified=True,
                min_class_rows=minimum,
            )
        except Exception:
            return None
        Xg = X if rows is None else X[rows]
        yg = y if rows is None else y[rows]
        groups = self._validation_groups(rows)
        weights = (
            np.ones(len(yg), dtype=float)
            if inclusion_weight is None
            else np.asarray(inclusion_weight, dtype=float)
        )
        if not SymbolicPredicateMap._fittable(
            Xg,
            yg == classes[0],
            numeric_rules=True,
        ):
            return None

        exclusive_groups = ()
        if getattr(self, "_prep", None) is not None:
            _, meta, _ = self._reason_groups()
            exclusive_groups = tuple(group["cols"] for group in meta if group["kind"] == "onehot")

        common = dict(boost)
        common["rounds"] = min(
            int(common.get("rounds", _MULTICLASS_RULE_GATE_ROUNDS)),
            _MULTICLASS_RULE_GATE_ROUNDS,
        )
        common["refit"] = False
        base_metric = str(common.get("validation_metric", "logloss"))
        rank_metric = "macro_ovo_auc"
        trace_cfg = dict(common)
        trace_cfg["validation_metric"] = base_metric
        trace_cfg["stratified_holdout"] = True
        if base_metric not in {"auc", rank_metric}:
            tracked = tuple(trace_cfg.get("track_validation_metrics", ()))
            trace_cfg["track_validation_metrics"] = tuple(dict.fromkeys((*tracked, rank_metric)))

        try:
            evidence_splits = self._validation_splits(
                yg,
                folds=2,
                classification=True,
                groups=groups,
            )
            workspace = ClassificationEvidenceWorkspace(
                yg,
                seed=self.seed,
                folds=2,
                metric="roc_auc",
                sample_weight=weights,
                splits=tuple(evidence_splits) if groups is not None else None,
            )
        except ValueError:
            return None

        def checkpoint_scores(model, features, metric):
            scorer = getattr(model, "_scores_at_checkpoint", None)
            if scorer is not None:
                try:
                    return scorer(features, metric)
                except ValueError:
                    pass
            return model._scores(features)

        def probabilities(scores):
            exp_scores = np.exp(scores - scores.max(1, keepdims=True))
            return exp_scores / exp_scores.sum(1, keepdims=True)

        def checkpoint_evaluators(mapper_factory=None):
            cache = {}

            def for_metric(metric):
                def evaluate(train, valid):
                    key = np.asarray(valid, dtype=np.int64).tobytes()
                    if key not in cache:
                        mapper = None if mapper_factory is None else mapper_factory(train, valid)
                        train_features = Xg[train] if mapper is None else mapper.transform(Xg[train])
                        model = self._fit_certified(
                            self._classifier(**trace_cfg),
                            train_features,
                            yg[train],
                            groups=(None if groups is None else groups[train]),
                            sample_weight=weights[train],
                        )
                        cache[key] = (model, mapper)
                    model, mapper = cache[key]
                    valid_features = Xg[valid] if mapper is None else mapper.transform(Xg[valid])
                    score = checkpoint_scores(model, valid_features, metric)
                    return probabilities(score), np.asarray(model.classes_)

                return evaluate

            return cache, for_metric

        def residual_mapper(features, labels, mapper_weights, probability, mapper_classes):
            return MulticlassResidualPredicateMap(
                seed=self.seed,
                exclusive_groups=exclusive_groups,
            ).fit(
                features,
                labels,
                probability,
                classes=mapper_classes,
                sample_weight=mapper_weights,
            )

        try:
            raw_cache, raw_evaluators = checkpoint_evaluators()
            baseline = workspace.evaluate(
                "multiclass_baseline",
                raw_evaluators(base_metric),
            )
            rank = workspace.evaluate(
                "multiclass_rank_checkpoint",
                raw_evaluators(rank_metric),
            )
            rank_selected, rank_deltas = workspace.accepts(
                rank,
                baseline,
                min_fold_gain=_MULTICLASS_RULE_MIN_FOLD_GAIN,
                min_mean_gain=_MULTICLASS_RULE_MIN_MEAN_GAIN,
            )

            full_mapper = residual_mapper(
                Xg,
                yg,
                weights,
                baseline.probabilities,
                workspace.classes,
            )
            rules_selected = False
            rule_deltas = incremental_deltas = np.asarray([], dtype=float)
            rule_evidence = None
            head_evidence = None
            head_deltas = np.asarray([], dtype=float)
            head_screen_passed = False
            head_selected = False
            crossfit_mapper = None
            if full_mapper.predicates:
                fold_mapper_cache = {}

                def mapper_factory(train, valid):
                    key = np.asarray(valid, dtype=np.int64).tobytes()
                    if key in fold_mapper_cache:
                        return fold_mapper_cache[key]
                    raw_model, _ = raw_cache[key]
                    proposal = getattr(raw_model, "ver_", None)
                    proposal = (
                        np.arange(len(train), dtype=int)
                        if proposal is None
                        else np.asarray(proposal, dtype=int)
                    )
                    proposal_labels = yg[train][proposal]
                    if len(proposal) < MulticlassResidualPredicateMap.MIN_FIT_ROWS or np.unique(
                        proposal_labels
                    ).size != len(classes):
                        proposal = np.arange(len(train), dtype=int)
                        proposal_labels = yg[train]
                    proposal_features = Xg[train][proposal]
                    proposal_weights = weights[train][proposal]
                    raw_scores = checkpoint_scores(
                        raw_model,
                        proposal_features,
                        base_metric,
                    )
                    mapper = residual_mapper(
                        proposal_features,
                        proposal_labels,
                        proposal_weights,
                        probabilities(raw_scores),
                        np.asarray(raw_model.classes_),
                    )
                    if not mapper.predicates:
                        raise ValueError("multiclass residual fold produced no replayable predicates")
                    fold_mapper_cache[key] = mapper
                    return mapper

                def evaluate_head(train, valid):
                    key = np.asarray(valid, dtype=np.int64).tobytes()
                    raw_model, _ = raw_cache[key]
                    mapper = mapper_factory(train, valid)
                    raw_scores = checkpoint_scores(
                        raw_model,
                        Xg[valid],
                        base_metric,
                    )
                    updated = mapper.residual_score_update(
                        raw_scores,
                        Xg[valid],
                        np.asarray(raw_model.classes_),
                        learning_rate=float(common.get("lr", 0.05)),
                    )
                    return probabilities(updated), np.asarray(raw_model.classes_)

                _rule_cache, rule_evaluators = checkpoint_evaluators(mapper_factory=mapper_factory)
                try:
                    head_evidence = workspace.evaluate(
                        "multiclass_residual_stump_head",
                        evaluate_head,
                    )
                    head_deltas = workspace.deltas(head_evidence, baseline)
                    head_screen_passed = bool(
                        len(head_deltas)
                        and np.all(head_deltas >= _MULTICLASS_HEAD_SCREEN_MIN_FOLD_GAIN)
                        and float(head_deltas.mean()) >= 0.0
                    )
                    head_selected, _head_acceptance_deltas = workspace.accepts(
                        head_evidence,
                        baseline,
                        min_fold_gain=_MULTICLASS_RULE_MIN_FOLD_GAIN,
                        min_mean_gain=_MULTICLASS_RULE_MIN_MEAN_GAIN,
                    )
                    head_selected = bool(head_selected and not rank_selected)
                    try:
                        fold_maps = []
                        fold_valid_rows = []
                        for _train, valid in workspace.splits:
                            key = np.asarray(valid, dtype=np.int64).tobytes()
                            fold_maps.append(fold_mapper_cache[key])
                            fold_valid_rows.append(valid)
                        candidate_mapper = MulticlassCrossfitPredicateMap(
                            seed=self.seed,
                            exclusive_groups=exclusive_groups,
                        ).fit_from_folds(
                            Xg,
                            yg,
                            baseline.probabilities,
                            workspace.classes,
                            fold_maps,
                            fold_valid_rows,
                            sample_weight=weights,
                        )
                        if candidate_mapper.predicates:
                            crossfit_mapper = candidate_mapper
                    except Exception:
                        crossfit_mapper = None
                    if head_screen_passed:
                        rule_evidence = workspace.evaluate(
                            "multiclass_rank_rules",
                            rule_evaluators(rank_metric),
                        )
                        absolute_selected, rule_deltas = workspace.accepts(
                            rule_evidence,
                            baseline,
                            min_fold_gain=_MULTICLASS_RULE_MIN_FOLD_GAIN,
                            min_mean_gain=_MULTICLASS_RULE_MIN_MEAN_GAIN,
                        )
                        lower = head_evidence if head_selected else (rank if rank_selected else baseline)
                        incremental_selected, incremental_deltas = workspace.accepts(
                            rule_evidence,
                            lower,
                            min_fold_gain=_MULTICLASS_RULE_MIN_FOLD_GAIN,
                            min_mean_gain=_MULTICLASS_RULE_MIN_MEAN_GAIN,
                        )
                        rules_selected = bool(absolute_selected and incremental_selected)
                except Exception:
                    rule_evidence = None
                    rule_deltas = incremental_deltas = np.asarray([], dtype=float)
                    rules_selected = False

        except Exception:
            return None

        deploy_head = bool(head_selected and not rules_selected)
        deploy_rank = bool(rank_selected or rules_selected)
        if deploy_rank:
            boost["validation_metric"] = rank_metric
        deployed_head_mapper = crossfit_mapper if deploy_head and crossfit_mapper is not None else full_mapper
        if deploy_head:
            boost["base_feature_count"] = int(X.shape[1])
            boost["residual_stumps"] = tuple(
                (
                    int(X.shape[1] + index),
                    owner.item() if isinstance(owner, np.generic) else owner,
                    float(update[0]),
                    float(update[1]),
                )
                for index, (owner, update) in enumerate(
                    zip(
                        deployed_head_mapper.predicate_classes_,
                        deployed_head_mapper.predicate_updates_,
                        strict=False,
                    )
                )
            )
        selected_mapper = full_mapper if rules_selected else (deployed_head_mapper if deploy_head else None)
        class_rule_counts = [
            {
                "class": (label.item() if isinstance(label, np.generic) else label),
                "predicates": int(sum(owner == label for owner in deployed_head_mapper.predicate_classes_)),
            }
            for label in workspace.classes
        ]
        self.candidate_report_ = [
            gate_report(
                "certified_boost",
                not deploy_rank and not deploy_head and selected_mapper is None,
                stage="predictor",
                metric=rank_metric,
                mean_score=baseline.mean_score,
            ),
            gate_report(
                "multiclass_rank_checkpoint",
                rank_selected,
                stage="predictor",
                metric=rank_metric,
                mean_score=rank.mean_score,
                fold_auc_delta=[float(delta) for delta in rank_deltas],
                deployed=deploy_rank,
                required_by_selected_rules=rules_selected,
            ),
            gate_report(
                "multiclass_residual_stump_head",
                deploy_head,
                stage="predictor",
                metric=rank_metric,
                mean_score=(None if head_evidence is None else head_evidence.mean_score),
                fold_auc_delta=[float(delta) for delta in head_deltas],
                screen_passed=head_screen_passed,
                predicates=int(len(deployed_head_mapper.predicates)),
                deployment_objective=getattr(
                    deployed_head_mapper,
                    "proposal_objective_",
                    None,
                ),
                residual_allocator=getattr(deployed_head_mapper, "residual_allocator_", None),
                interval_union_candidates=int(getattr(deployed_head_mapper, "interval_union_candidates_", 0)),
                interval_union_predicates=int(
                    getattr(deployed_head_mapper, "interval_union_predicates_selected_", 0)
                ),
                residual_selected_families=getattr(
                    deployed_head_mapper,
                    "residual_selected_family_counts_",
                    {},
                ),
            ),
            gate_report(
                "multiclass_residual_predicate_boost",
                rules_selected,
                stage="schema",
                metric=rank_metric,
                mean_score=(None if rule_evidence is None else rule_evidence.mean_score),
                fold_auc_delta=[float(delta) for delta in rule_deltas],
                incremental_auc_delta=[float(delta) for delta in incremental_deltas],
                stump_head_mean_score=(None if head_evidence is None else head_evidence.mean_score),
                stump_head_fold_auc_delta=[float(delta) for delta in head_deltas],
                stump_head_screen_passed=head_screen_passed,
                proposal_objective=full_mapper.proposal_objective_,
                class_predicate_counts=class_rule_counts,
                residual_allocator=getattr(full_mapper, "residual_allocator_", None),
                interval_union_candidates=int(getattr(full_mapper, "interval_union_candidates_", 0)),
                interval_union_predicates=int(getattr(full_mapper, "interval_union_predicates_selected_", 0)),
                residual_selected_families=getattr(
                    full_mapper,
                    "residual_selected_family_counts_",
                    {},
                ),
            ),
        ]
        self.multiclass_architecture_report_ = {
            "rows": int(len(yg)),
            "classes": int(len(classes)),
            "rank_checkpoint": deploy_rank,
            "residual_stump_head": deploy_head,
            "symbolic_rules": rules_selected,
            "predicates": int(
                len(deployed_head_mapper.predicates) if deploy_head else len(full_mapper.predicates)
            ),
            "class_predicate_counts": class_rule_counts,
            "interval_union_candidates": int(
                getattr(deployed_head_mapper if deploy_head else full_mapper, "interval_union_candidates_", 0)
            ),
            "interval_union_predicates": int(
                getattr(
                    deployed_head_mapper if deploy_head else full_mapper,
                    "interval_union_predicates_selected_",
                    0,
                )
            ),
            "residual_allocator": getattr(
                deployed_head_mapper if deploy_head else full_mapper,
                "residual_allocator_",
                None,
            ),
            "residual_selected_families": getattr(
                deployed_head_mapper if deploy_head else full_mapper,
                "residual_selected_family_counts_",
                {},
            ),
            "proposal_objective": (
                getattr(deployed_head_mapper, "proposal_objective_", None)
                if deploy_head
                else full_mapper.proposal_objective_
            ),
            "fold_booster_fits": 4 if rule_evidence is not None else 2,
        }
        return selected_mapper

    def _auto_interactions(self, X, y, boost):
        """Select a finite symbolic feature program only after a cross-fitted win.

        The compiler may inspect only the fit side of each fold. Each candidate is
        a bounded Boolean program over input facts, not a learned embedding or a
        second predictor. We keep it only when it improves ROC-AUC on *both*
        untouched folds by a material mean margin; otherwise the certified booster
        remains the sole deployed model.
        """
        from sklearn.metrics import roc_auc_score

        from tabpvn.predicate_compiler import SymbolicPredicateMap

        X, y = np.asarray(X, float), np.asarray(y)
        source_rows = len(y)
        sampled_rows, _ = self._bounded_evidence_rows(
            y,
            getattr(SymbolicPredicateMap, "MAX_ROWS", 10_000),
            self.seed + 509,
            stratified=True,
        )
        Xg = X if sampled_rows is None else X[sampled_rows]
        yg = y if sampled_rows is None else y[sampled_rows]
        groups = self._validation_groups(sampled_rows)
        numeric_rules = bool(
            getattr(self, "_threshold_predicates", False) and SymbolicPredicateMap.auto_numeric_applicable(Xg)
        )
        if not SymbolicPredicateMap.applicable(Xg, yg, numeric_rules=numeric_rules):
            self.candidate_report_ = [gate_report("certified_boost", True, stage="predictor")]
            return None

        # Categorical one-hot levels are mutually exclusive facts, not separate
        # concepts. Their groups are passed to the compiler so it cannot waste a
        # predicate on a tautological same-category pair.
        exclusive_groups = ()
        if getattr(self, "_prep", None) is not None:
            _, meta, _ = self._reason_groups()
            exclusive_groups = tuple(group["cols"] for group in meta if group["kind"] == "onehot")

        if numeric_rules:
            return self._auto_threshold_interactions(
                Xg,
                yg,
                boost,
                exclusive_groups,
                validation_groups=groups,
            )

        # The decision gate deliberately uses the same model on both candidate
        # feature spaces. It is bounded and does not refit so zero-knob fitting
        # stays practical even when no predicate is selected.
        gate_cfg = dict(boost)
        gate_cfg["rounds"] = min(gate_cfg.get("rounds", 400), 300)
        gate_cfg["refit"] = False
        base_scores, program_scores = [], []
        try:
            splits = self._validation_splits(
                yg,
                folds=2,
                classification=True,
                groups=groups,
            )
            for tr, va in splits:
                mapper = SymbolicPredicateMap(
                    seed=self.seed,
                    exclusive_groups=exclusive_groups,
                    numeric_rules=numeric_rules,
                ).fit(Xg[tr], yg[tr])
                if not mapper.predicates:
                    self.candidate_report_ = [gate_report("certified_boost", True, stage="predictor")]
                    return None
                train_groups = None if groups is None else groups[tr]
                base = self._fit_certified(
                    self._classifier(**gate_cfg),
                    Xg[tr],
                    yg[tr],
                    groups=train_groups,
                )
                mapped_train = mapper.transform(Xg[tr])
                program_cfg = dict(gate_cfg)
                if "allowed" in program_cfg:
                    program_cfg["allowed"] = tuple(program_cfg["allowed"]) + tuple(
                        range(Xg.shape[1], mapped_train.shape[1])
                    )
                program = self._fit_certified(
                    self._classifier(**program_cfg),
                    mapped_train,
                    yg[tr],
                    groups=train_groups,
                )

                def auc(model, features, *, validation_rows=va):
                    scores = model._scores(features)
                    probs = np.exp(scores - scores.max(1, keepdims=True))
                    probs /= probs.sum(1, keepdims=True)
                    positive = model.classes_[1]
                    return float(roc_auc_score(yg[validation_rows] == positive, probs[:, 1]))

                base_scores.append(auc(base, Xg[va]))
                program_scores.append(auc(program, mapper.transform(Xg[va])))
        except Exception:
            # A default candidate must fail closed: any compiler or gate failure
            # leaves the verified baseline unchanged.
            self.candidate_report_ = [gate_report("certified_boost", True, stage="predictor")]
            return None

        deltas = np.asarray(program_scores) - np.asarray(base_scores)
        mapper = SymbolicPredicateMap(
            seed=self.seed,
            exclusive_groups=exclusive_groups,
            numeric_rules=numeric_rules,
        ).fit(Xg, yg)
        selected = bool(mapper.predicates and np.all(deltas > 0.0) and float(deltas.mean()) >= 0.003)
        self.candidate_report_ = [
            gate_report(
                "certified_boost",
                not selected,
                stage="predictor",
                mean_auc=float(np.mean(base_scores)),
            ),
            gate_report(
                "symbolic_predicate_boost",
                selected,
                stage="schema",
                mean_auc=float(np.mean(program_scores)),
                fold_auc_delta=[float(delta) for delta in deltas],
                evidence_rows=int(len(yg)),
                source_rows=int(source_rows),
                source_binary_columns=int(getattr(mapper, "source_binary_columns_", 0)),
                screened_binary_columns=int(getattr(mapper, "screened_binary_columns_", 0)),
            ),
        ]
        return mapper if selected else None

    def _auto_threshold_interactions(
        self,
        X,
        y,
        boost,
        exclusive_groups,
        *,
        validation_groups=None,
    ):
        """Select numeric clauses without displacing a verified binary program.

        Threshold clauses are an augmentation layer. The gate first measures the
        existing binary program against raw features, then measures the augmented
        program against whichever lower layer survived. This prevents a weak new
        proposer from removing a categorical program that already passed its own
        untouched-fold check.
        """
        from sklearn.metrics import roc_auc_score

        from tabpvn.predicate_compiler import SymbolicPredicateMap

        min_fold_gain = 0.002
        gate_cfg = dict(boost)
        gate_cfg["rounds"] = min(gate_cfg.get("rounds", 400), 300)
        gate_cfg["refit"] = False
        base_scores, binary_scores, threshold_scores = [], [], []
        binary_complete = threshold_complete = True

        def auc(model, features, labels):
            scores = model._scores(features)
            probs = np.exp(scores - scores.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
            positive = model.classes_[1]
            return float(roc_auc_score(labels == positive, probs[:, 1]))

        try:
            splits = self._validation_splits(
                y,
                folds=2,
                classification=True,
                groups=validation_groups,
            )
            for tr, va in splits:
                train_groups = None if validation_groups is None else validation_groups[tr]
                base = self._fit_certified(
                    self._classifier(**gate_cfg),
                    X[tr],
                    y[tr],
                    groups=train_groups,
                )
                base_scores.append(auc(base, X[va], y[va]))

                binary_mapper = SymbolicPredicateMap(
                    seed=self.seed,
                    exclusive_groups=exclusive_groups,
                ).fit(X[tr], y[tr])
                if binary_complete and binary_mapper.predicates:
                    binary_train = binary_mapper.transform(X[tr])
                    binary_cfg = dict(gate_cfg)
                    if "allowed" in binary_cfg:
                        binary_cfg["allowed"] = tuple(binary_cfg["allowed"]) + tuple(
                            range(X.shape[1], binary_train.shape[1])
                        )
                    binary = self._fit_certified(
                        self._classifier(**binary_cfg),
                        binary_train,
                        y[tr],
                        groups=train_groups,
                    )
                    binary_scores.append(auc(binary, binary_mapper.transform(X[va]), y[va]))
                else:
                    binary_complete = False

                if threshold_complete:
                    try:
                        threshold_mapper = SymbolicPredicateMap(
                            seed=self.seed,
                            exclusive_groups=exclusive_groups,
                            numeric_rules=True,
                        ).fit(X[tr], y[tr])
                        has_threshold = any(
                            predicate.kind.startswith("threshold_")
                            for predicate in threshold_mapper.predicates
                        )
                        if not has_threshold:
                            threshold_complete = False
                            continue
                        threshold_train = threshold_mapper.transform(X[tr])
                        threshold_cfg = dict(gate_cfg)
                        if "allowed" in threshold_cfg:
                            threshold_cfg["allowed"] = tuple(threshold_cfg["allowed"]) + tuple(
                                range(X.shape[1], threshold_train.shape[1])
                            )
                        threshold = self._fit_certified(
                            self._classifier(**threshold_cfg),
                            threshold_train,
                            y[tr],
                            groups=train_groups,
                        )
                        threshold_scores.append(auc(threshold, threshold_mapper.transform(X[va]), y[va]))
                    except Exception:
                        # The augmentation fails closed while the lower binary
                        # layer remains independently eligible for deployment.
                        threshold_complete = False
        except Exception:
            self.candidate_report_ = [gate_report("certified_boost", True, stage="predictor")]
            return None

        full_binary = SymbolicPredicateMap(
            seed=self.seed,
            exclusive_groups=exclusive_groups,
        ).fit(X, y)
        binary_complete = binary_complete and len(binary_scores) == len(base_scores)
        binary_deltas = (
            np.asarray(binary_scores) - np.asarray(base_scores)
            if binary_complete
            else np.asarray([], dtype=float)
        )
        binary_selected = bool(
            full_binary.predicates
            and len(binary_deltas)
            and np.all(binary_deltas > 0.0)
            and float(binary_deltas.mean()) >= 0.003
        )

        try:
            full_threshold = SymbolicPredicateMap(
                seed=self.seed,
                exclusive_groups=exclusive_groups,
                numeric_rules=True,
            ).fit(X, y)
            has_full_threshold = any(
                predicate.kind.startswith("threshold_") for predicate in full_threshold.predicates
            )
        except Exception:
            full_threshold = full_binary
            has_full_threshold = threshold_complete = False
        threshold_complete = threshold_complete and len(threshold_scores) == len(base_scores)
        threshold_deltas = (
            np.asarray(threshold_scores) - np.asarray(base_scores)
            if threshold_complete
            else np.asarray([], dtype=float)
        )
        lower_scores = binary_scores if binary_selected else base_scores
        incremental_deltas = (
            np.asarray(threshold_scores) - np.asarray(lower_scores)
            if threshold_complete
            else np.asarray([], dtype=float)
        )
        threshold_selected = bool(
            has_full_threshold
            and len(threshold_deltas)
            and np.all(threshold_deltas >= min_fold_gain)
            and float(threshold_deltas.mean()) >= 0.003
            and np.all(incremental_deltas >= min_fold_gain)
            and float(incremental_deltas.mean()) >= 0.003
        )

        selected_mapper = full_threshold if threshold_selected else full_binary if binary_selected else None
        self.candidate_report_ = [
            gate_report(
                "certified_boost",
                selected_mapper is None,
                stage="predictor",
                mean_auc=float(np.mean(base_scores)),
            ),
            gate_report(
                "symbolic_predicate_boost",
                binary_selected,
                stage="schema",
                mean_auc=float(np.mean(binary_scores)) if binary_complete else float("nan"),
                fold_auc_delta=[float(delta) for delta in binary_deltas],
            ),
            gate_report(
                "threshold_predicate_boost",
                threshold_selected,
                stage="schema",
                mean_auc=(float(np.mean(threshold_scores)) if threshold_complete else float("nan")),
                fold_auc_delta=[float(delta) for delta in threshold_deltas],
                incremental_auc_delta=[float(delta) for delta in incremental_deltas],
            ),
        ]
        return selected_mapper

    def _auto_monotone(self, X, y, boost):
        """DISCOVER per-feature monotone invariants from the data, then VERIFY them — the proposer/verifier
        pattern applied to the model's own inductive bias. Propose: a feature is a candidate if its
        quantile-binned target trend is strongly and consistently monotone (rank-correlation sign agrees,
        step consistency ≥ 0.8, effect ≥ 0.15·σ_y). Verify: enforcing the candidate set on an internal
        holdout must not worsen RMSE — if it does, the invariant isn't real for this data, so drop it (fall
        back to the single strongest candidate, then to none). Returns {col: +1/-1}. Fully automatic: the
        user supplies no feature list, direction, or threshold."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        if n < 400 or d == 0:
            return {}
        rng = np.random.default_rng(self.seed)
        groups = self._validation_groups()
        if groups is None:
            proposal_X, proposal_y = X, y
            tr = va = None
        else:
            tr, va = self._validation_splits(
                y,
                folds=2,
                classification=False,
                groups=groups,
                single=True,
                holdout=0.2,
            )[0]
            proposal_X, proposal_y = X[tr], y[tr]
        proposal_n = len(proposal_y)
        target_scale = proposal_y.std() + 1e-12
        cand, score = {}, {}
        for j in range(d):
            xj = proposal_X[:, j]
            fin = np.isfinite(xj)
            if fin.sum() < 0.5 * proposal_n:
                continue
            xv, yv = xj[fin], proposal_y[fin]
            if len(np.unique(xv)) < 6:  # not a smooth ordered axis (near-categorical)
                continue
            edges = np.unique(np.quantile(xv, np.linspace(0, 1, 11)))
            if len(edges) < 5:
                continue
            b = np.clip(np.searchsorted(edges, xv, side="right") - 1, 0, len(edges) - 2)
            m = np.array([yv[b == k].mean() if np.any(b == k) else np.nan for k in range(len(edges) - 1)])
            m = m[np.isfinite(m)]
            if len(m) < 4:
                continue
            diffs = np.diff(m)
            pos = int((diffs > 0).sum())
            neg = int((diffs < 0).sum())
            if pos == neg:
                continue
            dirn = 1 if pos > neg else -1
            consistency = max(pos, neg) / len(diffs)
            effect = (m.max() - m.min()) / target_scale
            s = min(len(xv), 4000)
            ri = rng.choice(len(xv), s, replace=False)
            rho = np.corrcoef(np.argsort(np.argsort(xv[ri])), np.argsort(np.argsort(yv[ri])))[0, 1]
            if consistency >= 0.8 and effect >= 0.15 and abs(rho) >= 0.10 and np.sign(rho) == dirn:
                cand[j] = int(dirn)
                score[j] = abs(rho) * effect
        if not cand:
            return {}
        # verify on FULL data: unlike config ranking (robust to subsampling), the "does this constraint hurt"
        # signal is only faithful on a model close to the final one — a subsampled/underfit model can't feel
        # the harm of a spurious constraint, so it would wave bad invariants through. refit off for speed.
        if groups is None:
            idx = rng.permutation(n)
            c = int(0.8 * n)
            tr, va = idx[:c], idx[c:]
        light = {k: v for k, v in boost.items() if k not in ("mono", "refit")}
        light["rounds"] = min(light.get("rounds", 400), 400)

        def _rmse(mono):
            mdl = self._fit_certified(
                AdditiveCertifiedRegressor(
                    seed=self.seed,
                    mono=(mono or None),
                    refit=False,
                    **light,
                ),
                X[tr],
                y[tr],
                groups=(None if groups is None else groups[tr]),
            )
            return float(np.sqrt(((mdl.predict(X[va]) - y[va]) ** 2).mean()))

        base = _rmse(None)
        if _rmse(cand) <= base * 1.002:  # whole set is neutral-or-better -> keep it
            return cand
        top = max(score, key=score.get)  # else salvage the single strongest invariant
        return {top: cand[top]} if _rmse({top: cand[top]}) <= base * 1.002 else {}

    def _auto_linear_leaf(self, X, y, boost):
        """Enable path-constrained LINEAR LEAVES only if they clearly help — the proposer/verifier pattern. On
        an internal 80/20 holdout, fit the deployed config with constant vs linear leaves (refit off for speed)
        and keep linear only on a clear (>0.2%) holdout-RMSE improvement. Regression only. Certificates stay
        exact either way — affine leaves generalize the constant interval / provably-constant `stability` to a
        provably-within-±ε band — but that is a real semantics change, so the gate demands a genuine gain, not
        neutrality. Returns True/False. Fully automatic (part of the self-configuration pipeline; no knob)."""
        # All-text (sparse bag-of-words) input: linear leaves are ridge models over the leaf's PATH features,
        # of which sparse text has ~none, so they never help (verified) — skip the costly gate fits entirely.
        if getattr(self, "_prep", None) is not None and self._prep.text_cols and not self._prep.num_cols:
            return False
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(y)
        if n < 400:
            return False
        groups = self._validation_groups()
        if groups is None:
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(n)
            c = int(0.8 * n)
            tr, va = idx[:c], idx[c:]
        else:
            tr, va = self._validation_splits(
                y,
                folds=2,
                classification=False,
                groups=groups,
                single=True,
                holdout=0.2,
            )[0]
        light = {k: v for k, v in boost.items() if k not in ("linear_leaf", "refit")}
        light["rounds"] = min(light.get("rounds", 400), 400)

        def _rmse(linear_leaf):
            mdl = self._fit_certified(
                AdditiveCertifiedRegressor(seed=self.seed, linear_leaf=linear_leaf, refit=False, **light),
                X[tr],
                y[tr],
                groups=(None if groups is None else groups[tr]),
            )
            return float(np.sqrt(((mdl.predict(X[va]) - y[va]) ** 2).mean()))

        return _rmse(True) < _rmse(False) * 0.998  # clear win only (mirrors the research's OOF gate margin)

    def _auto_linear_leaf_clf(self, X, y, boost):
        """Classification analog of `_auto_linear_leaf` (logit-linear leaves). Gate on 3-fold CV LOG-LOSS with
        FOLD-CONSISTENCY: fire only if linear leaves beat constant on log-loss in EVERY fold AND the mean gain
        is clear (>1%). Two design choices make this reliable where a naive gate isn't: (1) log-loss (the exact
        objective the softmax booster minimizes) is continuous/low-variance, unlike raw accuracy on a small
        holdout — an accuracy-margin gate fires on noise that doesn't transfer (it hurt heart/pima); (2) needing
        a win in ALL folds is something sampling noise can't fake, so a single lucky split can't trigger it.
        refit off for speed. Zero-downside by construction; stays off where linear leaves don't genuinely help."""
        from sklearn.metrics import log_loss

        # All-text (sparse bag-of-words): linear leaves are path-constrained -> never help (verified); skip.
        if getattr(self, "_prep", None) is not None and self._prep.text_cols and not self._prep.num_cols:
            return False
        if len(y) < 400:
            return False
        Xg, yg = X, np.asarray(y)
        groups = self._validation_groups()
        if len(yg) > _LINEAR_LEAF_GATE_MAX_N:
            from sklearn.model_selection import train_test_split

            _classes, counts = np.unique(yg, return_counts=True)
            if groups is not None:
                selected, _weight = self._bounded_evidence_rows(
                    yg,
                    _LINEAR_LEAF_GATE_MAX_N,
                    self.seed + 37,
                    stratified=True,
                )
                gate = np.arange(len(yg)) if selected is None else selected
            elif counts.min() >= 2:
                _, gate = train_test_split(
                    np.arange(len(yg)),
                    test_size=_LINEAR_LEAF_GATE_MAX_N,
                    random_state=self.seed + 37,
                    stratify=yg,
                )
            else:  # preserve a deterministic fail-closed gate for a rare class.
                gate = np.random.default_rng(self.seed + 37).choice(
                    len(yg), _LINEAR_LEAF_GATE_MAX_N, replace=False
                )
            Xg, yg = X[gate], yg[gate]
            groups = None if groups is None else groups[gate]
        light = {
            k: v
            for k, v in boost.items()
            if k
            in {
                "rounds",
                "lr",
                "depth",
                "leaf",
                "holdout",
                "patience",
                "class_weight",
                "max_leaves",
                "best_first_pair",
                "adaptive_best_first_pair",
                "allowed",
            }
        }
        light["rounds"] = min(light.get("rounds", 800), 400)

        def fold_ll(tr, va, linear_leaf):
            m = self._fit_certified(
                self._classifier(linear_leaf=linear_leaf, refit=False, **light),
                Xg[tr],
                yg[tr],
                groups=(None if groups is None else groups[tr]),
            )
            F = m._scores(Xg[va])
            e = np.exp(F - F.max(1, keepdims=True))
            return log_loss(yg[va], e / e.sum(1, keepdims=True), labels=list(m.classes_))

        # SHORT-CIRCUIT: firing needs a win in EVERY fold, so bail the instant one fold loses — the common case
        # (linear leaves don't help) stops after 1 fold instead of fitting all 3 (~3x cheaper). Decision is
        # identical to computing every fold; only the wasted work when it won't fire is removed.
        if groups is None:
            from sklearn.model_selection import KFold

            splits = list(KFold(3, shuffle=True, random_state=self.seed).split(Xg))
        else:
            try:
                splits = self._validation_splits(
                    yg,
                    folds=3,
                    classification=True,
                    groups=groups,
                )
            except ValueError:
                return False
        lin, con = [], []
        for tr, va in splits:
            a, b = fold_ll(tr, va, True), fold_ll(tr, va, False)
            if a >= b:  # lost this fold -> can't win all -> stop
                return False
            lin.append(a)
            con.append(b)
        return float(np.mean(lin)) < float(np.mean(con)) * 0.99  # all folds won; require a clear mean margin

    def verify_monotone(self, X, frac=0.5):
        """Certified-invariant check: for each discovered monotone feature, bump it in its locked direction
        and confirm no prediction moves the wrong way. Guaranteed 0 violations by construction (the additive
        model is monotone in these features); this re-checks it on real rows. Returns {col: violation_rate}."""
        if not self.monotone_:
            return {}
        X = np.asarray(X, float)
        out = {}
        for j, sgn in self.monotone_.items():
            step = frac * (np.nanstd(X[:, j]) + 1e-9)
            base = self._pred.predict(X)
            Xb = X.copy()
            Xb[:, j] += sgn * step
            out[int(j)] = float(((self._pred.predict(Xb) - base) * sgn < -1e-6).mean())
        return out

    def _fit_relational(self, base, target):
        if target is None:
            raise ValueError("relational fit needs target=<relation to learn>")
        self.base = base + [(r + "_inv", t, h) for (r, h, t) in base]  # inverses -> inverse/symmetric rules
        self.target = target
        rels = sorted({r for (r, _, _) in self.base})
        self.positives, self._rules = induce(
            self.base,
            rels,
            target,
            tau=(0.7 if self.tau is None else self.tau),  # tau=0.0 is valid, not "unset"
        )

    # ---- common API ----
    def rules(self):
        """The certified structures + their guarantee, uniform across modes."""
        if self.mode == "relational":
            return [
                {"head": self.target, "body": list(seq), "precision": p, "recall": r, "cover": s}
                for (p, r, s, seq) in self._rules
            ]
        if self.additive:  # additive ensemble: N kernel-verified region stages
            n = len(self._pred.trees_)
            return [{"engine": "additive-certified", "stages": n, "structure": "kernel-verified regions"}]
        if self.mode == "classification":
            return [
                {"class": int(c), "rule": [str(x) for x in rule], "precision": float(pr)}
                for c, rule, pr in self._pred._clf.rules
            ]
        return [
            {"region": rid, "head": h[0], "val_mae_bound": float(v)}
            for rid, (preds, h, v) in enumerate(self._pred.regions_)
        ]

    def explain(self):
        if self.mode == "relational":
            return (
                "\n".join(
                    f"{self.target}(X,Z) :- " + " , ".join(seq) + f"   [prec {p:.2f} rec {r:.2f}]"
                    for (p, r, s, seq) in self._rules
                )
                or f"(no rule ≥ τ for {self.target})"
            )
        if self.additive:
            cfg = (
                f"; self-selected config depth={self.boost_.get('depth')} lr={self.boost_.get('lr')} "
                f"leaf={self.boost_.get('leaf')}"
                + (f", pruned {len(self.pruned_)} inert features" if self.pruned_ else "")
                if self.boost_
                else ""
            )
            mono = (
                "; certified monotone in features "
                + ", ".join(f"{j}({'↑' if s > 0 else '↓'})" for j, s in self.monotone_.items())
                if self.monotone_
                else ""
            )
            sdm = (
                f"; blended with an SDM-attention memory (weight {self._sdm_w:.2f}) — an associative read over "
                "the token vectors that lifts accuracy on graded text (the booster still carries the proof)"
                if getattr(self, "_sdm", None) is not None
                else ""
            )
            posterior = (
                f"; categorical Dirichlet posterior challenger (weight {self._category_posterior_w:.2f}, "
                f"{self._category_posterior_smoothing} smoothing, "
                f"{self._category_posterior_aggregation}, cross-fitted "
                f"{self._category_posterior_permission} permission)"
                if getattr(self, "_category_posterior", None) is not None
                else ""
            )
            interval_permission = getattr(self, "_numeric_interval_permission", None)
            interval_surface = (
                "public rank admitted"
                if interval_permission == "decision_and_rank"
                else "predict_proba unchanged"
            )
            interval = (
                f"; numeric interval decision head (weight {self._numeric_interval_w:.2f}, "
                f"{self._numeric_interval_smoothing} smoothing, "
                f"{self._numeric_interval_aggregation}, cross-fitted {interval_permission} permission; "
                f"{interval_surface})"
                if getattr(self, "_numeric_interval", None) is not None
                else ""
            )
            no_signal = (
                "; dominant multiclass OOF evidence was indistinguishable from the class prior, so the "
                "probability surface uses that prior while preserving the certified class"
                if getattr(self, "_no_signal_prior", None) is not None
                else ""
            )
            return (
                f"additive certified {self.mode}: {len(self._pred.trees_)} boosted region stages; each row's "
                f"prediction = base + Σ (kernel-verified region contributions) — proof via .proof(X,row)"
                + cfg
                + mono
                + sdm
                + posterior
                + interval
                + no_signal
            )
        if self.mode == "classification":
            return self._pred._clf.rules_text()
        return "\n".join(
            f"region {i}: {h[0]} head, held-out MAE ≤ {v:.3g}"
            for i, (pr, h, v) in enumerate(self._pred.regions_)
        )

    def certify(self, X=None, sample=1500):
        """Soundness = the FOLKernel reproduces the answer exactly (float 1.0 if so). Tabular additive ->
        prediction/scores reproduced by closure over the region stages; selective -> per-rule agreement;
        relational -> re-verify the top rule's precision matches its kernel closure."""
        self._require_fitted()
        if self.mode == "relational":
            if not self._rules:
                return float("nan")
            p, r, s, seq = self._rules[0]
            pk, _, _ = _kg_verify(self.base, list(seq), self.positives)
            return float(abs(pk - p) < 1e-9)
        if self.additive:
            encoded = self._X(X)
            rep = self._pred.kernel_certify(encoded)
            verified = float(rep.get("prediction_reproduced", rep.get("scores_reproduced", float("nan"))))
            has_decision_override = (
                (
                    getattr(self, "_category_posterior", None) is not None
                    and getattr(self, "_category_posterior_permission", None) == "class_change"
                )
                or getattr(self, "_numeric_interval", None) is not None
                or (
                    getattr(self, "_affine_rank", None) is not None
                    and getattr(self, "_affine_rank_permission", None)
                    in {"decision_only", "decision_and_rank"}
                )
            )
            if verified == 1.0 and has_decision_override:
                limit = min(len(encoded), int(sample))
                batch = encoded[:limit]
                for row in range(limit):
                    evidence = self._decision_evidence_encoded(batch, row)
                    if evidence is not None and not self.check_proof(evidence):
                        return 0.0
            return verified
        return self._pred.certify(X, sample=sample)

    @staticmethod
    def check_proof(
        proof,
        base_facts=None,
        trusted_attestation_keys=None,
        *,
        artifact=None,
    ):
        """Independently verify an explicit proof artifact.

        A clean public response can also be checked when its matching
        ``proof_artifact()`` result is supplied through ``artifact=``. No
        fitted model state is required. Pass ``base_facts`` to additionally
        constrain every input leaf to a declared fact set.
        """
        if is_proof_response(proof):
            return bool(
                artifact is not None
                and proof_response_matches_artifact(proof, artifact)
                and TabPVN.check_proof(
                    artifact,
                    base_facts,
                    trusted_attestation_keys=trusted_attestation_keys,
                )
            )
        if is_proof_artifact(proof):
            payloads = machine_payloads(proof)
            return bool(
                payloads is not None
                and response_matches_machine(
                    proof,
                    trusted_attestation_keys=trusted_attestation_keys,
                )
                and all(
                    TabPVN.check_proof(
                        payload,
                        base_facts,
                        trusted_attestation_keys=trusted_attestation_keys,
                    )
                    for payload in payloads
                )
            )
        if isinstance(proof, dict) and proof.get("kind") in {
            "categorical_dirichlet_posterior",
            "categorical_dirichlet_posterior_pool",
        }:
            return CategoricalPosteriorChallenger.verify_evidence(proof)
        if isinstance(proof, dict) and proof.get("kind") in {
            "numeric_interval_dirichlet_posterior",
            "numeric_interval_dirichlet_posterior_pool",
        }:
            return NumericIntervalPosteriorChallenger.verify_evidence(proof)
        structured = verify_structured_payload(
            proof,
            lambda nested: TabPVN.check_proof(
                nested,
                base_facts,
                trusted_attestation_keys=trusted_attestation_keys,
            ),
            trusted_attestation_keys=trusted_attestation_keys,
        )
        if structured is not None:
            return structured

        from core.kernel_fol import check_proof as _check

        return bool(_check(proof, base_facts))

    def _X(self, X):
        """Apply the fitted preprocessor (raw DataFrame/categorical/missing -> numeric matrix); passthrough for
        numeric arrays (the backward-compatible fast path)."""
        if getattr(self, "_event_mode", False):
            import pandas as pd

            if not isinstance(X, pd.DataFrame):
                raise TypeError("an event-fitted model requires the original pandas DataFrame schema")
            if X.columns.duplicated().any():
                duplicates = list(X.columns[X.columns.duplicated()])
                raise ValueError(f"predict-time event DataFrame has duplicate columns: {duplicates[:5]}")
            expected = list(self.event_schema_["input_columns"])
            extra = [column for column in X.columns if column not in expected]
            missing = [column for column in expected if column not in X.columns]
            if extra or missing:
                raise ValueError(
                    f"predict-time event columns differ from fit: {len(extra)} unexpected {extra[:5]}, "
                    f"{len(missing)} missing {missing[:5]}"
                )
            X = X.reindex(columns=expected)
            if self._temporal_map is not None:
                X = self._temporal_map.augment(X)
            if self.event_schema_.get("drop_entity", True):
                X = X.drop(columns=[self.event_schema_["entity"]])
        p = getattr(self, "_prep", None)
        if p is not None:
            X = p.transform(X)
        else:
            X = np.asarray(X, float)
            if X.ndim != 2:
                raise ValueError(f"X must be a 2-D table; got shape {X.shape}")
            expected = getattr(self, "n_features_in_", None)
            if expected is not None and X.shape[1] != expected:
                raise ValueError(f"X has {X.shape[1]} columns, but the fitted model expects {expected}")
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must contain at least one row in a 2-D table")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values after preprocessing")
        interactions = getattr(self, "_interactions", None)
        return interactions.transform(X) if interactions is not None else X

    # ---- tabular: one predict; certified=True adds the built-in guarantee (bound / precision) ----
    @property
    def classes_(self):
        """Class labels (order matches `predict_proba` columns); None for regression."""
        self._require_fitted(modes={"classification", "regression"})
        return list(self._pred.classes_) if self.mode == "classification" else None

    def configure_decisions(
        self, reward=1.0, penalty=1.0, abstain_cost=0.0, prior=None, epsilon=None, delta=0.05, n_bins=None
    ):
        """Opt into the certified DECISION LAYER: after this, ordinary predict() calls
        automatically apply the base-rate (Bayes) correction and the fair-price answer/abstain, and stash a
        re-checkable certificate — no need to invoke decide()/certified_decision() per prediction.

        Standard `fit(...).predict(...)` is already fully self-configuring and returns the accuracy-optimal
        class without this layer. Supply a `penalty` (cost of a wrong answer) or a deployment `prior` only when
        you have that business knowledge. Classification only. Clear with clear_decisions()."""
        self._require_fitted(modes={"classification"}, additive=True)
        self._policy = {
            "reward": reward,
            "penalty": penalty,
            "abstain_cost": abstain_cost,
            "prior": prior,
            "epsilon": epsilon,
            "delta": delta,
            "n_bins": n_bins,
        }
        return self

    def clear_decisions(self):
        """Disable the internal decision policy and discard its last certificate."""
        self._policy = None
        self._last_decision = None
        return self

    def last_decision(self):
        """The certified-decision bundle from the most recent policy-driven predict() (re-check with
        TabPVN.verify_decision). None if no policy is configured or predict() has not run under one."""
        return self._last_decision

    def predict(self, X, abstain=False, precision=0.9, max_error=None):
        """Standard predict — returns the predictions (array). Accepts a raw DataFrame (categoricals/missing
        handled) or a numeric array. Optimizes RAW ACCURACY by default: normally probability argmax, with an
        OOF-admitted finite interval or affine decision head where it transfers. On
        imbalanced data raw and balanced accuracy trade off (a business-cost choice), so balanced-accuracy
        optimization is a deliberate mode via `predict_balanced` rather than a silent default that would lower
        raw accuracy. With `abstain=True` (calibration built) this becomes a SELECTIVE predictor:
        - CLASSIFICATION: abstain (return None) where the calibration region's Wilson accuracy lower bound is
          below `precision`; this is a population-level reliability screen, not an individual truth proof.
        - REGRESSION: abstain (return NaN) where the conformal error bound exceeds `max_error`; conformal
          coverage remains distribution-level under its calibration assumptions, not deterministic per-row truth.
        The certificate is built at fit and read with `confidence(X)`/`proof(X,row)`.

        If a decision policy is configured (configure_decisions), predict() runs the certified decision layer
        INTERNALLY: it returns the fair-price answer/abstain (labels, None where the bet is unfavourable) and
        stores a re-checkable certificate in last_decision() — the user never calls the decision methods directly."""
        self._require_fitted(modes={"classification", "regression"})
        pol = getattr(self, "_policy", None)
        if pol is not None and self.mode == "classification" and self.additive and not abstain:
            b = self.certified_decision(
                X,
                reward=pol["reward"],
                penalty=pol["penalty"],
                abstain_cost=pol["abstain_cost"],
                prior=pol["prior"],
                epsilon=pol["epsilon"],
                delta=pol["delta"],
                n_bins=pol["n_bins"],
            )
            self._last_decision = b
            preds = b["decision"]["prediction"]
            if any(p is None for p in preds):
                return preds  # abstentions present -> object array with None
            return np.asarray([p for p in preds])  # nothing abstained -> clean native dtype (unchanged)
        X = self._X(X)
        sdm = getattr(self, "_sdm", None)
        if self.mode == "regression":
            pred = self._pred.predict(X)
            if (
                sdm is not None
            ):  # blend the Nadaraya-Watson attention read (conformal bound calibrated on this)
                pred = (1.0 - self._sdm_w) * pred + self._sdm_w * sdm.read(X)
        elif any(
            member is not None
            for member in (
                getattr(self, "_smooth", None),
                getattr(self, "_category_memory", None),
                getattr(self, "_proof_path_memory", None),
                getattr(self, "_category_posterior", None),
                getattr(self, "_numeric_interval", None),
                getattr(self, "_affine_rank", None),
                sdm,
            )
        ):
            pred = self._classification_prediction(X)
        else:
            pred = self._pred.predict(X)
        if not abstain or getattr(self, "_conf", None) is None:
            return pred
        keep = self._selective_mask(X, precision, max_error)
        if keep is None:
            return pred
        if self.mode == "classification":
            return np.array([p if k else None for p, k in zip(pred, keep, strict=False)], dtype=object)
        out = pred.astype(float).copy()
        out[~keep] = np.nan  # regression: abstained rows -> NaN
        return out

    def adaptive_depth_report(self, X):
        """Audit exact booster early exits and whether public ``predict`` can use them.

        The report performs a full-score comparison, so it is a diagnostic
        rather than the hot prediction path. Probability and downstream
        class-changing members continue to require the complete score vector.
        """
        self._require_fitted(modes={"classification"}, additive=True)
        X = self._X(X)
        report = dict(self._pred.adaptive_depth_report(X))
        downstream_members = any(
            member is not None
            for member in (
                getattr(self, "_smooth", None),
                getattr(self, "_category_memory", None),
                getattr(self, "_proof_path_memory", None),
                getattr(self, "_category_posterior", None),
                getattr(self, "_numeric_interval", None),
                getattr(self, "_affine_rank", None),
                getattr(self, "_sdm", None),
            )
        )
        report["public_predict_eligible"] = not downstream_members
        report["public_predict_uses_adaptive_depth"] = bool(
            report.get("active") and report.get("selected_for_predict") and not downstream_members
        )
        if downstream_members:
            report["public_predict_reason"] = "downstream_decision_member_requires_full_scores"
        return report

    def predict_balanced(self, X):
        """Predict for BALANCED ACCURACY on imbalanced binary problems (fraud / churn / attrition, where
        catching the minority class matters more than raw accuracy). Decides with the balanced-accuracy-optimal
        threshold tuned leak-safe on out-of-fold proba at fit — a big lift in minority recall / balanced-acc at
        the cost of some raw accuracy (the trade you want when the rare class is the point). Falls back to the
        standard argmax when the tuned threshold doesn't beat it (or non-binary). Classification only."""
        self._require_fitted(modes={"classification"})
        X = self._X(X)
        cls = np.asarray(self._pred.classes_)
        probability = self._blended_proba(
            X,
            include_interval_rank=False,
            include_affine_rank=False,
        )
        if getattr(self, "_bal_thr", None) is not None:
            return np.where(probability[:, 1] >= self._bal_thr, cls[1], cls[0])
        return cls[probability.argmax(1)]

    def predict_rare(self, X):
        """Binary rare-event operating point selected without user tuning.

        The threshold maximizes importance-weighted F1 on leak-safe verifier
        predictions, so case-control sampling does not substitute its enriched
        prevalence for the source prevalence. Standard ``predict`` remains the
        certified argmax; this method is the recall/precision-oriented fraud
        default.
        """
        self._require_fitted(modes={"classification"})
        if not self.rare_event_:
            raise ValueError("predict_rare is available only when rare-event mode was selected at fit")
        X = self._X(X)
        classes = np.asarray(self._pred.classes_)
        proba = self._blended_proba(
            X,
            include_interval_rank=False,
            include_affine_rank=False,
        )
        rare_index = int(np.flatnonzero(classes == self.rare_class_)[0])
        threshold = getattr(self, "_rare_thr", None)
        if threshold is None:
            return classes[proba.argmax(1)]
        other_index = 1 - rare_index
        return np.where(proba[:, rare_index] >= threshold, classes[rare_index], classes[other_index])

    def _selective_mask(self, X, precision, max_error):
        """Rows admitted by calibration: classification -> regional precision lower bound ≥ target; regression ->
        conformal error bound ≤ max_error. X already preprocessed. None if the criterion isn't applicable."""
        if self.mode == "classification":
            return self._conf.certified_subset(X, precision)
        if max_error is None:
            return None  # regression selection needs an error tolerance
        return self._conf.bound(X) <= float(max_error)

    def select(self, X, precision=0.9, max_error=None):
        """Boolean mask of rows the verifier stands behind — the answered subset of the selective predictor.
        Classification: regional accuracy lower bound ≥ `precision`. Regression: conformal bound ≤ `max_error`.
        Requires the calibration layer (built automatically); answers everything if calibration was skipped."""
        self._require_fitted(modes={"classification", "regression"}, additive=True)
        X = self._X(X)
        if getattr(self, "_conf", None) is None:
            return np.ones(len(X), bool)
        m = self._selective_mask(X, precision, max_error)
        return np.ones(len(X), bool) if m is None else m

    def predict_proba(self, X):
        """Per-class probabilities (columns follow `classes_`) — the TEMPERATURE-CALIBRATED softmax over the
        certified additive class scores (raw boosting logits are overconfident; the temperature is fit on
        out-of-fold scores at calibration time), plus any OOF-admitted probability members. A selected finite
        categorical posterior applies its discounted likelihood ratio last. On dominant multiclass data whose
        shared OOF rank, top-1 accuracy, and log-loss are indistinguishable from the training prior, the base
        probability surface falls back to that prior while preserving every certified class. A posterior may
        correct the class only when
        its gate earned ``class_change`` permission; ``rank_only`` updates are projected back into the certified
        class. On strongly imbalanced multiclass tasks, a final OOF-selected half-prior projection may expose a
        better macro-OVO ranking while preserving that class exactly. An interval posterior selected first for
        decision accuracy may also appear here, but only after a separate material rank gain on every OOF fold.
        A compact-table global affine logit may then add an OOF-admitted rank read and can alter top-1 only
        when its stricter paired OOF gate grants decision authority.
        In either case these normalized values are a rank surface, while ``predict_calibrated_proba``, posterior,
        and pricing methods retain the calibrated pre-rank surface. Classification only."""
        self._require_fitted(modes={"classification"}, additive=True)
        return self._blended_proba(self._X(X))

    def predict_calibrated_proba(self, X):
        """Return the decision probability stack before any final prior-rank projection.

        This is the surface used by explicit deployment-prior correction,
        pricing, operating points, and decision certificates. It excludes both
        independently admitted interval ranking, global affine ranking, and
        the final multiclass prior-rank projection.
        """
        self._require_fitted(modes={"classification"}, additive=True)
        return self._blended_proba(
            self._X(X),
            include_prior_rank=False,
            include_interval_rank=False,
            include_affine_rank=False,
        )

    def _classification_prediction(self, X):
        """Return the accuracy decision while keeping its hidden update out of predict_proba."""
        probability = self._blended_proba(
            X,
            include_prior_rank=False,
            include_interval_rank=False,
            include_affine_rank=False,
        )
        interval = getattr(self, "_numeric_interval", None)
        if interval is not None:
            probability = interval.combine(probability, X, self._numeric_interval_w)
        affine_rank = getattr(self, "_affine_rank", None)
        if affine_rank is not None and getattr(self, "_affine_rank_permission", None) in {
            "decision_only",
            "decision_and_rank",
        }:
            probability = AffineLogitRead.combine(
                probability,
                affine_rank.proba(X),
                self._affine_rank_weight,
                composition=self._affine_composition,
                prior=getattr(self, "_prior_train", None),
            )
        return np.asarray(self._pred.classes_)[probability.argmax(1)]

    def _blended_proba(
        self,
        X,
        include_posterior=True,
        include_prior_rank=True,
        include_interval_rank=True,
        include_affine_rank=True,
    ):
        """Calibrated booster probabilities, blended with the gated proposer members if present: the smooth
        k-NN (small n), proof-path evidence, and the SDM-attention associative memory (text). Each is a convex
        blend applied only when its OOF gate kept it, then projected to retain the certified booster class. A
        selected categorical posterior runs last: a ``rank_only`` update is projected into the certified class,
        while a ``class_change`` update may override it because paired cross-fitted evidence earned that
        authority. A decision-admitted numeric interval posterior may refine the public surface only after an
        independent all-fold rank win. An OOF-gated global affine logit may then refine compact-table ranking;
        it preserves the immediately preceding class unless its stricter paired OOF accuracy gate granted
        ``decision_and_rank`` authority. Finally, an admitted half-prior projection can expose
        minority ranking on strongly imbalanced multiclass tasks. A shared-OOF no-signal finding first replaces an uninformative
        multiclass surface with the training prior while preserving every certified class. Both rank layers are
        bypassed by calibrated decision economics. X already preprocessed."""
        F = self._pred._scores(X) / getattr(self, "_temp", 1.0)
        e = np.exp(F - F.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        certified = p.copy()
        no_signal_prior = getattr(self, "_no_signal_prior", None)
        if no_signal_prior is not None and np.shape(no_signal_prior) == (p.shape[1],):
            p = _preserve_certified_class(
                certified,
                np.tile(no_signal_prior, (len(p), 1)),
            )
        if getattr(self, "_smooth", None) is not None:
            p = (1.0 - self._smooth_w) * p + self._smooth_w * self._smooth.proba(X)
        if getattr(self, "_category_memory", None) is not None:
            # The OOF gate evaluated category evidence after the projected
            # smooth read, so reproduce that exact certified-class geometry.
            p = _preserve_certified_class(certified, p)
            p = (1.0 - self._category_memory_w) * p + self._category_memory_w * self._category_memory.proba(X)
        if getattr(self, "_proof_path_memory", None) is not None:
            # The path gate challenges the residual geometry after the local
            # and categorical members, matching the OOF composition above.
            p = _preserve_certified_class(
                certified,
                (1.0 - self._proof_path_memory_w) * p
                + self._proof_path_memory_w * self._proof_path_memory.proba(X),
            )
        if getattr(self, "_sdm", None) is not None:  # SDM-attention read over the token vectors (graded text)
            p = (1.0 - self._sdm_w) * p + self._sdm_w * self._sdm.proba(X)
        p = _preserve_certified_class(certified, p)
        posterior = getattr(self, "_category_posterior", None)
        if include_posterior and posterior is not None:
            updated = posterior.combine(p, X, self._category_posterior_w)
            permission = getattr(self, "_category_posterior_permission", None) or "class_change"
            p = _preserve_certified_class(p, updated) if permission == "rank_only" else updated
        interval = getattr(self, "_numeric_interval", None)
        if (
            include_interval_rank
            and interval is not None
            and getattr(self, "_numeric_interval_permission", None) == "decision_and_rank"
        ):
            p = interval.combine(p, X, self._numeric_interval_w)
        affine_rank = getattr(self, "_affine_rank", None)
        affine_permission = getattr(self, "_affine_rank_permission", None)
        if (
            include_affine_rank
            and affine_rank is not None
            and affine_permission in {"rank_only", "decision_and_rank"}
        ):
            incumbent = p
            updated = AffineLogitRead.combine(
                p,
                affine_rank.proba(X),
                self._affine_rank_weight,
                composition=self._affine_composition,
                prior=getattr(self, "_prior_train", None),
            )
            p = (
                updated
                if affine_permission == "decision_and_rank"
                else _preserve_certified_class(incumbent, updated)
            )
        prior_rank_strength = getattr(self, "_prior_rank_strength", 0.0)
        if include_prior_rank and prior_rank_strength > 0.0:
            p = _multiclass_prior_rank_projection(
                p,
                self._prior_train,
                strength=prior_rank_strength,
            )
        return p

    def _posterior_evidence_encoded(self, X, row):
        row = int(row)
        query = np.asarray(X)[row : row + 1]
        fallback = None
        interval = getattr(self, "_numeric_interval", None)
        if interval is not None:
            base_probability = self._blended_proba(
                query,
                include_prior_rank=False,
                include_interval_rank=False,
                include_affine_rank=False,
            )
            fallback = interval.evidence(query, 0, base_probability, self._numeric_interval_w)
            if fallback["override"]:
                return fallback

        posterior = getattr(self, "_category_posterior", None)
        permission = getattr(self, "_category_posterior_permission", None) or "class_change"
        if posterior is None or permission != "class_change":
            return fallback
        base_probability = self._blended_proba(
            query,
            include_posterior=False,
            include_prior_rank=False,
            include_interval_rank=False,
            include_affine_rank=False,
        )
        evidence = posterior.evidence(query, 0, base_probability, self._category_posterior_w)
        return evidence if evidence["override"] else fallback or evidence

    def _affine_evidence_encoded(self, X, row):
        affine = getattr(self, "_affine_rank", None)
        if affine is None or getattr(self, "_affine_rank_permission", None) not in {
            "decision_only",
            "decision_and_rank",
        }:
            return None
        row = int(row)
        query = np.asarray(X)[row : row + 1]
        base_probability = self._blended_proba(
            query,
            include_prior_rank=False,
            include_interval_rank=False,
            include_affine_rank=False,
        )
        interval = getattr(self, "_numeric_interval", None)
        if interval is not None:
            base_probability = interval.combine(
                base_probability,
                query,
                self._numeric_interval_w,
            )
        nested = self._posterior_evidence_encoded(query, 0)
        base_proof = (
            nested if nested is not None and nested.get("override", False) else self._pred.proof(query, 0)
        )
        evidence = affine.evidence(
            query,
            0,
            base_probability,
            self._affine_rank_weight,
            base_proof=base_proof,
            composition=self._affine_composition,
            prior=getattr(self, "_prior_train", None),
            verify_base=self.check_proof,
        )
        return evidence if evidence["override"] else None

    def _decision_evidence_encoded(self, X, row):
        """Return the final class-changing evidence, if this row has one."""
        affine = self._affine_evidence_encoded(X, row)
        return affine if affine is not None else self._posterior_evidence_encoded(X, row)

    def posterior_evidence(self, X, row):
        """Finite category/interval facts and arithmetic behind a correction.

        Returns None when the OOF gate rejected class-changing authority or
        selected only a rank refinement. Numeric interval evidence always has
        independently admitted decision authority; the same posterior reaches
        ``predict_proba`` only after its separate all-fold rank gate. The returned
        record is independently re-checkable.
        """
        self._require_fitted(modes={"classification"}, additive=True)
        return self._posterior_evidence_encoded(self._X(X), row)

    def affine_evidence(self, X, row):
        """Explicit affine arithmetic for an OOF-authorized class override."""
        self._require_fitted(modes={"classification"}, additive=True)
        return self._affine_evidence_encoded(self._X(X), row)

    @staticmethod
    def verify_posterior_evidence(evidence):
        if not isinstance(evidence, dict):
            return False
        if evidence.get("kind") in {
            "numeric_interval_dirichlet_posterior",
            "numeric_interval_dirichlet_posterior_pool",
        }:
            return NumericIntervalPosteriorChallenger.verify_evidence(evidence)
        return CategoricalPosteriorChallenger.verify_evidence(evidence)

    @staticmethod
    def verify_decision_evidence(evidence):
        """Verify any supported class-changing evidence record."""
        if not isinstance(evidence, dict):
            return False
        if evidence.get("kind") == "affine_logit_decision":
            return AffineLogitRead.verify_evidence(
                evidence,
                verify_base=TabPVN.check_proof,
            )
        return TabPVN.verify_posterior_evidence(evidence)

    def confidence(self, X):
        """Per-row certificate, always available (built at fit) — regression: the conformal error bound
        (holds w.p. ≥ 1−α); classification: the region's precision lower bound. None if calibration was
        skipped (too few rows or certify=False)."""
        self._require_fitted(modes={"classification", "regression"})
        if getattr(self, "_conf", None) is None:
            return None
        X = self._X(X)
        return self._conf.bound(X) if self.mode == "regression" else self._conf.certified_precision(X)

    def posterior(self, X, prior):
        """BAYESIAN base-rate correction: re-weight the model's calibrated probabilities to a DEPLOYMENT class
        prior (Bayes' theorem / prior-probability shift). A classifier calibrated at its training base rate is
        overconfident when the class is rarer in the field — the disease-test trap. `prior` is a {class: prob}
        dict or a sequence aligned to classes_ (the deployment base rate). Returns the corrected per-class
        posteriors, exact under prior shift (only p(y) changes). Re-check with tabpvn.bayes.check_prior_shift.
        HONEST: Bayes updates a prior, it cannot set one — a wrong deployment prior yields a wrong posterior."""
        self._require_fitted(modes={"classification"}, additive=True)
        from tabpvn import bayes

        classes = self._pred.classes_
        pd = bayes.as_prior(prior, classes)
        pt = (
            self._prior_train
            if getattr(self, "_prior_train", None) is not None
            else np.full(len(classes), 1.0 / len(classes))
        )
        calibrated = self.predict_calibrated_proba(X)
        return bayes.prior_shift(calibrated, pt, pd)

    def decide(self, X, reward=1.0, penalty=1.0, abstain_cost=0.0, prior=None):
        """FAIR-PRICE selective prediction: treat each answer as an OPTION and exercise (answer) only when it is a
        favourable bet. Given a `reward` for a correct answer, a `penalty` for a wrong one, and a small
        `abstain_cost`, the break-even confidence (fair strike) is p* = (penalty - abstain_cost)/(reward+penalty);
        the row is answered iff its confidence >= p*, the expected-value-optimal rule (Chow's rule). This turns the
        abstention threshold into a quantity DERIVED from the cost of being wrong. With `prior` given (a deployment
        base rate), the decision is taken on the BAYESIAN posterior (base-rate corrected) rather than the raw
        proba — Bayes fixes the probability, the fair strike acts on it. Returns {strike, confidence, answer,
        prediction (label or None), expected_value, payoff}; classification only. Re-check with
        TabPVN.check_decision. Pair with no_arbitrage_certificate() to confirm the prices are fair."""
        self._require_fitted(modes={"classification"}, additive=True)
        from tabpvn import pricing

        P = self.posterior(X, prior) if prior is not None else self.predict_calibrated_proba(X)
        conf = P.max(1)
        idx = P.argmax(1)
        classes = self._pred.classes_
        actions, strike, evs = pricing.decide(conf, reward, penalty, abstain_cost)
        preds = np.array(
            [classes[i] if a else None for i, a in zip(idx, actions, strict=False)], dtype=object
        )
        return {
            "strike": strike,
            "confidence": conf,
            "answer": np.array(actions),
            "prediction": preds,
            "expected_value": np.array(evs),
            "payoff": {"reward": reward, "penalty": penalty, "abstain_cost": abstain_cost},
        }

    @staticmethod
    def check_decision(decision):
        """Re-verify a decide() result: recompute the fair strike from the payoff and confirm every answer/abstain
        is exactly (confidence >= strike). Sound certificate for the selective-answer decision."""
        from tabpvn import pricing

        pf = decision["payoff"]
        return pricing.check_decision(
            list(decision["answer"]),
            list(decision["confidence"]),
            pf["reward"],
            pf["penalty"],
            pf["abstain_cost"],
        )

    def no_arbitrage_certificate(self, epsilon=None, delta=0.05, n_bins=None):
        """NO-ARBITRAGE (fair-price) certificate on the model's own confidences — the guarantee that makes the
        fair strike sound. Over the leak-safe OOF calibration set it bins (confidence, correct) pairs and bounds
        the expected profit of any confidence-bin betting strategy against the stated probabilities: with a
        Hoeffding slack, certified_edge = max_bin(|accuracy - confidence| + slack). The prices are certified fair
        (no arbitrage beyond `epsilon` at confidence 1-delta) iff certified_edge <= epsilon.

        SELF-CONFIGURING (no knobs to remember): `n_bins` defaults to ~one bin per 80 calibration rows (clamped
        3..15) so every bin has enough samples for a stable slack; `epsilon` defaults to twice the largest bin's
        sampling-noise slack, i.e. "no arbitrage beyond what finite-sample noise explains" — lenient on little
        data, tight on a lot, and never an arbitrary magic number. Returns the report + {holds, epsilon, auto}.
        Returns None if calibration was skipped. Re-check with TabPVN.check_no_arbitrage."""
        self._require_fitted(modes={"classification"})
        if getattr(self, "_cal_conf", None) is None:
            return None
        from tabpvn import pricing

        conf, correct = self._cal_conf
        auto_bins = n_bins is None
        if auto_bins:
            n_bins = int(np.clip(len(conf) // 80, 3, 15))  # ~80 rows/bin -> a stable Hoeffding slack
        rep = pricing.no_arbitrage_report(conf, correct, n_bins=n_bins, delta=delta)
        auto_eps = epsilon is None
        if auto_eps:  # "calibrated to within sampling noise"
            epsilon = max(rep["certified_edge"], 2.0 * max(b["slack"] for b in rep["bins"]))
        rep["holds"] = pricing.check_no_arbitrage(rep, epsilon)
        rep["epsilon"] = epsilon
        rep["auto"] = {"epsilon": auto_eps, "n_bins": auto_bins}
        return rep

    @staticmethod
    def check_no_arbitrage(certificate):
        """Re-verify a no_arbitrage_certificate: certified_edge <= epsilon."""
        from tabpvn import pricing

        return pricing.check_no_arbitrage(certificate, certificate["epsilon"])

    def certified_decision(
        self, X, reward=1.0, penalty=1.0, abstain_cost=0.0, prior=None, epsilon=None, delta=0.05, n_bins=None
    ):
        """The WHOLE from the parts: one decision bundle that composes the base-rate correction (Bayes), the
        fair-price answer/abstain (Chow's rule), and the no-arbitrage guarantee — carrying every ingredient
        needed to re-verify all three END TO END in one object. `prior` (a deployment base rate) shifts the
        probabilities before the fair strike acts on them; the no-arbitrage certificate confirms the prices are
        fair to begin with. Returns a bundle re-checkable by TabPVN.verify_decision (no model needed). The
        product shape: one certificate per decision, re-checkable by a third party. Classification only."""
        self._require_fitted(modes={"classification"}, additive=True)
        raw = self.predict_calibrated_proba(X)
        posterior = self.posterior(X, prior) if prior is not None else None
        d = self.decide(X, reward=reward, penalty=penalty, abstain_cost=abstain_cost, prior=prior)
        na = self.no_arbitrage_certificate(epsilon=epsilon, delta=delta, n_bins=n_bins)
        classes = list(self._pred.classes_)
        pt = self._prior_train.tolist() if getattr(self, "_prior_train", None) is not None else None
        pd = None
        if prior is not None:
            from tabpvn import bayes

            pd = bayes.as_prior(prior, classes).tolist()
        bundle = {
            "classes": classes,
            "raw_proba": raw,
            "posterior": posterior,
            "prior_train": pt,
            "prior_deploy": pd,
            "decision": {
                k: d[k] for k in ("strike", "confidence", "answer", "prediction", "expected_value", "payoff")
            },
            "no_arbitrage": na,
        }
        bundle["verified"] = TabPVN.verify_decision(bundle)
        return bundle

    @staticmethod
    def verify_decision(bundle, tol=1e-9):
        """Re-check a certified_decision bundle with NO model — the third-party verifier for the whole chain:
        (1) the posterior, if used, is the exact prior-shift of the raw probabilities toward the deployment prior;
        (2) the fair-price answer/abstain matches the strike on the confidences actually used; (3) the no-arbitrage
        certificate holds. Returns True iff all three re-verify."""
        import numpy as np

        from tabpvn import bayes, pricing

        used = bundle["raw_proba"]
        if bundle.get("posterior") is not None:  # (1) posterior = exact Bayes prior-shift
            recomputed = bayes.prior_shift(bundle["raw_proba"], bundle["prior_train"], bundle["prior_deploy"])
            if not np.allclose(recomputed, bundle["posterior"], atol=1e-6):
                return False
            used = bundle["posterior"]
        dec = bundle["decision"]
        if not np.allclose(
            np.asarray(used).max(1), np.asarray(dec["confidence"]), atol=1e-6
        ):  # confidence provenance
            return False
        pf = dec["payoff"]  # (2) fair-price decision matches the strike
        if not pricing.check_decision(
            list(dec["answer"]),
            list(dec["confidence"]),
            pf["reward"],
            pf["penalty"],
            pf["abstain_cost"],
            tol=tol,
        ):
            return False
        na = bundle.get("no_arbitrage")  # (3) prices certified fair
        return na is None or pricing.check_no_arbitrage(na, na["epsilon"])

    @staticmethod
    def _proof_row(row, n_rows):
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer index")
        row = int(row)
        if not 0 <= row < n_rows:
            raise IndexError(f"row index {row} is outside the table with {n_rows} rows")
        return row

    def _prediction_proof_encoded(self, X, row):
        if self.mode == "classification":
            evidence = self._decision_evidence_encoded(X, row)
            if evidence is not None and evidence["override"]:
                return evidence, evidence["prediction"], True
            booster_proof = self._pred.proof(X, row)
            prediction = self._classification_prediction(X[row : row + 1])[0]
            return booster_proof, prediction, False

        booster_proof = self._pred.proof(X, row)
        base_prediction = float(self._pred.predict(X[row : row + 1])[0])
        member = getattr(self, "_sdm", None)
        if member is None:
            return booster_proof, base_prediction, False
        member_prediction = float(member.read(X[row : row + 1])[0])
        member_weight = float(self._sdm_w)
        prediction = (1.0 - member_weight) * base_prediction + member_weight * member_prediction
        return (
            {
                "kind": "regression_blend",
                "prediction": prediction,
                "base": {
                    "kind": "certified_additive_booster",
                    "weight": 1.0 - member_weight,
                    "prediction": base_prediction,
                },
                "member": {
                    "kind": "associative_memory",
                    "weight": member_weight,
                    "prediction": member_prediction,
                },
                "base_proof": booster_proof,
            },
            prediction,
            False,
        )

    def _proof_artifact_encoded(self, X, row, attestation=None, trusted_attestation_keys=None):
        row = self._proof_row(row, len(X))
        prediction_proof, prediction, override = self._prediction_proof_encoded(X, row)
        confidence = getattr(self, "_conf", None)
        guarantee_proof = None if override or confidence is None else confidence.certify_region_kernel(X, row)
        guarantee = None
        if guarantee_proof is not None:
            key = "bound" if self.mode == "regression" else "certified_precision"
            guarantee = float(guarantee_proof[key])
        response = build_proof_artifact(
            prediction_proof,
            mode=self.mode,
            prediction=prediction,
            prediction_verified=TabPVN.check_proof(prediction_proof),
            guarantee=guarantee,
            guarantee_proof=guarantee_proof,
            guarantee_verified=(None if guarantee_proof is None else TabPVN.check_proof(guarantee_proof)),
            feature_names=getattr(self, "feature_names_", None),
            attestation=attestation,
            trusted_attestation_keys=trusted_attestation_keys,
        )
        legacy = prediction_proof if override or guarantee_proof is None else guarantee_proof
        return response, legacy

    def proof(
        self,
        X,
        row,
        *,
        raw=False,
        attestation: TargetAttestation | None = None,
        trusted_attestation_keys=None,
    ):
        """Return an implementation-neutral proof reply for one prediction.

        The default response contains user-facing reasons and assurance status,
        without model arithmetic or derivation internals. Use
        ``proof_artifact()`` for an independently checkable audit artifact. An
        optional ``TargetAttestation`` binds a separately observed outcome;
        ``raw=True`` retains the legacy low-level payload migration path.
        """
        self._require_fitted(modes={"classification", "regression"}, additive=True)
        if raw and attestation is not None:
            raise ValueError("a target attestation requires the structured proof response")
        artifact, legacy = self._proof_artifact_encoded(
            self._X(X),
            row,
            attestation=attestation,
            trusted_attestation_keys=trusted_attestation_keys,
        )
        return legacy if raw else public_proof_response(artifact)

    def proof_artifact(
        self,
        X,
        row,
        *,
        attestation: TargetAttestation | None = None,
        trusted_attestation_keys=None,
    ):
        """Return the opt-in derivation artifact for independent auditing.

        Normal applications should return ``proof()`` to users. This method is
        intended for auditors and verification services that explicitly need
        model arithmetic and derivation details.
        """
        self._require_fitted(modes={"classification", "regression"}, additive=True)
        artifact, _legacy = self._proof_artifact_encoded(
            self._X(X),
            row,
            attestation=attestation,
            trusted_attestation_keys=trusted_attestation_keys,
        )
        return artifact

    def stability(self, X, row):
        """Exact robustness certificate for a regression prediction: the box of inputs over which the answer is
        PROVABLY constant (per-feature radius), verified by the FOLKernel. Complements confidence() — how far
        inputs can move with no change vs. how wrong the value might be. Regression additive only."""
        self._require_fitted(modes={"regression"}, additive=True)
        X = self._X(X)
        _, meta, _ = self._reason_groups()
        return self._render_box(self._pred.stability_box(X, row), X[row], meta)

    def _render_box(self, res, x, meta):
        """Express a provably-constant stability box in ORIGINAL feature names (one-hot collapsed to the fixed
        `feature = level`), so the client reads which real features may vary and by how much."""
        names = self.feature_names_

        def colname(j):
            return names[j] if names is not None and j < len(names) else f"feature[{j}]"

        def numstr(lab, lo, hi):
            if lo is not None and hi is not None:
                return f"{lo:g} < {lab} ≤ {hi:g}"
            if hi is not None:
                return f"{lab} ≤ {hi:g}"
            if lo is not None:
                return f"{lab} > {lo:g}"
            return f"{lab} (fixed)"

        box = res.get("box", {})
        parts = []
        if meta is None:
            for j in sorted(box):
                parts.append(numstr(colname(j), box[j]["lo"], box[j]["hi"]))
        else:
            tcount = {}  # text: col -> #tokens held fixed over the box (collapsed; a per-token list is unreadable)
            for g in meta:
                present = [j for j in g["cols"] if j in box]
                if not present:
                    continue
                if g["kind"] == "onehot":
                    active = [
                        lv for c, lv in zip(g["cols"], g["levels"], strict=False) if c < len(x) and x[c] > 0.5
                    ]
                    parts.append(f"{g['label']} = {active[0]} (fixed)" if active else f"{g['label']} (fixed)")
                elif g["kind"] == "isna":
                    parts.append(f"{g['label']} is {'missing' if x[present[0]] > 0.5 else 'present'} (fixed)")
                elif g["kind"] == "text":
                    tcount[g["label"]] = tcount.get(g["label"], 0) + 1
                elif g["kind"] == "encoded_category":
                    parts.append(f"{g['label']} category is fixed")
                elif g["kind"] == "num":
                    value_col = g.get("value_col", g["cols"][0])
                    missing_col = g.get("missing_col")
                    if missing_col is not None and x[missing_col] > 0.5:
                        parts.append(f"{g['label']} is missing (fixed)")
                    elif value_col in box:
                        parts.append(
                            numstr(
                                g["label"],
                                box[value_col]["lo"],
                                box[value_col]["hi"],
                            )
                        )
                    else:
                        parts.append(f"{g['label']} is present (fixed)")
                else:
                    j = present[0]
                    parts.append(numstr(g["label"], box[j]["lo"], box[j]["hi"]))
            for lab, k in tcount.items():
                parts.append(f"{lab}: {k} token(s) fixed")
        res["stable_region"] = parts
        res["n_features_constant_over"] = len(parts)
        return res

    @staticmethod
    def _posterior_reason(evidence):
        numeric = str(evidence.get("kind", "")).startswith("numeric_interval_")
        conditions = []
        for condition in evidence["conditions"]:
            if condition.get("kind") != "numeric_interval":
                conditions.append(f"{condition['name']} = {condition['level']}")
                continue
            lower, upper, name = condition.get("lower"), condition.get("upper"), condition["name"]
            if lower is None:
                conditions.append(f"{name} < {upper:g}")
            elif upper is None:
                conditions.append(f"{name} >= {lower:g}")
            else:
                conditions.append(f"{lower:g} <= {name} < {upper:g}")
        return {
            "kind": "posterior_override",
            "class": evidence["prediction"],
            "conditions": evidence["conditions"],
            "conditions_readable": conditions,
            "n_features_used": len(conditions),
            "rule": (
                f"prediction is class = {evidence['prediction']} because the declared baseline "
                f"probability is updated by " + " AND ".join(conditions)
            ),
            "explanation": (
                "This is a row-level Dirichlet posterior correction backed by finite "
                + ("numeric interval" if numeric else "category")
                + " counts. "
                "A pooled correction lists every non-overlapping factor used in its sequential update; "
                "a hierarchical pair also lists both single-fact parents used for shrinkage. "
                "It is arithmetic evidence, not a claim that all other features are invariant."
            ),
            "posterior_evidence": evidence,
            "proof": evidence,
            "certified": bool(evidence["verified"]),
        }

    def _affine_reason(self, evidence):
        classes = list(evidence["classes"])
        predicted_index = classes.index(evidence["prediction"])
        model_index = int(evidence["class_order"][predicted_index])
        coefficients = np.asarray(evidence["coefficients"], dtype=float)
        values = np.asarray(evidence["input"], dtype=float)
        if len(classes) == 2:
            direction = 1.0 if model_index == 1 else -1.0
            contributions = direction * coefficients[0] * values
        else:
            contributions = coefficients[model_index] * values
        order = np.argsort(np.abs(contributions))[::-1]
        names = self.feature_names_ or [f"feature[{index}]" for index in range(len(values))]
        terms = [
            {
                "feature": str(names[int(index)]),
                "value": float(values[index]),
                "logit_contribution": float(contributions[index]),
            }
            for index in order[: min(8, len(order))]
            if abs(float(contributions[index])) > 0.0
        ]
        composition = evidence.get("composition", "arithmetic")
        if composition == "prior_ratio":
            composition_rule = (
                "the OOF-authorized normalized base * (affine / training prior)^weight composition"
            )
            composition_explanation = (
                "The proof divides the affine probability by the declared training prior, "
                "raises that likelihood ratio to the admitted weight, multiplies it by the "
                "base probability, and normalizes the result."
            )
        else:
            composition_rule = "the OOF-authorized arithmetic affine probability composition"
            composition_explanation = (
                "The proof combines base and affine probabilities using the admitted convex weight."
            )
        return {
            "kind": "affine_override",
            "class": evidence["prediction"],
            "base_class": evidence["base_prediction"],
            "composition": composition,
            "top_affine_terms": terms,
            "n_features_used": len(values),
            "rule": (
                f"prediction changes from class {evidence['base_prediction']} to "
                f"class {evidence['prediction']} after {composition_rule}"
            ),
            "explanation": (
                "The global linear logits, input values, sigmoid/softmax, composition, and final "
                f"argmax are recomputed in the attached proof. {composition_explanation} The "
                "listed terms are the largest contributions, not a minimal sufficient subset, "
                "and this verifies model execution rather than unknown ground truth."
            ),
            "affine_evidence": evidence,
            "proof": evidence,
            "certified": bool(evidence["verified"]),
        }

    def _override_reason(self, evidence):
        kind = evidence.get("kind")
        if kind == "affine_logit_decision":
            return self._affine_reason(evidence)
        return self._posterior_reason(evidence)

    def reason(self, X, row, eps=None):
        """Minimal certified SUFFICIENT REASON — the smallest set of feature conditions that ALONE force the
        answer, every other feature provably irrelevant to it (abduction, kernel-verified). This is the
        black-box differentiator: the client sees the exact rule that determines the prediction, in feature
        names, with a guarantee that nothing else could change it. The result carries a plain-English `rule`
        and `explanation`. The boosted classification path returns a sufficient sub-cell reason. A categorical
        posterior override instead returns its finite count evidence and explicitly makes no invariance claim
        about other features. An affine override returns its complete linear arithmetic and largest terms,
        likewise without claiming a minimal subset. Regression pins the value to within its own conformal
        bound. Additive only."""
        self._require_fitted(modes={"classification", "regression"}, additive=True)
        raw_row = X.iloc[row] if hasattr(X, "iloc") else None
        X = self._X(X)
        evidence = self._decision_evidence_encoded(X, row) if self.mode == "classification" else None
        if evidence is not None and evidence["override"]:
            return self._override_reason(evidence)
        col_groups, meta, widen = self._reason_groups()
        if self.mode == "classification":
            res = self._pred.sufficient_reason(X, row, groups=col_groups, widen=widen)
            return self._render_reason(res, X[row], "classification", meta, raw_row=raw_row)
        if eps is None:  # explain to the model's own guaranteed precision
            b = self._conf.bound(X) if getattr(self, "_conf", None) is not None else None
            eps = float(b[row]) if b is not None else 0.0
        res = self._pred.sufficient_reason(X, row, eps, groups=col_groups, widen=widen)
        return self._render_reason(res, X[row], "regression", meta, raw_row=raw_row)

    def _reason_groups(self):
        """Map encoded columns back to the original input features: (col_groups, meta, widen). col_groups
        partitions the encoded columns so a categorical's one-hot dummies drop/keep atomically; meta lets the
        renderer collapse them to a readable `feature = level` condition; widen is the set of numeric columns
        worth generalizing (one-hot dummies are pointless to widen). None for a raw numeric ndarray."""
        p = self._prep
        if p is None and not getattr(self, "interaction_features_", []):
            return None, None, None
        if p is None:
            n_input = self.n_input_features_
            names = self.feature_names_ or [f"feature[{j}]" for j in range(n_input)]
            col_groups = [[j] for j in range(n_input)]
            meta = [{"label": str(names[j]), "kind": "num", "cols": [j]} for j in range(n_input)]
            for offset, name in enumerate(self.interaction_features_):
                col_groups.append([n_input + offset])
                meta.append({"label": name, "kind": "derived", "cols": [n_input + offset]})
            return col_groups, meta, set(range(n_input))
        col_groups, meta, widen = [], [], set()
        numeric_width = len(p.num_cols)
        datetime_width = sum(
            p.datetime_feat[column].n_features_out_ for column in getattr(p, "datetime_cols", [])
        )
        missing_index = {
            column: numeric_width + datetime_width + offset for offset, column in enumerate(p.na_cols)
        }
        for idx, c in enumerate(p.num_cols):
            cols = [idx]
            if c in missing_index:
                cols.append(missing_index[c])
            col_groups.append(cols)
            meta.append(
                {
                    "label": str(c),
                    "source": c,
                    "kind": "num",
                    "cols": cols,
                    "value_col": idx,
                    "missing_col": missing_index.get(c),
                }
            )
            widen.add(idx)
        idx = numeric_width
        for c in getattr(p, "datetime_cols", []):
            feature_names = p.datetime_feat[c].feature_names(c)
            cols = list(range(idx, idx + len(feature_names)))
            col_groups.append(cols)
            meta.append(
                {
                    "label": str(c),
                    "source": c,
                    "kind": "datetime",
                    "cols": cols,
                    "feature_names": feature_names,
                }
            )
            idx += len(cols)
        idx += len(p.na_cols)
        for c in p.cat_cols:
            if c in p.onehot:
                cols = list(range(idx, idx + len(p.onehot[c])))
                col_groups.append(cols)
                meta.append({"label": str(c), "kind": "onehot", "cols": cols, "levels": list(p.onehot[c])})
                idx += len(cols)
            else:
                width = 1 + len(p.target_encoding.get(c, {}).get("prior", []))
                cols = list(range(idx, idx + width))
                # All encodings represent one raw high-cardinality category. Keep
                # them atomic for sufficient reasons and never propose an invalid
                # continuous recourse on an encoded category value.
                col_groups.append(cols)
                meta.append(
                    {
                        "label": str(c),
                        "source": c,
                        "kind": "encoded_category",
                        "cols": cols,
                    }
                )
                idx += width
        for (
            c
        ) in p.text_cols:  # bag-of-words token columns, LAST (matches transform/names). One group per token;
            for t in p.text_feat[c].vocab:  # binary presence -> not widened (like one-hot dummies)
                col_groups.append([idx])
                meta.append({"label": str(c), "kind": "text", "token": t, "cols": [idx]})
                idx += 1
        for name in getattr(self, "interaction_features_", []):
            col_groups.append([idx])
            # Derived facts are recomputed from the raw schema. They can explain
            # a region but are never independently actionable in recourse.
            meta.append({"label": name, "kind": "derived", "cols": [idx]})
            idx += 1
        return col_groups, meta, widen

    def _render_reason(self, res, x, mode, meta=None, raw_row=None):  # noqa: C901 - output normalization
        """Attach a plain-English `rule` and `explanation` to a sufficient-reason result — expressed in ORIGINAL
        feature names, with a categorical's one-hot dummies collapsed to a single `feature = level` condition so
        the client reads the actual determining rule, not encoded dummy columns."""
        names = self.feature_names_
        cond = {c["feature"]: c for c in res["conditions"]}  # encoded col -> {lo, hi}
        parts = []

        def box_str(label, lo, hi):
            if lo is not None and hi is not None:
                return f"{lo:g} < {label} ≤ {hi:g}"
            if hi is not None:
                return f"{label} ≤ {hi:g}"
            if lo is not None:
                return f"{label} > {lo:g}"
            return None

        if meta is None:  # raw numeric ndarray -> per-column, generic names

            def nm(j):
                return names[j] if names is not None and j < len(names) else f"feature[{j}]"

            for j in sorted(cond):
                s = box_str(nm(j), cond[j].get("lo"), cond[j].get("hi"))
                parts.append(s if s else f"{nm(j)} = {x[j]:g}")
            n_used = len(parts)
            total = len(x)
        else:
            tpos, tneg = (
                {},
                {},
            )  # text: col -> [tokens] the reason requires PRESENT / ABSENT (aggregated below)
            for g in meta:
                present = [j for j in g["cols"] if j in cond]
                if not present:
                    continue
                if g["kind"] == "onehot":
                    col2lvl = dict(zip(g["cols"], g["levels"], strict=False))
                    active = [col2lvl[j] for j in present if cond[j].get("lo") is not None]  # dummy forced =1
                    if active:
                        parts.append(f"{g['label']} = {active[0]}")  # =level implies all siblings are 0
                    else:  # only "=0" dummies -> level is excluded
                        excl = [col2lvl[j] for j in present]
                        parts.append(
                            f"{g['label']} ≠ {excl[0]}"
                            if len(excl) == 1
                            else f"{g['label']} ∉ {{{', '.join(map(str, excl))}}}"
                        )
                elif g["kind"] == "isna":
                    j = present[0]
                    missing = cond[j].get("lo") is not None  # isna flag forced =1
                    parts.append(f"{g['label']} is {'missing' if missing else 'present'}")
                elif g["kind"] == "text":  # lo set => token forced PRESENT, else ABSENT — collect per column
                    j = present[0]
                    (tpos if cond[j].get("lo") is not None else tneg).setdefault(g["label"], []).append(
                        g["token"]
                    )
                elif g["kind"] == "derived":
                    j = present[0]
                    required = cond[j].get("lo") is not None
                    parts.append(f"{g['label']} is {'true' if required else 'false'}")
                elif g["kind"] == "encoded_category":
                    value = None
                    if raw_row is not None:
                        try:
                            value = raw_row[g.get("source", g["label"])]
                        except (KeyError, IndexError, TypeError):
                            value = None
                    rendered = str(value)
                    if value is None or rendered in {"nan", "<NA>", "None"}:
                        parts.append(f"{g['label']} is missing")
                    else:
                        parts.append(f"{g['label']} = {rendered}")
                elif g["kind"] == "num":
                    value_col = g.get("value_col", g["cols"][0])
                    missing_col = g.get("missing_col")
                    missing = missing_col is not None and x[missing_col] > 0.5
                    if missing:
                        parts.append(f"{g['label']} is missing")
                        continue
                    if value_col not in cond:
                        parts.append(f"{g['label']} is present")
                        continue
                    s = box_str(
                        g["label"],
                        cond[value_col].get("lo"),
                        cond[value_col].get("hi"),
                    )
                    presence = missing_col is not None and missing_col in cond
                    if s and presence:
                        parts.append(f"{g['label']} is present with {s}")
                    else:
                        parts.append(s if s else f"{g['label']} = {x[value_col]:g}")
                elif g["kind"] == "datetime":
                    labels = dict(zip(g["cols"], g["feature_names"], strict=True))
                    for j in present:
                        label = labels[j]
                        if label.endswith("__datetime_isna"):
                            missing = cond[j].get("lo") is not None
                            parts.append(f"{g['label']} is {'missing' if missing else 'present'}")
                            continue
                        s = box_str(label, cond[j].get("lo"), cond[j].get("hi"))
                        parts.append(s if s else f"{label} = {x[j]:g}")
                else:  # frequency or other scalar encoding
                    j = present[0]
                    lab = g["label"] + (" (frequency)" if g["kind"] == "freq" else "")
                    s = box_str(lab, cond[j].get("lo"), cond[j].get("hi"))
                    parts.append(s if s else f"{g['label']} = {x[j]:g}")
            # emit text conditions readably: the PRESENT words (the signal) in full; the many ABSENT ones that
            # abduction needs to exclude other classes are collapsed to a count so the rule stays legible.
            for lab in list(tpos) + [c for c in tneg if c not in tpos]:
                if tpos.get(lab):
                    parts.append(f"{lab} contains {{{', '.join(sorted(tpos[lab]))}}}")
                neg = tneg.get(lab, [])
                if neg:
                    parts.append(
                        f"{lab} excludes {{{', '.join(sorted(neg))}}}"
                        if len(neg) <= 6
                        else f"{lab} excludes {len(neg)} other keywords"
                    )
            n_used = len(parts)
            total = len(self._prep.input_cols) if self._prep is not None else self.n_input_features_

        if mode == "classification":
            outcome = f"class = {res['class']}"
            head = (
                f"prediction is {outcome} BECAUSE "
                if parts
                else f"prediction is {outcome} regardless of any feature"
            )
        else:
            a, b = res.get("certified_band", (None, None))
            band = f" (calibrated interval [{a:g}, {b:g}])" if a is not None else ""
            outcome = f"value ≈ {res.get('prediction', '')}{band}"
            head = (
                f"predicted {outcome} BECAUSE " if parts else f"predicted {outcome} regardless of any feature"
            )
        res["rule"] = head + " AND ".join(parts)
        res["conditions_readable"] = parts
        res["n_features_used"] = n_used
        certified = res.get("certified", res.get("margin", 1.0) > 0)
        irrelevant = max(total - n_used, 0)  # every OTHER original input feature is provably irrelevant
        res["explanation"] = (
            f"These {n_used} condition(s) ALONE determine the prediction — the remaining {irrelevant} input "
            f"feature(s) are provably irrelevant (any value leaves the prediction unchanged)."
            + ("" if certified else " [NOTE: not certified — a tie at this point.]")
        )
        res["certified"] = bool(certified)
        return res

    def reason_text(self, X, row, with_explanation=True):
        """Convenience: the certified sufficient reason as a single human-readable string (the `rule`, plus the
        irrelevance `explanation` unless with_explanation=False) — for logging, audit trails, or a one-call API
        response. See `reason()` for the full structured result (conditions, proof, margin/band, certified)."""
        r = self.reason(X, row)
        return r["rule"] + ("\n" + r["explanation"] if with_explanation else "")

    def _has_text(self):
        return self._prep is not None and bool(getattr(self._prep, "text_cols", []))

    def _text_robustness(self, Xenc, row, max_flips=8):
        """Robustness for TEXT features, where the continuous IQR radius is meaningless (a binary token has a
        degenerate IQR). The right notion is a Hamming radius over WORDS: (a) EXACT — check every single-word
        add/remove; if none flips the class, the prediction is certified stable to any one-word change; and
        (b) an achievable upper bound — greedily flip the word that most erodes the class margin until the
        class changes (kernel-verified), reporting how many word changes it took and which. Only the token
        (bag-of-words) columns are perturbed; all other features stay fixed."""
        _, meta, _ = self._reason_groups()
        tcols = [(g["cols"][0], g["label"], g["token"]) for g in (meta or []) if g["kind"] == "text"]
        cols = [c for c, _, _ in tcols]
        x = np.asarray(Xenc[row], float)

        def argcol(V):  # winning score-column index per row
            return self._pred._scores(V).argmax(1)

        c0 = int(argcol(x[None, :])[0])
        # (a) exact single-word-change neighborhood
        V = np.tile(x, (len(cols), 1))
        for r, j in enumerate(cols):
            V[r, j] = 0.0 if x[j] > 0.5 else 1.0
        one = argcol(V) if len(cols) else np.array([], int)
        breakers = [tcols[r] for r in range(len(cols)) if int(one[r]) != c0]
        # (b) greedy minimal multi-word flip
        cur, flips, remaining, changed = x.copy(), [], list(range(len(cols))), False
        for _ in range(max_flips):
            if not remaining:
                break
            C = np.tile(cur, (len(remaining), 1))
            for r, ti in enumerate(remaining):
                j = cols[ti]
                C[r, j] = 0.0 if cur[j] > 0.5 else 1.0
            F = self._pred._scores(C)
            win = F.argmax(1)
            hit = [r for r in range(len(remaining)) if int(win[r]) != c0]
            if hit:  # this single extra flip already changes the class -> minimal reached
                ti = remaining[hit[0]]
                cur[cols[ti]] = 0.0 if cur[cols[ti]] > 0.5 else 1.0
                flips.append(ti)
                changed = True
                break
            Fo = F.copy()
            Fo[:, c0] = -np.inf
            margin = F[:, c0] - Fo.max(1)  # pick the flip that erodes the class margin most
            b = int(np.argmin(margin))
            ti = remaining[b]
            cur[cols[ti]] = 0.0 if cur[cols[ti]] > 0.5 else 1.0
            flips.append(ti)
            remaining.remove(ti)
        words = [f'{"remove" if x[cols[ti]] > 0.5 else "add"} "{tcols[ti][2]}"' for ti in flips]
        kr = bool(self._pred.proof(cur[None, :], 0)["class"] == self._pred.predict(cur[None, :])[0])
        return {
            "class": self._pred.classes_[c0],
            "n_text_tokens": len(cols),
            "certified_stable_to_1_word_change": len(breakers) == 0,  # EXACT over all single add/removes
            "min_word_changes_to_flip": len(flips)
            if changed
            else None,  # greedy achievable bound (kernel-verified)
            "flip_words": words if changed else [],
            "kernel_reproduced": kr,
            "note": "1-word stability is exact; min_word_changes_to_flip is a kernel-verified achievable "
            "(greedy) bound. radius_iqr is omitted — undefined for binary token features.",
        }

    def robustness(self, X, row, rel=0.1, delta=None):
        """Certified classification robustness: is the predicted class PROVABLY unable to flip for any input
        within ±delta of this row (exact score-interval domination — not sampling/attacks)? Returns the class,
        whether it's certified stable at that radius, the score margin, and the largest certified radius (as a
        multiple of per-feature IQR). For TEXT models the IQR radius is undefined (binary tokens) → reports a
        WORD Hamming radius instead (see `_text_robustness`). A finite posterior override is certified for
        its row arithmetic but reports no unsupported continuous radius. Classification additive only."""
        self._require_fitted(modes={"classification"}, additive=True)
        Xe = self._X(X)
        evidence = self._decision_evidence_encoded(Xe, row)
        if evidence is not None and evidence["override"]:
            return {
                "class": evidence["prediction"],
                "certified_stable": False,
                "radius_iqr": 0.0,
                "decision_evidence": evidence,
                "note": (
                    "The decision update is re-checkable for this row, but no continuous input-radius "
                    "certificate is claimed across the class override."
                ),
            }
        if self._has_text():
            return self._text_robustness(Xe, row)
        out = self._pred.certified_robustness(Xe, row, rel=rel, delta=delta)
        out["radius_iqr"] = self._pred.robust_radius(Xe, row)
        return out

    def predict_interval(self, X, row, rel=0.1, delta=None):
        """Certified output band under bounded input uncertainty: the prediction is GUARANTEED within [min,max]
        for any input within ±delta of this row (delta defaults to rel · per-feature IQR — knob-free; pass
        delta to use known sensor precision). Exact interval propagation over the additive stages, distinct
        from the conformal label bound. Regression additive only."""
        self._require_fitted(modes={"regression"}, additive=True)
        return self._pred.predict_interval(self._X(X), row, rel=rel, delta=delta)

    def recourse(self, X, row, target=None, max_options=3):
        """Certified counterfactual recourse: minimal input change to reach a goal, each option kernel-verified.
        Regression: prediction ≤ target (a change on a certified-monotone feature is GUARANTEED to hold).
        Classification: flip the class to `target` (or any other class if target=None) via a greedy
        single-feature search over tree thresholds. `target`/goal is the business question, not a model knob.
        Additive only."""
        self._require_fitted(modes={"classification", "regression"}, additive=True)
        X = self._X(X)
        _, meta, _ = self._reason_groups()
        if self.mode == "classification":
            evidence = self._decision_evidence_encoded(X, row)
            if evidence is not None and evidence["override"]:
                satisfied = target is not None and target == evidence["prediction"]
                return {
                    "current_class": evidence["prediction"],
                    "target": target,
                    "satisfied": bool(satisfied),
                    "reachable": bool(satisfied),
                    "options": [],
                    "decision_evidence": evidence,
                    "summary": (
                        "goal already satisfied - no change needed"
                        if satisfied
                        else "no certified counterfactual is claimed across a decision override"
                    ),
                }
            res = self._pred.recourse(X, row, target=target, max_options=max_options, meta=meta)
        else:
            res = self._pred.recourse(X, row, float(target), max_options=max_options, meta=meta)
        return self._render_recourse(res)

    def _render_recourse(self, res):
        """Attach a plain-English `action` to each recourse option (original feature names; a categorical shows
        the level switch) and a `summary`, so the client reads what to change — not encoded column indices."""
        names = self.feature_names_

        def colname(j):
            return names[j] if names is not None and j < len(names) else f"feature[{j}]"

        for o in res.get("options", []):
            if o["kind"] == "categorical":
                o["feature"] = o["label"]
                frm = o["from_level"] if o["from_level"] is not None else "—"
                o["action"] = f"change {o['label']}: {frm} → {o['to_level']}"
            elif o["kind"] == "text":
                o["feature"] = o["label"]
                verb, prep = ("add", "to") if o["add"] else ("remove", "from")
                o["action"] = f'{verb} the word "{o["token"]}" {prep} {o["label"]}'
            else:
                lab = o["label"] if isinstance(o["label"], str) else colname(o["col"])
                o["feature"] = lab
                move = "decrease to ≤" if o["op"] == "≤" else "increase above"
                o["action"] = f"{lab}: {o['from']:g} → {move} {o['threshold']:g}"
        if res.get("satisfied"):
            res["summary"] = "goal already satisfied — no change needed"
        elif not res.get("reachable"):
            res["summary"] = "no single-feature change reaches the goal (a combined change may be required)"
        else:
            res["summary"] = " ; ".join(o["action"] for o in res["options"])
        return res

    def certificate(
        self,
        X,
        row,
        *,
        attestation: TargetAttestation | None = None,
        trusted_attestation_keys=None,
    ):
        """Every per-row certificate the verification system provides, out of the box in ONE call — no client
        wiring, nothing to configure. Regression: prediction + conformal error bound + exact stability box +
        minimal sufficient reason + input-robustness band. Classification: prediction + calibration-region
        precision lower bound + region proof. These verify model execution and statistical support, not unknown
        individual ground truth. A separately sourced ``TargetAttestation`` can confirm or refute the result in a
        post-outcome audit. A class override adds its re-checkable finite-count or affine evidence and withholds
        booster-only stability claims. (Recourse is separate — it needs a business target.)"""
        self._require_fitted(modes={"classification", "regression"})
        encoded = self._X(X)
        row = self._proof_row(row, len(encoded))
        raw_row = X.iloc[row] if hasattr(X, "iloc") else None
        proof_artifact, _legacy = self._proof_artifact_encoded(
            encoded,
            row,
            attestation=attestation,
            trusted_attestation_keys=trusted_attestation_keys,
        )
        primary_proof = proof_artifact["machine_proof"]["prediction"]
        evidence = (
            primary_proof
            if isinstance(primary_proof, dict)
            and primary_proof.get("kind")
            in {
                "categorical_dirichlet_posterior",
                "categorical_dirichlet_posterior_pool",
                "numeric_interval_dirichlet_posterior",
                "numeric_interval_dirichlet_posterior_pool",
                "affine_logit_decision",
            }
            else (self._decision_evidence_encoded(encoded, row) if self.mode == "classification" else None)
        )
        override = bool(evidence is not None and evidence["override"])
        conclusion = proof_artifact["conclusion"]
        guarantee_record = conclusion["guarantee"]
        guarantee = None if guarantee_record is None else float(guarantee_record["value"])
        out = {
            "prediction": conclusion["prediction"],
            "guarantee": guarantee,
            "individual_correctness": proof_artifact["claims"]["individual_correctness"],
            "proof": public_proof_response(proof_artifact),
        }
        if evidence is not None:
            out["base_prediction"] = evidence["base_prediction"]
            out["decision_evidence"] = evidence
            if evidence.get("kind") == "affine_logit_decision":
                out["affine_evidence"] = evidence
            else:
                out["posterior_evidence"] = evidence
        col_groups, meta, widen = self._reason_groups()
        if self.mode == "regression" and self.additive:
            out["stability"] = self._render_box(self._pred.stability_box(encoded, row), encoded[row], meta)
            rr = self._pred.sufficient_reason(encoded, row, guarantee or 0.0, groups=col_groups, widen=widen)
            out["sufficient_reason"] = self._render_reason(
                rr, encoded[row], "regression", meta, raw_row=raw_row
            )
            out["input_robustness"] = self._pred.predict_interval(encoded, row)
        elif self.mode == "classification" and self.additive:
            if override:
                assert evidence is not None
                out["robustness"] = {
                    "class": evidence["prediction"],
                    "certified_stable": False,
                    "radius_iqr": 0.0,
                    "note": "No input-radius claim is attached to a class override.",
                }
                out["sufficient_reason"] = self._override_reason(evidence)
            elif (
                self._has_text()
            ):  # word-Hamming robustness; the continuous IQR radius is undefined for tokens
                out["robustness"] = self._text_robustness(encoded, row)
            else:
                rob = self._pred.certified_robustness(encoded, row)
                rob["radius_iqr"] = self._pred.robust_radius(encoded, row)
                out["robustness"] = rob
            if not override:
                rr = self._pred.sufficient_reason(encoded, row, groups=col_groups, widen=widen)
                out["sufficient_reason"] = self._render_reason(
                    rr, encoded[row], "classification", meta, raw_row=raw_row
                )
        return out

    def coverage(self, X):
        self._require_fitted(modes={"classification", "regression"})
        return 1.0 if self.additive else self._pred.coverage(X)  # additive = full coverage

    # ---- relational ----
    def _rule_closures(self):
        """Per-rule (precision, support, seq, kernel, facts, provenance), computed ONCE and cached — the
        closures depend only on the fixed rules + base graph, so `derive`/`query`/`derive_certified` reuse them
        instead of recomputing a full graph closure per rule on every call (was O(rules × closure) per query)."""
        if not hasattr(self, "_rc_cache"):
            rc = []
            for p, _r, s, seq in self._rules:
                rule = (
                    ("q", "X0", f"X{len(seq)}"),
                    [(seq[i], f"X{i}", f"X{i + 1}") for i in range(len(seq))],
                )
                k = FOLKernel([rule])
                facts, prov = k.closure(self.base)
                rc.append((p, s, list(seq), k, facts, prov))
            self._rc_cache = rc
        return self._rc_cache

    def derive(self, head):
        """Apply the induced rules via the FOLKernel: the target tails derivable from `head`, with proofs."""
        self._require_fitted(modes={"relational"})
        out = set()
        for _p, _s, _seq, _k, facts, _prov in self._rule_closures():
            out |= {
                t
                for (f, h, t) in ((x[0], x[1], x[2]) for x in facts if len(x) == 3)
                if f == "q" and h == head
            }
        return out

    def query(self, h, t):
        """Is (h, target, t) entailed by an induced rule? (kernel-derived, with the rule's precision.)"""
        self._require_fitted(modes={"relational"})
        return t in self.derive(h)

    def derive_certified(self, head):
        """Per-derivation CONFIDENCE — the relational analog of certified precision. Each target tail derivable
        from `head`, tagged with a WILSON LOWER BOUND on the entailing rule's precision (support-adjusted, so a
        high-precision-but-tiny-support rule gets a conservative bound — NOT the raw in-sample fraction), plus
        the raw precision/support and a FOLKernel proof of the derivation. If several rules derive a tail, the
        one with the highest lower-bound confidence is reported. Ranked by confidence."""
        self._require_fitted(modes={"relational"})
        best = {}
        for p, s, seq, k, facts, prov in self._rule_closures():
            lb = _wilson_lb(round(p * s), s)  # support-adjusted lower bound on the rule's precision
            for f, h, t in ((x[0], x[1], x[2]) for x in facts if len(x) == 3):
                if f == "q" and h == head and (t not in best or lb > best[t][0]):
                    best[t] = (lb, p, s, list(seq), k.proof(("q", head, t), prov))
        return sorted(
            (
                {
                    "tail": t,
                    "confidence": round(lb, 3),
                    "precision": round(p, 3),
                    "support": int(s),
                    "rule": seq,
                    "proof": pf,
                }
                for t, (lb, p, s, seq, pf) in best.items()
            ),
            key=lambda o: -o["confidence"],
        )

    def query_confidence(self, h, t):
        """Certified confidence that (h, target, t) holds = the WILSON LOWER BOUND on the precision of the best
        rule deriving it (0.0 if not derivable) — a support-adjusted per-fact guarantee, not raw in-sample
        precision."""
        self._require_fitted(modes={"relational"})
        for d in self.derive_certified(h):
            if d["tail"] == t:
                return d["confidence"]
        return 0.0


_ADAPTER_EXPORTS = {"TabPVNMultiOutput", "TabPVNOrdinal", "TabPVNTextPair"}


def __getattr__(name: str):
    """Resolve legacy adapter imports without coupling the estimator to wrappers."""
    if name not in _ADAPTER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tabpvn import adapters

    value = getattr(adapters, name)
    globals()[name] = value
    return value
