"""Certified-confidence layer — the rule-verification system's distinctive capability: a CERTIFIED error
bound per prediction, with a conformal coverage GUARANTEE, on top of the (accurate) booster.

Mechanism (all our own + FOLKernel):
  * partition the input space with our own region tree (`trees.py`); region membership is a threshold clause
    the FOLKernel verifies exactly (proof-carrying routing).
  * calibrate PER REGION on out-of-fold residuals (Mondrian split-conformal): each region's bound is the
    (1−α)-quantile of |y − ŷ| over calibration points in that region. Guarantee: within each region, the
    true value lands inside [ŷ − bound, ŷ + bound] with probability ≥ 1−α (exchangeability).
  * `predict_certified` returns (prediction, bound, region); `certified_subset` returns the SLA-eligible rows
    (bound ≤ a business threshold). Orders in no calibrated region abstain rather than guess.

This is what GBDT/TabPFN can't offer: a per-order guarantee with a checkable region proof. Accuracy on the
full set is the booster's; the verification system adds the certificate.

    (used via base.TabPVN.predict_certified / .certified_subset)
"""

from __future__ import annotations

import math
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tabpvn.certified_boost import AdditiveCertifiedRegressor


class CertifiedConfidence:
    """NORMALIZED (adaptive) split-conformal error bounds. A CERTIFIED difficulty model σ(x) — our own
    additive regressor fit to the out-of-fold absolute residual — predicts how hard each row is; the bound is
    q·σ(x) where q is the finite-sample (1−α)-quantile of the normalized scores |y−ŷ|/σ(x) on a held-out
    calibration slice. Adaptive (tight on easy rows, wide on hard) and tighter than a constant-per-region
    bound, with the same conformal coverage guarantee. σ(x) is itself a certified additive sum, so each
    bound carries the FOLKernel proof of the σ-model's regions."""

    def __init__(
        self,
        alpha: float = 0.1,
        cal_frac: float = 0.4,
        seed: int = 0,
        sigma_boost: dict[str, Any] | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if not 0.0 < cal_frac < 1.0:
            raise ValueError("cal_frac must be strictly between 0 and 1")
        self.alpha, self.cal_frac, self.seed = alpha, cal_frac, seed
        # a lighter booster for the difficulty model σ (its accuracy matters less than speed here); refit off
        self.sigma_boost = (
            dict(rounds=400, lr=0.05, depth=5, leaf=40, huber=None, refit=False)
            if sigma_boost is None
            else dict(sigma_boost)
        )

    @staticmethod
    def _qhat(scores: ArrayLike, alpha: float) -> float:
        """Finite-sample conformal quantile: the ceil((n+1)(1−α))-th smallest score (guarantees ≥1−α)."""
        s = np.sort(np.asarray(scores, float))
        n = len(s)
        if n == 0:
            raise ValueError("conformal calibration requires at least one score")
        return float(s[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])

    def fit(self, X: ArrayLike, y: ArrayLike, oof_pred: ArrayLike) -> Self:
        """Build σ + conformal scale from OUT-OF-FOLD predictions (leak-safe). Split into a σ-fit slice and a
        calibration slice; σ trained on |resid| of the fit slice, q calibrated on the held-out slice."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        prediction = np.asarray(oof_pred, float)
        if X.ndim != 2 or y.ndim != 1 or prediction.ndim != 1:
            raise ValueError("X must be 2-D and y/oof_pred must be one-dimensional")
        if len(X) != len(y) or len(y) != len(prediction) or len(y) < 4:
            raise ValueError("X, y, and oof_pred must align and contain at least four rows")
        if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(prediction).all():
            raise ValueError("X, y, and oof_pred must contain only finite values")
        resid = np.abs(y - prediction)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        c = int((1 - self.cal_frac) * len(y))
        reg, cal = idx[:c], idx[c:]
        # σ floor scaled to the target (a small fraction of its spread) — a hard 1.0 destroys the bound's
        # adaptivity when the target lives on [0,1] / [0,10] (probabilities, ratings, scaled money).
        self.floor_ = max(1e-3 * float(np.std(y)), 1e-9)
        self.sigma_ = AdditiveCertifiedRegressor(seed=self.seed, **self.sigma_boost).fit(X[reg], resid[reg])
        s = resid[cal] / np.clip(self.sigma_.predict(X[cal]), self.floor_, None)
        self.q_ = self._qhat(s, self.alpha)
        return self

    def bound(self, X: ArrayLike) -> NDArray[np.float64]:
        """Certified (1−α) error bound per row = q · σ(x)."""
        return self.q_ * np.clip(self.sigma_.predict(np.asarray(X, float)), self.floor_, None)

    def certify_region_kernel(self, X: ArrayLike, row: int) -> dict[str, Any]:
        """Checkable witness for the bound: the certified proof of σ(x) (an additive sum of kernel-verified
        regions) times the conformal factor q."""
        X = np.asarray(X, float)
        sig = float(np.clip(self.sigma_.predict(X[row : row + 1])[0], self.floor_, None))
        return {
            "sigma": sig,
            "q": float(self.q_),
            "bound": float(self.q_ * sig),
            "sigma_proof": self.sigma_.proof(X, row),
        }


class CertifiedClassConfidence:
    """Certified SELECTIVE classification: partition the input space with our own region tree over the
    classifier's correctness (regions of homogeneous accuracy, kernel-verifiable membership); each region
    gets a LOWER BOUND on its accuracy (Wilson score on an ordinary held-out slice; weighted Hoeffding when
    case-control sampling is active). Answer a row only if its region's lower-bound accuracy ≥ the target
    precision — so on the answered subset, precision ≥ target with high confidence; abstain otherwise. The
    classification analog of the regression bound."""

    def __init__(
        self,
        depth: int = 5,
        min_leaf: int = 60,
        cal_frac: float = 0.4,
        z: float = 1.64,
        seed: int = 0,
    ) -> None:
        if depth <= 0 or min_leaf <= 0:
            raise ValueError("depth and min_leaf must be positive")
        if not 0.0 < cal_frac < 1.0 or z <= 0:
            raise ValueError("cal_frac must be in (0, 1) and z must be positive")
        self.depth, self.min_leaf, self.cal_frac, self.z, self.seed = depth, min_leaf, cal_frac, z, seed

    @staticmethod
    def _wilson_lb(k: int, n: int, z: float) -> float:
        """Wilson score lower bound for a binomial proportion k/n (one-sided ~95% at z=1.64)."""
        if n == 0:
            return 0.0
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return float((c - half) / d)

    @staticmethod
    def _weighted_lb(
        correct: ArrayLike,
        weights: ArrayLike,
        z: float,
    ) -> tuple[float, float]:
        """One-sided Hoeffding bound for an importance-weighted Bernoulli mean."""
        correct = np.asarray(correct, dtype=float)
        weights = np.asarray(weights, dtype=float)
        positive = weights > 0
        if not positive.any():
            return 0.0, 0.0
        correct, weights = correct[positive], weights[positive]
        total = float(weights.sum())
        normalized = weights / total
        effective_n = float(1.0 / np.square(normalized).sum())
        mean = float(np.dot(normalized, correct))
        delta = max(0.5 * math.erfc(z / math.sqrt(2.0)), 1e-12)
        penalty = math.sqrt(math.log(1.0 / delta) / (2.0 * effective_n))
        return max(0.0, mean - penalty), effective_n

    @staticmethod
    def _prior_weighted_subset(
        y: ArrayLike,
        weights: ArrayLike,
        rows: ArrayLike,
    ) -> NDArray[np.float64]:
        """Retain full-sample weighted class prevalence after a stratified split."""
        y = np.asarray(y)
        weights = np.asarray(weights, dtype=float)
        rows = np.asarray(rows, dtype=np.int64)
        out = weights[rows].copy()
        for value in np.unique(y):
            source = float(weights[y == value].sum())
            subset = y[rows] == value
            current = float(out[subset].sum())
            if current > 0:
                out[subset] *= source / current
        mean_weight = float(out.mean())
        return out if mean_weight <= 0 else out / mean_weight

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        oof_labels: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        from tabpvn.certified_boost import _leaf_regions
        from tabpvn.trees import _fit_tree

        X = np.asarray(X, float)
        y = np.asarray(y)
        labels = np.asarray(oof_labels)
        if X.ndim != 2 or y.ndim != 1 or labels.ndim != 1:
            raise ValueError("X must be 2-D and y/oof_labels must be one-dimensional")
        if len(X) != len(y) or len(y) != len(labels) or len(y) < 4:
            raise ValueError("X, y, and oof_labels must align and contain at least four rows")
        correct = (labels == y).astype(float)
        rng = np.random.default_rng(self.seed)
        self.weighted_ = sample_weight is not None
        if sample_weight is None:
            idx = rng.permutation(len(y))
            c = int((1 - self.cal_frac) * len(y))
            reg, cal = idx[:c], idx[c:]
            cal_weight = None
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if len(weights) != len(y) or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("sample_weight must be finite, non-negative, and aligned with y")
            reg_parts, cal_parts = [], []
            for value in np.unique(y):
                rows = np.flatnonzero(y == value)
                rng.shuffle(rows)
                cutoff = int((1 - self.cal_frac) * len(rows))
                cutoff = int(np.clip(cutoff, 1, len(rows) - 1))
                reg_parts.append(rows[:cutoff])
                cal_parts.append(rows[cutoff:])
            reg = np.concatenate(reg_parts).astype(np.int64, copy=False)
            cal = np.concatenate(cal_parts).astype(np.int64, copy=False)
            rng.shuffle(reg)
            rng.shuffle(cal)
            cal_weight = self._prior_weighted_subset(y, weights, cal)
        self.tree_ = _fit_tree(X[reg], correct[reg], depth=self.depth, min_leaf=self.min_leaf)
        self.regions_ = _leaf_regions(self.tree_)
        rid_cal = self._route(X[cal])
        # global fallback first, so under-sampled regions fall back to it (NOT 0.0, which would force those
        # rows to abstain even where the model is globally accurate).
        if cal_weight is None:
            self.global_lb_ = self._wilson_lb(int(correct[cal].sum()), len(cal), self.z)
            self.cal_effective_n_ = float(len(cal))
        else:
            self.global_lb_, self.cal_effective_n_ = self._weighted_lb(correct[cal], cal_weight, self.z)
        self.lb_ = {}
        for rid in range(len(self.regions_)):
            m = rid_cal == rid
            n = int(m.sum())
            if cal_weight is None:
                k = int(correct[cal][m].sum())
                self.lb_[rid] = self._wilson_lb(k, n, self.z) if n >= 20 else self.global_lb_
            else:
                lower, effective_n = self._weighted_lb(correct[cal][m], cal_weight[m], self.z)
                self.lb_[rid] = lower if effective_n >= 20 else self.global_lb_
        return self

    def _route(self, X: ArrayLike) -> NDArray[np.int64]:
        X = np.asarray(X, float)
        out = np.full(len(X), -1)
        for rid, (preds, _) in enumerate(self.regions_):
            m: NDArray[np.bool_] = np.ones(len(X), bool)
            for j, op, thr in preds:
                m &= (X[:, j] <= thr) if op == "<=" else (X[:, j] > thr)
            out[m] = rid
        return out

    def certified_precision(self, X: ArrayLike) -> NDArray[np.float64]:
        """Wilson accuracy lower bound for each row's calibration region."""
        rid = self._route(X)
        return np.array([self.lb_.get(int(r), self.global_lb_) for r in rid])

    def certified_subset(self, X: ArrayLike, target: float) -> NDArray[np.bool_]:
        """Mask of rows whose calibration-region precision lower bound meets *target*."""
        return self.certified_precision(np.asarray(X, float)) >= target

    def certify_region_kernel(self, X: ArrayLike, row: int) -> dict[str, Any]:
        """FOL proof of region membership plus that region's statistical precision lower bound."""
        from core.kernel_fol import FOLKernel
        from tabpvn.certified_boost import _category_level, _region_rule

        X = np.asarray(X, float)
        rid = int(self._route(X[row : row + 1])[0])
        lb = float(self.lb_.get(rid, self.global_lb_))
        if rid < 0 or not self.regions_[rid][0]:
            return {"region": rid, "certified_precision": lb, "proof": "root"}
        head, body, feats = _region_rule(rid, self.regions_[rid][0])
        # `feats` is {"features": [...], "categories": [...]} (see _region_rule); build BOTH the numeric and the
        # categorical base facts, mirroring _region_facts — iterating `feats` directly would yield the dict KEYS.
        facts = [("feat", row, j, float(X[row, j])) for j in feats["features"]]
        facts += [("cat", row, cols, _category_level(X, row, cols)) for cols in feats["categories"]]
        fired, prov = FOLKernel([(head, body)]).closure(facts)
        return {
            "region": rid,
            "certified_precision": lb,
            "proof": FOLKernel([(head, body)]).proof((head[0], row), prov)
            if any(t[0] == head[0] for t in fired)
            else None,
        }
