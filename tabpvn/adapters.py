"""Composed estimators built from the single-output TabPVN runtime."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from tabpvn.base import TabPVN


class TabPVNMultiOutput:
    """MULTI-LABEL and MULTI-OUTPUT tasks — a composition of per-column certified `TabPVN` models over shared
    tabular input. `fit(data, Y)` takes a 2D `Y` (n×L): each column becomes its own single-output
    TabPVN, auto-detecting classification (a multi-label TAG) or regression (a multi-output SCORE) per column.
    The prediction is a per-label vector — or, for tags, a SET (`predict_sets`) — and EVERY certificate
    (precision bound, sufficient `reason`, kernel `proof`, conformal `confidence`) is available PER LABEL. The
    sound single-output engine is reused UNCHANGED, so a multi-label answer is proof-carrying iff each label's
    is (`certify` = min over labels). Our own composition — no new learner."""

    def __init__(self, seed: int = 0, alpha: float = 0.1) -> None:
        self.seed, self.alpha = seed, alpha

    def fit(self, data: Any, Y: Any) -> Self:
        import pandas as pd

        if isinstance(Y, pd.DataFrame):
            self.labels_ = list(Y.columns)
            Ymat = Y.to_numpy()
        else:
            Ymat = np.asarray(Y)
            if Ymat.ndim == 1:
                Ymat = Ymat[:, None]
            if Ymat.ndim != 2:
                raise ValueError("Y must be a one- or two-dimensional target array")
            self.labels_ = list(range(Ymat.shape[1]))
        if Ymat.shape[1] == 0:
            raise ValueError("Y must contain at least one target column")
        if len(Ymat) != len(data):
            raise ValueError("data and Y must contain the same number of rows")
        self.models_ = []
        self.is_clf_ = []
        for j in range(
            Ymat.shape[1]
        ):  # one certified single-output TabPVN per label (per-label featurization)
            if (
                pd.Series(Ymat[:, j]).dropna().nunique() < 2
            ):  # all-NaN / constant column -> clear error, not a crash
                raise ValueError(
                    f"label {self.labels_[j]!r}: target has fewer than 2 distinct non-missing "
                    "values (all-NaN or constant) — drop this column."
                )
            m = TabPVN(seed=self.seed, alpha=self.alpha).fit(data, Ymat[:, j])
            self.models_.append(m)
            self.is_clf_.append(m.mode == "classification")
        return self

    def _idx(self, label: Any) -> int:
        return self.labels_.index(label) if label in self.labels_ else int(label)

    def predict(self, X: Any) -> NDArray[Any]:
        """(n×L): per-label prediction — tag indicator (classification cols) or value (regression cols)."""
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        """(n×L): per-label positive-class probability for classification columns (NaN for regression cols)."""
        cols = []
        for m, clf in zip(self.models_, self.is_clf_, strict=True):
            if clf:
                p = m.predict_proba(X)
                cols.append(p[:, -1] if p.shape[1] == 2 else p.max(1))  # P(positive) for a binary tag
            else:
                cols.append(np.full(len(m.predict(X)), np.nan))
        return np.column_stack(cols)

    def predict_sets(self, X: Any, threshold: float = 0.5) -> list[set[Any]]:
        """Multi-label output: for each row, the SET of label names whose classification column fires
        (P(positive) ≥ threshold)."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        P = self.predict_proba(X)
        return [
            {self.labels_[j] for j in range(len(self.labels_)) if self.is_clf_[j] and P[i, j] >= threshold}
            for i in range(len(P))
        ]

    def confidence(self, X: Any) -> NDArray[np.float64]:
        """(n×L): per-label certified bound (precision bound / conformal error). NaN where calibration skipped."""
        cols = []
        for m in self.models_:
            c = m.confidence(X)
            cols.append(np.full(len(m.predict(X)), np.nan) if c is None else np.asarray(c, float))
        return np.column_stack(cols)

    def select(
        self,
        X: Any,
        precision: float = 0.9,
        max_error: float | None = None,
    ) -> NDArray[np.bool_]:
        """(n×L): per-label answered mask (the rows each label's verifier stands behind)."""
        return np.column_stack([m.select(X, precision=precision, max_error=max_error) for m in self.models_])

    def reason(self, X: Any, row: int, label: Any) -> Any:
        return self.models_[self._idx(label)].reason(X, row)

    def certificate(self, X: Any, row: int, label: Any) -> Any:
        return self.models_[self._idx(label)].certificate(X, row)

    def proof(self, X: Any, row: int, label: Any) -> Any:
        return self.models_[self._idx(label)].proof(X, row)

    def certify(self, X: Any = None) -> float:
        """Sound iff EVERY label's model is reproduced by the FOLKernel — min over labels (0.0 if any fails)."""
        return float(min(m.certify(X) for m in self.models_))


class TabPVNOrdinal:
    """ORDINAL targets — ordered categories (rating 1<2<3<…, severity low<med<high) — via the Frank-Hall
    cumulative decomposition: K-1 certified binary TabPVN classifiers, the k-th predicting P(y > classes_[k]).
    The ordered class distribution is P(y=c) = P(y>c₋₁) − P(y>c) (clamped monotone); prediction = argmax. This
    RESPECTS ORDER (unlike nominal classification) without assuming equal spacing/continuity (unlike
    regression) — best on the ordinal metrics (MAE, quadratic-weighted κ). Each threshold is a certified binary
    decision (`reason`, precision, proof); the ordinal answer is sound iff every threshold reproduces. Our own
    composition of the certified base — no new learner."""

    def __init__(self, seed: int = 0, alpha: float = 0.1) -> None:
        self.seed, self.alpha = seed, alpha

    def fit(self, data: Any, y: Any) -> Self:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("ordinal y must be one-dimensional")
        if len(y) != len(data):
            raise ValueError("data and y must contain the same number of rows")
        self.classes_ = sorted(set(y.tolist()))  # ordered categories (ascending)
        if len(self.classes_) < 3:
            raise ValueError("ordinal needs >= 3 ordered classes; use TabPVN for binary classification.")
        self.thresholds_ = self.classes_[:-1]
        self.models_ = [
            TabPVN(seed=self.seed, alpha=self.alpha).fit(data, (y > k).astype(int)) for k in self.thresholds_
        ]  # one certified binary per threshold: P(y > k)
        return self

    def _cum(self, X: Any) -> NDArray[np.float64]:
        """P(y > threshold_k) per threshold — (n × K-1)."""
        cols = []
        for m in self.models_:
            p = m.predict_proba(X)
            cols.append(p[:, -1] if p.shape[1] == 2 else p.max(1))
        return np.column_stack(cols)

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        """Ordered K-class distribution from the cumulative thresholds (clamped monotone, normalized)."""
        cum = self._cum(X)
        n, K = len(cum), len(self.classes_)
        P = np.zeros((n, K))
        prev = np.ones(n)  # P(y > class_{-1}) = 1
        for i in range(K):
            hi = cum[:, i] if i < K - 1 else np.zeros(n)  # P(y > class_i); 0 past the top class
            P[:, i] = np.clip(prev - hi, 0.0, None)
            prev = hi
        return P / (P.sum(1, keepdims=True) + 1e-12)

    def predict(self, X: Any) -> NDArray[Any]:
        return np.array(self.classes_)[self.predict_proba(X).argmax(1)]

    def reason(self, X: Any, row: int, threshold: Any = None) -> Any:
        """Certified reason from a threshold model. Default: the DECISIVE boundary (P(y>k) nearest 0.5) — the
        threshold that most determines the predicted class. Pass `threshold` (a class value or index) to pick."""
        if threshold is None:
            threshold = int(np.argmin(np.abs(self._cum(X)[row] - 0.5)))
        elif threshold in self.thresholds_:
            threshold = self.thresholds_.index(threshold)
        return self.models_[threshold].reason(X, row)

    def confidence(self, X: Any) -> NDArray[np.float64]:
        """Per-threshold calibration-region precision bound; NaN where calibration was skipped."""
        cols = []
        for m in self.models_:
            c = m.confidence(X)
            cols.append(np.full(len(m.predict(X)), np.nan) if c is None else np.asarray(c, float))
        return np.column_stack(cols)

    def certify(self, X: Any = None) -> float:
        """Sound iff EVERY threshold's binary model is reproduced by the FOLKernel — min over thresholds."""
        return float(min(m.certify(X) for m in self.models_))


__all__ = ["TabPVNMultiOutput", "TabPVNOrdinal"]
