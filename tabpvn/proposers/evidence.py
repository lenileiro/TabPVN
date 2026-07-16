"""Shared cross-fit evidence for zero-knob classification candidates.

Candidate architectures should be compared on the same rows, folds, class
ordering, metric, and source-prevalence weights.  This workspace owns that
contract and caches OOF predictions so a lower layer is fitted only once when
several bounded challengers are evaluated against it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ClassificationEvidence:
    """One candidate's aligned out-of-fold probability evidence."""

    name: str
    metric: str
    fold_scores: tuple[float, ...]
    probabilities: NDArray[np.float64]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.fold_scores))


class ClassificationEvidenceWorkspace:
    """Evaluate classification candidates on one deterministic fold geometry."""

    def __init__(
        self,
        y: ArrayLike,
        *,
        seed: int,
        folds: int = 2,
        metric: str = "roc_auc",
        positive_class: Any = None,
        sample_weight: ArrayLike | None = None,
        splits: Sequence[tuple[ArrayLike, ArrayLike]] | None = None,
    ) -> None:
        from sklearn.model_selection import StratifiedKFold

        self.y = np.asarray(y)
        self.classes = np.unique(self.y)
        if len(self.classes) < 2:
            raise ValueError("cross-fit classification evidence requires at least two classes")
        if splits is None and (folds < 2 or np.unique(self.y, return_counts=True)[1].min() < folds):
            raise ValueError("each class needs at least one row in every cross-fit fold")
        if metric not in {"roc_auc", "average_precision"}:
            raise ValueError("classification evidence metric must be roc_auc or average_precision")
        if metric == "average_precision" and len(self.classes) != 2:
            raise ValueError("average_precision evidence requires a binary target")
        self.metric = metric
        self.positive_class = self.classes[-1] if positive_class is None else positive_class
        if self.positive_class not in self.classes:
            raise ValueError("positive_class is not present in y")
        self.sample_weight = (
            np.ones(len(self.y), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if len(self.sample_weight) != len(self.y):
            raise ValueError("sample_weight must have one value per row")
        if splits is None:
            self.splits = tuple(
                StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.arange(len(self.y)), self.y)
            )
        else:
            normalized = []
            for train, valid in splits:
                train_rows = np.asarray(train, dtype=np.int64)
                valid_rows = np.asarray(valid, dtype=np.int64)
                if not len(train_rows) or not len(valid_rows):
                    raise ValueError("explicit evidence splits must have non-empty fit and validation rows")
                if np.intersect1d(train_rows, valid_rows).size:
                    raise ValueError("explicit evidence fit and validation rows must be disjoint")
                if not np.array_equal(np.unique(self.y[train_rows]), self.classes):
                    raise ValueError("explicit evidence fit rows must contain every class")
                if not np.array_equal(np.unique(self.y[valid_rows]), self.classes):
                    raise ValueError("explicit evidence validation rows must contain every class")
                normalized.append((train_rows, valid_rows))
            if not normalized:
                raise ValueError("explicit evidence splits must not be empty")
            self.splits = tuple(normalized)
        self.evidence_rows = np.unique(np.concatenate([valid for _train, valid in self.splits])).astype(
            np.int64, copy=False
        )
        self._cache: dict[str, ClassificationEvidence] = {}

    def _score(self, rows: ArrayLike, probabilities: ArrayLike) -> float:
        from sklearn.metrics import average_precision_score, roc_auc_score

        rows = np.asarray(rows, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        weights = self.sample_weight[rows]
        if self.metric == "average_precision":
            target = self.y[rows] == self.positive_class
            positive = int(np.flatnonzero(self.classes == self.positive_class)[0])
            return float(
                average_precision_score(
                    target,
                    probabilities[:, positive],
                    sample_weight=weights,
                )
            )
        if len(self.classes) == 2:
            positive = int(np.flatnonzero(self.classes == self.positive_class)[0])
            return float(
                roc_auc_score(
                    self.y[rows] == self.positive_class,
                    probabilities[:, positive],
                    sample_weight=weights,
                )
            )
        from tabpvn.trees import _multiclass_ovo_auc

        class_index = {label: index for index, label in enumerate(self.classes)}
        encoded = np.asarray([class_index[label] for label in self.y[rows]], dtype=int)
        # log(probability) is a softmax score representation of the already
        # normalized candidate probabilities. The local scorer supports source
        # weights, which sklearn's multiclass OVO implementation does not.
        scores = np.log(np.clip(probabilities, 1e-300, 1.0))
        return _multiclass_ovo_auc(scores, encoded, weights)

    def evaluate(
        self,
        name: str,
        evaluator: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> ClassificationEvidence:
        """Fit/predict each fold once and align returned class probabilities."""
        if name in self._cache:
            return self._cache[name]
        probabilities: NDArray[np.float64] = np.zeros((len(self.y), len(self.classes)), dtype=float)
        fold_scores = []
        for train, valid in self.splits:
            fold_probability, fold_classes = evaluator(train, valid)
            fold_probability = np.asarray(fold_probability, dtype=float)
            fold_classes = np.asarray(fold_classes)
            if fold_probability.shape != (len(valid), len(fold_classes)):
                raise ValueError("candidate returned an invalid fold probability shape")
            if set(fold_classes.tolist()) != set(self.classes.tolist()):
                raise ValueError("candidate fold classes do not match workspace classes")
            for source, label in enumerate(fold_classes):
                target = int(np.flatnonzero(self.classes == label)[0])
                probabilities[valid, target] = fold_probability[:, source]
            fold_scores.append(self._score(valid, probabilities[valid]))
        evidence = ClassificationEvidence(
            name=name,
            metric=self.metric,
            fold_scores=tuple(float(score) for score in fold_scores),
            probabilities=probabilities,
        )
        self._cache[name] = evidence
        return evidence

    @staticmethod
    def deltas(
        candidate: ClassificationEvidence,
        baseline: ClassificationEvidence,
    ) -> np.ndarray:
        if candidate.metric != baseline.metric:
            raise ValueError("candidate and baseline evidence metrics differ")
        if len(candidate.fold_scores) != len(baseline.fold_scores):
            raise ValueError("candidate and baseline fold counts differ")
        return np.asarray(candidate.fold_scores) - np.asarray(baseline.fold_scores)

    @classmethod
    def accepts(
        cls,
        candidate: ClassificationEvidence,
        baseline: ClassificationEvidence,
        *,
        min_fold_gain: float,
        min_mean_gain: float,
    ) -> tuple[bool, np.ndarray]:
        deltas = cls.deltas(candidate, baseline)
        selected = bool(
            len(deltas) and np.all(deltas >= min_fold_gain) and float(deltas.mean()) >= min_mean_gain
        )
        return selected, deltas
