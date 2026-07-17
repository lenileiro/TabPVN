"""Leak-safe binary operating-point calibration.

Probability calibration and decision calibration answer different questions.
This module selects optional balanced-accuracy and rare-event thresholds while
keeping their validation geometry explicit: shuffled cross-fitting for ordinary
tables and expanding past-to-future evaluation for declared event streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray

from tabpvn.validation import FutureValidation

Rows: TypeAlias = NDArray[np.int64]
Split: TypeAlias = tuple[Rows, Rows]
RareStats: TypeAlias = tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class _RareResult:
    threshold: float | None
    score: RareStats
    default_score: RareStats
    validation_mode: str
    cross_fitted: bool
    validation_folds: int
    evaluation_rows: Rows


@dataclass(frozen=True, slots=True)
class _BinaryThresholdEvidence:
    probabilities: NDArray[np.float64]
    target: NDArray[Any]
    classes: NDArray[Any]
    weights: NDArray[np.float64]
    groups: NDArray[np.int64] | None

    @classmethod
    def build(
        cls,
        probabilities: Any,
        target: Any,
        classes: Any,
        sample_weight: Any | None,
        validation_groups: Any | None,
    ) -> Self:
        proba = np.asarray(probabilities, dtype=float)
        labels = np.asarray(target)
        class_values = np.asarray(classes)
        if proba.ndim != 2 or proba.shape != (len(labels), 2):
            raise ValueError("binary threshold probabilities must have shape (n_rows, 2)")
        if class_values.ndim != 1 or len(class_values) != 2:
            raise ValueError("binary thresholds require exactly two classes")
        weights = (
            np.ones(len(labels), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if weights.ndim != 1 or len(weights) != len(labels):
            raise ValueError("sample_weight must have one value per threshold-calibration row")
        groups = None if validation_groups is None else np.asarray(validation_groups, dtype=np.int64)
        if groups is not None and (groups.ndim != 1 or len(groups) != len(labels)):
            raise ValueError("validation_groups must have one value per threshold-calibration row")
        return cls(proba, labels, class_values, weights, groups)

    @property
    def rows(self) -> Rows:
        return np.arange(len(self.target), dtype=np.int64)

    @property
    def positive(self) -> NDArray[np.bool_]:
        return self.target == self.classes[1]

    def temporal_splits(self) -> list[Split]:
        if self.groups is None:
            return []
        try:
            return FutureValidation(self.groups).expanding_splits(
                self.target,
                folds=3,
                warmup=0.4,
                min_train=max(6, int(0.25 * len(self.target))),
                min_valid=max(2, int(0.05 * len(self.target))),
                require_train_class_coverage=True,
            )
        except ValueError:
            return []

    def balanced_score(self, predicted: NDArray[np.bool_], rows: Rows) -> float:
        local_positive = self.positive[rows]
        local_prediction = predicted[rows]
        local_weight = self.weights[rows]
        positive_weight = float(local_weight[local_positive].sum())
        negative_weight = float(local_weight[~local_positive].sum())
        if positive_weight <= 0.0 or negative_weight <= 0.0:
            return -np.inf
        tpr = float(local_weight[local_positive & local_prediction].sum()) / positive_weight
        tnr = float(local_weight[~local_positive & ~local_prediction].sum()) / negative_weight
        return 0.5 * (tpr + tnr)

    def select_balanced(self, rows: Rows) -> tuple[float, float]:
        probability = self.probabilities[:, 1]
        candidates = np.unique(np.r_[0.5, np.quantile(probability[rows], np.linspace(0.001, 0.999, 257))])
        values = np.asarray([self.balanced_score(probability >= threshold, rows) for threshold in candidates])
        index = int(np.argmax(values))
        return float(candidates[index]), float(values[index])

    def balanced_threshold(self, splits: list[Split]) -> float | None:
        probability = self.probabilities[:, 1]
        if self.groups is None:
            threshold, score = self.select_balanced(self.rows)
            default_score = self.balanced_score(probability >= 0.5, self.rows)
            return threshold if score > default_score + 0.005 else None
        if not splits:
            return None
        prediction: NDArray[np.bool_] = np.zeros(len(self.target), dtype=bool)
        for train_rows, validation_rows in splits:
            threshold, _score = self.select_balanced(train_rows)
            prediction[validation_rows] = probability[validation_rows] >= threshold
        evaluation_rows = np.concatenate([valid for _train, valid in splits])
        if np.unique(self.target[evaluation_rows]).size != 2:
            return None
        score = self.balanced_score(prediction, evaluation_rows)
        default_score = self.balanced_score(probability >= 0.5, evaluation_rows)
        threshold, _score = self.select_balanced(self.rows)
        return threshold if score > default_score + 0.005 else None

    @staticmethod
    def wilson_interval(
        rate: Any,
        effective_n: float,
        z: float = 1.64,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        values = np.asarray(rate, dtype=float)
        count = max(float(effective_n), 1e-12)
        denominator = 1.0 + z * z / count
        center = values + z * z / (2.0 * count)
        half = z * np.sqrt(values * (1.0 - values) / count + z * z / (4.0 * count**2))
        return (
            np.maximum(0.0, (center - half) / denominator),
            np.minimum(1.0, (center + half) / denominator),
        )

    def rare_stats(
        self,
        actual: NDArray[np.bool_],
        predicted: NDArray[np.bool_],
        rows: Rows,
    ) -> RareStats:
        local_actual = actual[rows]
        local_prediction = predicted[rows]
        local_weight = self.weights[rows]
        true_positive = float(local_weight[local_actual & local_prediction].sum())
        false_positive = float(local_weight[~local_actual & local_prediction].sum())
        false_negative = float(local_weight[local_actual & ~local_prediction].sum())
        true_negative = float(local_weight[~local_actual & ~local_prediction].sum())
        precision = true_positive / max(true_positive + false_positive, 1e-12)
        recall = true_positive / max(true_positive + false_negative, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        false_positive_rate = false_positive / max(false_positive + true_negative, 1e-12)
        positive_weight = local_weight[local_actual]
        negative_weight = local_weight[~local_actual]
        positive_n = float(positive_weight.sum() ** 2) / max(float(np.square(positive_weight).sum()), 1e-12)
        negative_n = float(negative_weight.sum() ** 2) / max(float(np.square(negative_weight).sum()), 1e-12)
        recall_lower = float(self.wilson_interval(recall, positive_n)[0])
        fpr_upper = float(self.wilson_interval(false_positive_rate, negative_n)[1])
        prevalence = float(positive_weight.sum()) / max(float(local_weight.sum()), 1e-12)
        conservative_true_positive = prevalence * recall_lower
        conservative_false_positive = (1.0 - prevalence) * fpr_upper
        precision_lower = conservative_true_positive / max(
            conservative_true_positive + conservative_false_positive, 1e-12
        )
        conservative_f1 = 2.0 * precision_lower * recall_lower / max(precision_lower + recall_lower, 1e-12)
        return f1, precision, recall, false_positive_rate, conservative_f1

    def prior_weighted_rows(
        self,
        actual: NDArray[np.bool_],
        rows: Rows,
        *,
        preserve_source_prior: bool,
    ) -> NDArray[np.float64]:
        subset_weight = self.weights[rows].copy()
        if not preserve_source_prior:
            return subset_weight
        for value in (False, True):
            mask = actual[rows] == value
            source_total = float(self.weights[actual == value].sum())
            subset_total = float(subset_weight[mask].sum())
            if subset_total > 0.0:
                subset_weight[mask] *= source_total / subset_total
        return subset_weight

    def select_rare(
        self,
        probability: NDArray[np.float64],
        actual: NDArray[np.bool_],
        rows: Rows,
        *,
        preserve_source_prior: bool,
    ) -> float:
        local_probability = probability[rows]
        local_actual = actual[rows]
        local_weight = self.prior_weighted_rows(
            actual,
            rows,
            preserve_source_prior=preserve_source_prior,
        )
        order = np.argsort(-local_probability, kind="mergesort")
        sorted_probability = local_probability[order]
        sorted_actual = local_actual[order]
        sorted_weight = local_weight[order]
        boundaries = np.flatnonzero(np.r_[sorted_probability[:-1] != sorted_probability[1:], True])
        true_positive = np.cumsum(sorted_weight * sorted_actual)[boundaries]
        false_positive = np.cumsum(sorted_weight * ~sorted_actual)[boundaries]
        positive_total = float(local_weight[local_actual].sum())
        negative_total = float(local_weight[~local_actual].sum())
        recall = true_positive / max(positive_total, 1e-12)
        false_positive_rate = false_positive / max(negative_total, 1e-12)
        positive_weight = local_weight[local_actual]
        negative_weight = local_weight[~local_actual]
        positive_n = float(positive_weight.sum() ** 2) / max(float(np.square(positive_weight).sum()), 1e-12)
        negative_n = float(negative_weight.sum() ** 2) / max(float(np.square(negative_weight).sum()), 1e-12)
        recall_lower = self.wilson_interval(recall, positive_n)[0]
        fpr_upper = self.wilson_interval(false_positive_rate, negative_n)[1]
        prevalence = positive_total / max(positive_total + negative_total, 1e-12)
        conservative_true_positive = prevalence * recall_lower
        conservative_false_positive = (1.0 - prevalence) * fpr_upper
        precision_lower = conservative_true_positive / np.maximum(
            conservative_true_positive + conservative_false_positive, 1e-12
        )
        conservative_f1 = (
            2.0 * precision_lower * recall_lower / np.maximum(precision_lower + recall_lower, 1e-12)
        )
        point_precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
        point_f1 = 2.0 * point_precision * recall / np.maximum(point_precision + recall, 1e-12)
        thresholds = sorted_probability[boundaries]
        best_index = int(np.lexsort((thresholds, point_f1, conservative_f1))[-1])
        return float(thresholds[best_index])

    def temporal_rare_result(
        self,
        probability: NDArray[np.float64],
        actual: NDArray[np.bool_],
        splits: list[Split],
    ) -> _RareResult:
        evaluation_rows = (
            np.concatenate([valid for _train, valid in splits]) if splits else np.empty(0, dtype=np.int64)
        )
        counts = np.bincount(actual[evaluation_rows].astype(np.int8), minlength=2)
        if not splits or int(counts.min()) < 2:
            default_score = self.rare_stats(actual, probability >= 0.5, self.rows)
            return _RareResult(
                None,
                default_score,
                default_score,
                "insufficient_prequential_evidence",
                False,
                len(splits),
                np.empty(0, dtype=np.int64),
            )
        prediction: NDArray[np.bool_] = np.zeros(len(self.target), dtype=bool)
        for train_rows, validation_rows in splits:
            threshold = self.select_rare(
                probability,
                actual,
                train_rows,
                preserve_source_prior=False,
            )
            prediction[validation_rows] = probability[validation_rows] >= threshold
        score = self.rare_stats(actual, prediction, evaluation_rows)
        default_score = self.rare_stats(actual, probability >= 0.5, evaluation_rows)
        selected_threshold: float | None = self.select_rare(
            probability,
            actual,
            self.rows,
            preserve_source_prior=False,
        )
        if score[4] <= default_score[4] + 0.002:
            selected_threshold = None
        return _RareResult(
            selected_threshold,
            score,
            default_score,
            "prequential_future",
            False,
            len(splits),
            evaluation_rows,
        )

    def exchangeable_rare_result(
        self,
        probability: NDArray[np.float64],
        actual: NDArray[np.bool_],
    ) -> _RareResult:
        class_counts = np.bincount(actual.astype(np.int8), minlength=2)
        cross_fitted = int(class_counts.min()) >= 6
        if cross_fitted:
            fold_id: NDArray[np.int8] = np.empty(len(self.target), dtype=np.int8)
            rng = np.random.default_rng(17_071)
            for value in (False, True):
                rows = np.flatnonzero(actual == value)
                rng.shuffle(rows)
                fold_id[rows] = np.arange(len(rows), dtype=np.int64) % 3
            prediction: NDArray[np.bool_] = np.zeros(len(self.target), dtype=bool)
            for fold in range(3):
                train_rows = np.flatnonzero(fold_id != fold)
                validation_rows = np.flatnonzero(fold_id == fold)
                threshold = self.select_rare(
                    probability,
                    actual,
                    train_rows,
                    preserve_source_prior=True,
                )
                prediction[validation_rows] = probability[validation_rows] >= threshold
            score = self.rare_stats(actual, prediction, self.rows)
            validation_mode = "stratified_cross_fit"
        else:
            threshold = self.select_rare(
                probability,
                actual,
                self.rows,
                preserve_source_prior=True,
            )
            score = self.rare_stats(actual, probability >= threshold, self.rows)
            validation_mode = "in_sample_low_evidence"
        default_score = self.rare_stats(actual, probability >= 0.5, self.rows)
        selected_threshold: float | None = self.select_rare(
            probability,
            actual,
            self.rows,
            preserve_source_prior=True,
        )
        if score[4] <= default_score[4] + 0.002:
            selected_threshold = None
        return _RareResult(
            selected_threshold,
            score,
            default_score,
            validation_mode,
            cross_fitted,
            3 if cross_fitted else 0,
            self.rows,
        )


def fit_binary_thresholds(
    probabilities: Any,
    target: Any,
    classes: Any,
    rare_class: Any | None = None,
    sample_weight: Any | None = None,
    validation_groups: Any | None = None,
) -> tuple[float | None, float | None, dict[str, Any] | None]:
    """Fit optional binary thresholds under exchangeable or temporal evidence."""
    evidence = _BinaryThresholdEvidence.build(
        probabilities,
        target,
        classes,
        sample_weight,
        validation_groups,
    )
    splits = evidence.temporal_splits()
    balanced_threshold = evidence.balanced_threshold(splits)
    if rare_class is None:
        evaluation_rows = (
            np.concatenate([valid for _train, valid in splits]) if splits else np.empty(0, dtype=np.int64)
        )
        report = {
            "balanced_threshold": balanced_threshold,
            "validation_mode": (
                "prequential_future"
                if evidence.groups is not None and splits
                else (
                    "insufficient_prequential_evidence"
                    if evidence.groups is not None
                    else "exchangeable_full_evidence"
                )
            ),
            "validation_folds": len(splits),
            "evaluation_rows": (
                int(len(evaluation_rows)) if evidence.groups is not None else len(evidence.target)
            ),
        }
        return balanced_threshold, None, report
    rare_matches = np.flatnonzero(evidence.classes == rare_class)
    if len(rare_matches) != 1:
        raise ValueError("rare_class must identify one fitted class")
    probability = evidence.probabilities[:, int(rare_matches[0])]
    actual = evidence.target == rare_class
    result = (
        evidence.temporal_rare_result(probability, actual, splits)
        if evidence.groups is not None
        else evidence.exchangeable_rare_result(probability, actual)
    )
    report = {
        "balanced_threshold": balanced_threshold,
        "threshold": result.threshold,
        "weighted_f1": result.score[0],
        "weighted_precision": result.score[1],
        "weighted_recall": result.score[2],
        "weighted_fpr": result.score[3],
        "conservative_f1": result.score[4],
        "default_f1": result.default_score[0],
        "default_conservative_f1": result.default_score[4],
        "cross_fitted": result.cross_fitted,
        "validation_mode": result.validation_mode,
        "validation_folds": result.validation_folds,
        "evaluation_rows": int(len(result.evaluation_rows)),
    }
    return balanced_threshold, result.threshold, report


__all__ = ["fit_binary_thresholds"]
