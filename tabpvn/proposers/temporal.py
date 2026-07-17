"""Power-aware future-window gate for causal temporal Laplace evidence."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tabpvn.certified_boost import AdditiveCertifiedClassifier, AdditiveCertifiedRegressor
from tabpvn.preprocessing import _Preprocessor
from tabpvn.proposers.base import gate_report
from tabpvn.temporal import TemporalLaplaceMap

_GATE_MAX_ROWS = 20_000
_POWER_GATE_CONTEXT_ROWS = 100_000
_POWER_GATE_MAX_ROWS = 50_000
_GATE_MAX_ROWS_PER_ENTITY = 512
_GATE_MIN_ROWS = 200
_GATE_VALID_FRACTION = 0.25
_POWER_GATE_TRAIN_FRACTION = 0.70
_POWER_GATE_FORWARD_WINDOWS = 3
_POWER_GATE_CONFIDENCE_BLOCKS = 12
_POWER_GATE_CONFIDENCE = 0.90
_CLASSIFICATION_MIN_GAIN = 0.005
_RARE_EVENT_MIN_GAIN = 0.01
_CLASSIFICATION_MIN_PRACTICAL_GAIN = 0.0005
_RARE_EVENT_MIN_PRACTICAL_GAIN = 0.001
_CLASSIFICATION_MAX_ACCURACY_LOSS = 0.002
_RARE_EVENT_MAX_AUC_LOSS = 0.002
_REGRESSION_MIN_RELATIVE_GAIN = 0.005
_REGRESSION_MIN_PRACTICAL_GAIN = 0.001
_REGRESSION_MAX_FORWARD_LOSS = 0.01

TemporalTask = Literal["classification", "regression"]


class TemporalEvidenceChallenger:
    """Compare a raw event schema with causal history on future windows.

    The deployment gate fits one baseline/candidate pair on a bounded past and
    evaluates it on several disjoint future windows. Equal timestamps remain
    atomic, and a deterministic time-block jackknife measures whether a small
    practical gain is distinguishable from temporal sampling noise.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def _window_rows(
        self,
        frame: Any,
        entity_codes: NDArray[np.int64],
        timestamps_ns: NDArray[np.int64],
        *,
        max_rows: int,
    ) -> NDArray[np.int64] | None:
        """Select bounded recent entity histories and return timestamp order."""
        row_ids: NDArray[np.int64] = np.arange(len(frame), dtype=np.int64)
        entity_order = np.lexsort((row_ids, timestamps_ns, entity_codes)).astype(
            np.int64,
            copy=False,
        )
        ordered_entities = entity_codes[entity_order]
        starts = np.r_[0, np.flatnonzero(ordered_entities[1:] != ordered_entities[:-1]) + 1]
        ends = np.r_[starts[1:], len(entity_order)]
        eligible = [
            group
            for group, (start, end) in enumerate(zip(starts, ends, strict=True))
            if end - start >= 3 and timestamps_ns[entity_order[start]] < timestamps_ns[entity_order[end - 1]]
        ]
        rng = np.random.default_rng(self.seed + 91_771)
        rng.shuffle(eligible)
        selected = []
        remaining = max_rows
        for group in eligible:
            rows = entity_order[starts[group] : ends[group]]
            take = min(len(rows), _GATE_MAX_ROWS_PER_ENTITY, remaining)
            candidate = rows[-take:]
            if timestamps_ns[candidate[0]] == timestamps_ns[candidate[-1]]:
                continue
            selected.append(candidate)
            remaining -= take
            if remaining == 0:
                break
        if not selected:
            return None
        selected_rows = np.concatenate(selected).astype(np.int64, copy=False)
        return selected_rows[np.lexsort((selected_rows, timestamps_ns[selected_rows]))].astype(
            np.int64,
            copy=False,
        )

    def _window_split(
        self,
        frame: Any,
        entity_codes: NDArray[np.int64],
        timestamps_ns: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
        order = self._window_rows(
            frame,
            entity_codes,
            timestamps_ns,
            max_rows=_GATE_MAX_ROWS,
        )
        if order is None or len(order) < _GATE_MIN_ROWS:
            return None
        ordered_times = timestamps_ns[order]
        requested = int((1.0 - _GATE_VALID_FRACTION) * len(order))
        boundary = ordered_times[requested]
        cut = int(np.searchsorted(ordered_times, boundary, side="left"))
        minimum_valid = max(40, int(0.10 * len(order)))
        if cut < 100 or len(order) - cut < minimum_valid:
            return None
        return order[:cut], order[cut:]

    @staticmethod
    def _atomic_partitions(
        order: NDArray[np.int64],
        timestamps_ns: NDArray[np.int64],
        requested: int,
    ) -> tuple[NDArray[np.int64], ...]:
        """Split timestamp-ordered rows into balanced, timestamp-atomic parts."""
        if len(order) == 0:
            return ()
        ordered_times = timestamps_ns[order]
        cuts: list[int] = []
        for part in range(1, requested):
            target = int(round(part * len(order) / requested))
            if target <= 0 or target >= len(order):
                continue
            cut = int(np.searchsorted(ordered_times, ordered_times[target], side="left"))
            if cut > 0 and (not cuts or cut > cuts[-1]):
                cuts.append(cut)
        boundaries = (0, *cuts, len(order))
        return tuple(
            order[start:end]
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            if end > start
        )

    def _power_window_split(
        self,
        frame: Any,
        _entity_codes: NDArray[np.int64],
        timestamps_ns: NDArray[np.int64],
    ) -> (
        tuple[
            NDArray[np.int64],
            NDArray[np.int64],
            tuple[NDArray[np.int64], ...],
        ]
        | None
    ):
        row_ids: NDArray[np.int64] = np.arange(len(frame), dtype=np.int64)
        order = row_ids[np.lexsort((row_ids, timestamps_ns))]
        if len(order) > _POWER_GATE_CONTEXT_ROWS:
            ordered_times = timestamps_ns[order]
            requested = len(order) - _POWER_GATE_CONTEXT_ROWS
            boundary = ordered_times[requested]
            left = int(np.searchsorted(ordered_times, boundary, side="left"))
            right = int(np.searchsorted(ordered_times, boundary, side="right"))
            start = left if len(order) - left <= _POWER_GATE_CONTEXT_ROWS else right
            order = order[start:]
        if len(order) < _GATE_MIN_ROWS:
            return None

        model_start = max(0, len(order) - _POWER_GATE_MAX_ROWS)
        if model_start > 0:
            ordered_times = timestamps_ns[order]
            boundary = ordered_times[model_start]
            left = int(np.searchsorted(ordered_times, boundary, side="left"))
            right = int(np.searchsorted(ordered_times, boundary, side="right"))
            model_start = left if len(order) - left <= _POWER_GATE_MAX_ROWS else right
        warmup_rows = order[:model_start]
        model_rows = order[model_start:]
        ordered_model_times = timestamps_ns[model_rows]
        requested = int(_POWER_GATE_TRAIN_FRACTION * len(model_rows))
        if requested >= len(model_rows):
            return None
        boundary = ordered_model_times[requested]
        cut = int(np.searchsorted(ordered_model_times, boundary, side="left"))
        minimum_future = max(120, int(0.20 * len(model_rows)))
        if cut < 100 or len(model_rows) - cut < minimum_future:
            return None
        windows = self._atomic_partitions(
            model_rows[cut:],
            timestamps_ns,
            _POWER_GATE_FORWARD_WINDOWS,
        )
        if len(windows) < 2 or min(map(len, windows)) < 40:
            return None
        return warmup_rows, model_rows[:cut], windows

    @staticmethod
    def _local_windows(
        windows: tuple[NDArray[np.int64], ...],
    ) -> tuple[NDArray[np.int64], ...]:
        offsets = np.cumsum([0, *(len(window) for window in windows)], dtype=np.int64)
        return tuple(
            np.arange(start, end, dtype=np.int64)
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        )

    @staticmethod
    def _model_frame(frame: Any, entity: Hashable, *, drop_entity: bool) -> Any:
        return frame.drop(columns=[entity]) if drop_entity else frame

    @staticmethod
    def _encode(
        train: Any,
        valid: Any,
        y_train: NDArray[Any],
        task: TemporalTask,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        preprocessor = _Preprocessor(
            target_encoding=False,
            task=task,
            compression_evidence=False,
        )
        encoded_train = preprocessor.fit_transform(train, y_train)
        encoded_valid = preprocessor.transform(valid)
        return encoded_train, encoded_valid

    @staticmethod
    def _probabilities(model: AdditiveCertifiedClassifier, X: NDArray[np.float64]) -> NDArray[np.float64]:
        scores = model._scores(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        return probability / probability.sum(axis=1, keepdims=True)

    @staticmethod
    def _aligned_probability(
        probability: NDArray[np.float64],
        model_classes: NDArray[Any],
        classes: NDArray[Any],
    ) -> NDArray[np.float64]:
        aligned: NDArray[np.float64] = np.zeros((len(probability), len(classes)), dtype=np.float64)
        for source, label in enumerate(model_classes):
            target = int(np.flatnonzero(classes == label)[0])
            aligned[:, target] = probability[:, source]
        return aligned

    @staticmethod
    def _classification_scores(
        y: NDArray[Any],
        probability: NDArray[np.float64],
        classes: NDArray[Any],
        metric: str,
        rare_class: Any,
    ) -> tuple[float, float, float]:
        from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

        if metric == "average_precision":
            positive = int(np.flatnonzero(classes == rare_class)[0])
            primary = float(average_precision_score(y == rare_class, probability[:, positive]))
            secondary = float(roc_auc_score(y == rare_class, probability[:, positive]))
        elif len(classes) == 2:
            primary = float(roc_auc_score(y == classes[-1], probability[:, -1]))
            secondary = primary
        else:
            primary = float(roc_auc_score(y, probability, labels=classes, multi_class="ovo", average="macro"))
            secondary = primary
        prediction = classes[probability.argmax(axis=1)]
        return primary, secondary, float(accuracy_score(y, prediction))

    @staticmethod
    def _classification_primary_score(
        y: NDArray[Any],
        probability: NDArray[np.float64],
        classes: NDArray[Any],
        metric: str,
        rare_class: Any,
    ) -> float:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if metric == "average_precision":
            positive = int(np.flatnonzero(classes == rare_class)[0])
            return float(average_precision_score(y == rare_class, probability[:, positive]))
        if len(classes) == 2:
            return float(roc_auc_score(y == classes[-1], probability[:, -1]))
        return float(roc_auc_score(y, probability, labels=classes, multi_class="ovo", average="macro"))

    @staticmethod
    def _has_class_coverage(y: NDArray[Any], classes: NDArray[Any]) -> bool:
        return bool(np.array_equal(np.unique(y), classes))

    def _classification_jackknife_lower(
        self,
        y: NDArray[Any],
        baseline_probability: NDArray[np.float64],
        candidate_probability: NDArray[np.float64],
        classes: NDArray[Any],
        metric: str,
        rare_class: Any,
        timestamps_ns: NDArray[np.int64],
        observed_gain: float,
    ) -> tuple[float | None, int, int]:
        desired_blocks = min(_POWER_GATE_CONFIDENCE_BLOCKS, max(4, len(y) // 1_000))
        blocks = self._atomic_partitions(
            np.arange(len(y), dtype=np.int64),
            timestamps_ns,
            desired_blocks,
        )
        if len(blocks) < 2:
            return None, 0, len(blocks)
        estimates = []
        all_rows: NDArray[np.int64] = np.arange(len(y), dtype=np.int64)
        for omitted in blocks:
            keep: NDArray[np.bool_] = np.ones(len(y), dtype=bool)
            keep[omitted] = False
            rows = all_rows[keep]
            sampled_y = y[rows]
            if not self._has_class_coverage(sampled_y, classes):
                continue
            baseline = self._classification_primary_score(
                sampled_y,
                baseline_probability[rows],
                classes,
                metric,
                rare_class,
            )
            candidate = self._classification_primary_score(
                sampled_y,
                candidate_probability[rows],
                classes,
                metric,
                rare_class,
            )
            estimates.append(candidate - baseline)
        if len(estimates) < max(3, len(blocks) - 1):
            return None, len(estimates), len(blocks)
        values = np.asarray(estimates, dtype=np.float64)
        standard_error = float(
            np.sqrt((len(values) - 1.0) / len(values) * np.sum(np.square(values - values.mean())))
        )
        critical = NormalDist().inv_cdf(_POWER_GATE_CONFIDENCE)
        return observed_gain - critical * standard_error, len(estimates), len(blocks)

    def _regression_jackknife_lower(
        self,
        y: NDArray[Any],
        baseline_prediction: NDArray[np.float64],
        candidate_prediction: NDArray[np.float64],
        timestamps_ns: NDArray[np.int64],
        observed_gain: float,
    ) -> tuple[float | None, int, int]:
        desired_blocks = min(_POWER_GATE_CONFIDENCE_BLOCKS, max(4, len(y) // 1_000))
        blocks = self._atomic_partitions(
            np.arange(len(y), dtype=np.int64),
            timestamps_ns,
            desired_blocks,
        )
        if len(blocks) < 2:
            return None, 0, len(blocks)
        target = y.astype(float)
        estimates = []
        all_rows: NDArray[np.int64] = np.arange(len(y), dtype=np.int64)
        for omitted in blocks:
            keep: NDArray[np.bool_] = np.ones(len(y), dtype=bool)
            keep[omitted] = False
            rows = all_rows[keep]
            baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_prediction[rows] - target[rows]))))
            candidate_rmse = float(np.sqrt(np.mean(np.square(candidate_prediction[rows] - target[rows]))))
            estimates.append((baseline_rmse - candidate_rmse) / max(baseline_rmse, 1e-12))
        values = np.asarray(estimates, dtype=np.float64)
        standard_error = float(
            np.sqrt((len(values) - 1.0) / len(values) * np.sum(np.square(values - values.mean())))
        )
        critical = NormalDist().inv_cdf(_POWER_GATE_CONFIDENCE)
        return observed_gain - critical * standard_error, len(estimates), len(blocks)

    def _classification_gate(
        self,
        baseline_train: NDArray[np.float64],
        baseline_valid: NDArray[np.float64],
        candidate_train: NDArray[np.float64],
        candidate_valid: NDArray[np.float64],
        y_train: NDArray[Any],
        y_valid: NDArray[Any],
        validation_groups: NDArray[np.int64],
        valid_timestamps: NDArray[np.int64],
        forward_windows: tuple[NDArray[np.int64], ...],
        baseline_cache: dict[str, Any],
    ) -> dict[str, Any]:
        classes, counts = np.unique(y_train, return_counts=True)
        rare_index = int(np.argmin(counts))
        rare_class = classes[rare_index]
        rare_rate = float(counts[rare_index] / counts.sum())
        metric = "average_precision" if len(classes) == 2 and rare_rate <= 0.10 else "roc_auc"
        leaf = int(np.clip(len(y_train) // 200, 4, 30))
        config = {
            "rounds": 96,
            "lr": 0.10,
            "depth": 3,
            "leaf": leaf,
            "holdout": 0.20,
            "patience": 12,
            "class_weight": "balanced" if metric == "average_precision" else None,
            "seed": self.seed,
            "refit": False,
            "stratified_holdout": True,
        }
        baseline_probability = baseline_cache.get("classification_probability")
        if baseline_probability is None:
            baseline_model = AdditiveCertifiedClassifier(**config).fit(
                baseline_train,
                y_train,
                validation_groups=validation_groups,
            )
            baseline_probability = self._aligned_probability(
                self._probabilities(baseline_model, baseline_valid),
                np.asarray(baseline_model.classes_),
                classes,
            )
            baseline_cache["classification_probability"] = baseline_probability
        candidate_model = AdditiveCertifiedClassifier(**config).fit(
            candidate_train,
            y_train,
            validation_groups=validation_groups,
        )
        candidate_probability = self._aligned_probability(
            self._probabilities(candidate_model, candidate_valid),
            np.asarray(candidate_model.classes_),
            classes,
        )
        baseline_score, baseline_secondary, baseline_accuracy = self._classification_scores(
            y_valid,
            baseline_probability,
            classes,
            metric,
            rare_class,
        )
        candidate_score, candidate_secondary, candidate_accuracy = self._classification_scores(
            y_valid,
            candidate_probability,
            classes,
            metric,
            rare_class,
        )
        gain = candidate_score - baseline_score
        legacy_minimum_gain = (
            _RARE_EVENT_MIN_GAIN if metric == "average_precision" else _CLASSIFICATION_MIN_GAIN
        )
        minimum_gain = (
            _RARE_EVENT_MIN_PRACTICAL_GAIN
            if metric == "average_precision"
            else _CLASSIFICATION_MIN_PRACTICAL_GAIN
        )
        safeguard = (
            candidate_secondary >= baseline_secondary - _RARE_EVENT_MAX_AUC_LOSS
            if metric == "average_precision"
            else candidate_accuracy >= baseline_accuracy - _CLASSIFICATION_MAX_ACCURACY_LOSS
        )
        forward_gains: list[float | None] = []
        forward_safeguards: list[bool | None] = []
        forward_severe_safeguards: list[bool | None] = []
        for rows in forward_windows:
            window_y = y_valid[rows]
            if not self._has_class_coverage(window_y, classes):
                forward_gains.append(None)
                forward_safeguards.append(None)
                forward_severe_safeguards.append(None)
                continue
            window_baseline, window_baseline_secondary, window_baseline_accuracy = (
                self._classification_scores(
                    window_y,
                    baseline_probability[rows],
                    classes,
                    metric,
                    rare_class,
                )
            )
            window_candidate, window_candidate_secondary, window_candidate_accuracy = (
                self._classification_scores(
                    window_y,
                    candidate_probability[rows],
                    classes,
                    metric,
                    rare_class,
                )
            )
            if metric == "average_precision":
                window_safeguard = (
                    window_candidate_secondary >= window_baseline_secondary - _RARE_EVENT_MAX_AUC_LOSS
                )
                severe_safeguard = (
                    window_candidate_secondary >= window_baseline_secondary - 2.0 * _RARE_EVENT_MAX_AUC_LOSS
                )
            else:
                window_safeguard = (
                    window_candidate_accuracy >= window_baseline_accuracy - _CLASSIFICATION_MAX_ACCURACY_LOSS
                )
                severe_safeguard = (
                    window_candidate_accuracy
                    >= window_baseline_accuracy - 2.0 * _CLASSIFICATION_MAX_ACCURACY_LOSS
                )
            forward_gains.append(window_candidate - window_baseline)
            forward_safeguards.append(bool(window_safeguard))
            forward_severe_safeguards.append(bool(severe_safeguard))

        evaluated = sum(value is not None for value in forward_gains)
        required = evaluated // 2 + 1
        passed = sum(
            gain_value is not None and gain_value > 0.0 and safeguard_value is True
            for gain_value, safeguard_value in zip(forward_gains, forward_safeguards, strict=True)
        )
        forward_consistency = evaluated >= 2 and passed >= required
        severe_safeguard = evaluated >= 2 and all(
            value is True for value in forward_severe_safeguards if value is not None
        )
        confidence_lower, confidence_samples, confidence_blocks = self._classification_jackknife_lower(
            y_valid,
            baseline_probability,
            candidate_probability,
            classes,
            metric,
            rare_class,
            valid_timestamps,
            gain,
        )
        confidence_passed = confidence_lower is not None and confidence_lower > 0.0
        legacy_passed = gain >= legacy_minimum_gain
        practical_gain_passed = gain >= minimum_gain
        confidence_or_legacy_passed = confidence_passed or legacy_passed
        selected = bool(
            practical_gain_passed
            and safeguard
            and forward_consistency
            and severe_safeguard
            and confidence_or_legacy_passed
        )
        selection_checks = {
            "practical_gain": bool(practical_gain_passed),
            "overall_safeguard": bool(safeguard),
            "forward_consistency": bool(forward_consistency),
            "forward_severe_safeguard": bool(severe_safeguard),
            "confidence_or_legacy_gain": bool(confidence_or_legacy_passed),
        }
        return {
            "selected": selected,
            "metric": metric,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "gain": gain,
            "minimum_gain": minimum_gain,
            "legacy_minimum_gain": legacy_minimum_gain,
            "baseline_accuracy": baseline_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "baseline_secondary_score": baseline_secondary,
            "candidate_secondary_score": candidate_secondary,
            "safeguard_passed": bool(safeguard),
            "power_aware": True,
            "confidence_level": _POWER_GATE_CONFIDENCE,
            "confidence_method": "delete_one_time_block_jackknife",
            "confidence_lower_gain": confidence_lower,
            "confidence_passed": bool(confidence_passed),
            "confidence_samples": int(confidence_samples),
            "confidence_blocks": int(confidence_blocks),
            "forward_windows": int(len(forward_windows)),
            "forward_window_rows": [int(len(rows)) for rows in forward_windows],
            "forward_window_gains": forward_gains,
            "forward_window_safeguards": forward_safeguards,
            "forward_windows_evaluated": int(evaluated),
            "forward_windows_passed": int(passed),
            "forward_windows_required": int(required),
            "forward_consistency_passed": bool(forward_consistency),
            "forward_severe_safeguard_passed": bool(severe_safeguard),
            "selection_checks": selection_checks,
            "failed_checks": [name for name, passed_check in selection_checks.items() if not passed_check],
        }

    def _regression_gate(
        self,
        baseline_train: NDArray[np.float64],
        baseline_valid: NDArray[np.float64],
        candidate_train: NDArray[np.float64],
        candidate_valid: NDArray[np.float64],
        y_train: NDArray[Any],
        y_valid: NDArray[Any],
        validation_groups: NDArray[np.int64],
        valid_timestamps: NDArray[np.int64],
        forward_windows: tuple[NDArray[np.int64], ...],
        baseline_cache: dict[str, Any],
    ) -> dict[str, Any]:
        leaf = int(np.clip(len(y_train) // 200, 4, 30))
        config = {
            "rounds": 128,
            "lr": 0.08,
            "depth": 3,
            "leaf": leaf,
            "colsample": 1.0,
            "holdout": 0.20,
            "patience": 16,
            "refit": False,
            "seed": self.seed,
        }
        baseline_prediction = baseline_cache.get("regression_prediction")
        if baseline_prediction is None:
            baseline_model = AdditiveCertifiedRegressor(**config).fit(
                baseline_train,
                y_train,
                validation_groups=validation_groups,
            )
            baseline_prediction = np.asarray(
                baseline_model.predict(baseline_valid),
                dtype=np.float64,
            )
            baseline_cache["regression_prediction"] = baseline_prediction
        candidate_model = AdditiveCertifiedRegressor(**config).fit(
            candidate_train,
            y_train,
            validation_groups=validation_groups,
        )
        candidate_prediction = np.asarray(candidate_model.predict(candidate_valid), dtype=np.float64)
        target = y_valid.astype(float)
        baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_prediction - target))))
        candidate_rmse = float(np.sqrt(np.mean(np.square(candidate_prediction - target))))
        relative_gain = (baseline_rmse - candidate_rmse) / max(baseline_rmse, 1e-12)
        forward_gains = []
        for rows in forward_windows:
            window_target = target[rows]
            window_baseline = float(np.sqrt(np.mean(np.square(baseline_prediction[rows] - window_target))))
            window_candidate = float(np.sqrt(np.mean(np.square(candidate_prediction[rows] - window_target))))
            forward_gains.append((window_baseline - window_candidate) / max(window_baseline, 1e-12))
        evaluated = len(forward_gains)
        required = evaluated // 2 + 1
        passed = sum(value > 0.0 for value in forward_gains)
        forward_consistency = evaluated >= 2 and passed >= required
        severe_safeguard = evaluated >= 2 and all(
            value >= -_REGRESSION_MAX_FORWARD_LOSS for value in forward_gains
        )
        confidence_lower, confidence_samples, confidence_blocks = self._regression_jackknife_lower(
            y_valid,
            baseline_prediction,
            candidate_prediction,
            valid_timestamps,
            relative_gain,
        )
        confidence_passed = confidence_lower is not None and confidence_lower > 0.0
        legacy_passed = relative_gain >= _REGRESSION_MIN_RELATIVE_GAIN
        practical_gain_passed = relative_gain >= _REGRESSION_MIN_PRACTICAL_GAIN
        confidence_or_legacy_passed = confidence_passed or legacy_passed
        selected = bool(
            practical_gain_passed and forward_consistency and severe_safeguard and confidence_or_legacy_passed
        )
        selection_checks = {
            "practical_gain": bool(practical_gain_passed),
            "forward_consistency": bool(forward_consistency),
            "forward_severe_safeguard": bool(severe_safeguard),
            "confidence_or_legacy_gain": bool(confidence_or_legacy_passed),
        }
        return {
            "selected": selected,
            "metric": "neg_rmse",
            "baseline_score": -baseline_rmse,
            "candidate_score": -candidate_rmse,
            "gain": baseline_rmse - candidate_rmse,
            "relative_gain": relative_gain,
            "minimum_relative_gain": _REGRESSION_MIN_PRACTICAL_GAIN,
            "legacy_minimum_relative_gain": _REGRESSION_MIN_RELATIVE_GAIN,
            "power_aware": True,
            "confidence_level": _POWER_GATE_CONFIDENCE,
            "confidence_method": "delete_one_time_block_jackknife",
            "confidence_lower_relative_gain": confidence_lower,
            "confidence_passed": bool(confidence_passed),
            "confidence_samples": int(confidence_samples),
            "confidence_blocks": int(confidence_blocks),
            "forward_windows": int(len(forward_windows)),
            "forward_window_rows": [int(len(rows)) for rows in forward_windows],
            "forward_window_relative_gains": [float(value) for value in forward_gains],
            "forward_windows_evaluated": int(evaluated),
            "forward_windows_passed": int(passed),
            "forward_windows_required": int(required),
            "forward_consistency_passed": bool(forward_consistency),
            "forward_severe_safeguard_passed": bool(severe_safeguard),
            "selection_checks": selection_checks,
            "failed_checks": [name for name, passed_check in selection_checks.items() if not passed_check],
        }

    @staticmethod
    def _rejection_reason(decision: dict[str, Any]) -> str:
        failed = set(decision.get("failed_checks", ()))
        if "practical_gain" in failed:
            return "future_holdout_gain_below_gate"
        if "overall_safeguard" in failed:
            return "future_holdout_safeguard_failed"
        if "forward_consistency" in failed:
            return "future_windows_inconsistent"
        if "forward_severe_safeguard" in failed:
            return "future_window_safeguard_failed"
        return "future_gain_confidence_not_established"

    def evaluate(
        self,
        events: Any,
        y: Any,
        *,
        entity: Hashable,
        timestamp: Hashable,
        value_columns: Sequence[Hashable] | None = None,
        task: TemporalTask,
        drop_entity: bool = True,
        _baseline_cache: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the serializable multi-window future deployment decision."""
        if task not in {"classification", "regression"}:
            raise ValueError("temporal evidence task must be classification or regression")
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(events):
            raise ValueError("temporal evidence y must be one-dimensional and aligned with events")

        probe = TemporalLaplaceMap(entity, timestamp, value_columns=value_columns)
        frame, entity_codes, _, timestamps_ns = probe._extract(events)
        split = self._power_window_split(frame, entity_codes, timestamps_ns)
        if split is None:
            return gate_report(
                "temporal_laplace_evidence",
                False,
                stage="schema",
                reason="insufficient_chronological_holdout",
                source_rows=int(len(frame)),
                gate_row_cap=_POWER_GATE_MAX_ROWS,
                context_row_cap=_POWER_GATE_CONTEXT_ROWS,
                power_aware=True,
            )
        warmup_rows, train_rows, valid_windows = split
        valid_rows = np.concatenate(valid_windows).astype(np.int64, copy=False)
        local_windows = self._local_windows(valid_windows)
        y_train = target[train_rows]
        y_valid = target[valid_rows]
        if task == "classification":
            train_classes = np.unique(y_train)
            if len(train_classes) < 2 or set(np.unique(y_valid).tolist()) != set(train_classes.tolist()):
                return gate_report(
                    "temporal_laplace_evidence",
                    False,
                    stage="schema",
                    reason="future_holdout_class_coverage",
                    source_rows=int(len(frame)),
                    train_rows=int(len(train_rows)),
                    valid_rows=int(len(valid_rows)),
                )

        past_rows = np.concatenate((warmup_rows, train_rows)).astype(np.int64, copy=False)
        past_events = frame.iloc[past_rows].reset_index(drop=True)
        train_events = frame.iloc[train_rows].reset_index(drop=True)
        valid_events = frame.iloc[valid_rows].reset_index(drop=True)
        temporal = TemporalLaplaceMap(entity, timestamp, value_columns=value_columns)
        try:
            candidate_past_frame = temporal.fit_augment(past_events)
            candidate_train_frame = candidate_past_frame.iloc[len(warmup_rows) :].reset_index(drop=True)
            candidate_valid_frame = temporal.augment(valid_events)
        except ValueError as error:
            return gate_report(
                "temporal_laplace_evidence",
                False,
                stage="schema",
                reason=f"future_window_ineligible:{error}",
                source_rows=int(len(frame)),
                train_rows=int(len(train_rows)),
                valid_rows=int(len(valid_rows)),
            )

        baseline_train_frame = self._model_frame(train_events, entity, drop_entity=drop_entity)
        baseline_valid_frame = self._model_frame(valid_events, entity, drop_entity=drop_entity)
        candidate_train_frame = self._model_frame(candidate_train_frame, entity, drop_entity=drop_entity)
        candidate_valid_frame = self._model_frame(candidate_valid_frame, entity, drop_entity=drop_entity)
        baseline_cache = {} if _baseline_cache is None else _baseline_cache
        encoded_baseline = baseline_cache.get("encoded")
        if encoded_baseline is None:
            encoded_baseline = self._encode(
                baseline_train_frame,
                baseline_valid_frame,
                y_train,
                task,
            )
            baseline_cache["encoded"] = encoded_baseline
        baseline_train, baseline_valid = encoded_baseline
        candidate_train, candidate_valid = self._encode(
            candidate_train_frame,
            candidate_valid_frame,
            y_train,
            task,
        )
        if task == "classification":
            decision = self._classification_gate(
                baseline_train,
                baseline_valid,
                candidate_train,
                candidate_valid,
                y_train,
                y_valid,
                timestamps_ns[train_rows],
                timestamps_ns[valid_rows],
                local_windows,
                baseline_cache,
            )
        else:
            decision = self._regression_gate(
                baseline_train,
                baseline_valid,
                candidate_train,
                candidate_valid,
                y_train,
                y_valid,
                timestamps_ns[train_rows],
                timestamps_ns[valid_rows],
                local_windows,
                baseline_cache,
            )
        selected = bool(decision.pop("selected"))
        metric = str(decision.pop("metric"))
        reason = None if selected else self._rejection_reason(decision)
        return gate_report(
            "temporal_laplace_evidence",
            selected,
            stage="schema",
            metric=metric,
            mean_score=float(decision["candidate_score"]),
            reason=reason,
            source_rows=int(len(frame)),
            gate_rows=int(len(train_rows) + len(valid_rows)),
            context_rows=int(len(warmup_rows) + len(train_rows) + len(valid_rows)),
            warmup_rows=int(len(warmup_rows)),
            train_rows=int(len(train_rows)),
            valid_rows=int(len(valid_rows)),
            gate_row_cap=_POWER_GATE_MAX_ROWS,
            context_row_cap=_POWER_GATE_CONTEXT_ROWS,
            gate_entities=int(len(np.unique(entity_codes[np.r_[train_rows, valid_rows]]))),
            entity_retained=not drop_entity,
            features=int(len(temporal.feature_names_)),
            scales_seconds=[float(scale) for scale in temporal.scales_seconds_],
            channels=[str(channel) for channel in temporal.channel_names_],
            **decision,
        )


__all__ = ["TemporalEvidenceChallenger"]
