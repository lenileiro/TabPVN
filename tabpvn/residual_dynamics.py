"""Transparent diagnostics for additive multiclass residual trajectories.

The booster state after round ``t`` is ``F_t = F_{t-1} + update_t``.  This
module measures that trajectory on verifier rows and decides whether a stable,
unresolved class pair warrants extra tree capacity.  It contains no learned
controller and does not alter the fitted tree representation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

_CONFIDENCE_Z = 1.959963984540054
_DYNAMICS_WINDOW = 4
_REDUNDANT_EXPLAINED_ENERGY = 0.90
_EPSILON = float(np.finfo(float).eps)


def _validate_scores(scores: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    target = np.asarray(target, dtype=np.int64)
    if scores.ndim != 2 or target.ndim != 1 or len(scores) != len(target):
        raise ValueError("scores and target must be aligned two-dimensional and one-dimensional arrays")
    if scores.shape[1] < 2:
        raise ValueError("residual dynamics require at least two classes")
    return scores, target


def _validate_weight(sample_weight: np.ndarray | None, rows: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weight = np.asarray(sample_weight, dtype=float)
    if weight.ndim != 1 or len(weight) != rows:
        raise ValueError("sample_weight must have one value per target row")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("sample_weight must contain finite non-negative values")
    return weight


def _weighted_mean_se(values: np.ndarray, weight: np.ndarray | None) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float(values.mean()) if len(values) else float("nan"), float("inf"))
    if weight is None:
        mean = float(values.mean())
        return mean, float(values.std(ddof=1) / np.sqrt(len(values)))
    weight = np.asarray(weight, dtype=float)
    total = float(weight.sum())
    squared_total = float(np.square(weight).sum())
    if total <= 0.0 or squared_total <= 0.0:
        return float(values.mean()), float("inf")
    mean = float(np.average(values, weights=weight))
    effective_rows = total * total / squared_total
    if effective_rows <= 1.0:
        return mean, float("inf")
    variance = float(np.average(np.square(values - mean), weights=weight))
    return mean, float(np.sqrt(max(variance, 0.0) / effective_rows))


def _class_loss(scores: np.ndarray, target_class: int, other_class: int) -> np.ndarray:
    return np.logaddexp(scores[:, target_class], scores[:, other_class]) - scores[:, target_class]


def pair_loss(
    scores: np.ndarray,
    target: np.ndarray,
    pair: tuple[int, int],
    sample_weight: np.ndarray | None = None,
    rows: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return class-balanced conditional log-loss and its standard error."""
    scores, target = _validate_scores(scores, target)
    weight = _validate_weight(sample_weight, len(target))
    selected_rows = np.arange(len(target)) if rows is None else np.asarray(rows, dtype=np.int64)
    means: list[float] = []
    variances: list[float] = []
    for target_class, other_class in (pair, pair[::-1]):
        class_rows = selected_rows[target[selected_rows] == target_class]
        class_weight = None if weight is None else weight[class_rows]
        mean, standard_error = _weighted_mean_se(
            _class_loss(scores[class_rows], target_class, other_class),
            class_weight,
        )
        means.append(mean)
        variances.append(standard_error * standard_error)
    if not np.isfinite(means).all():
        return float("nan"), float("inf")
    return 0.5 * (means[0] + means[1]), 0.5 * np.sqrt(sum(variances))


def hardest_class_pair(
    scores: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray | None = None,
    rows: np.ndarray | None = None,
) -> tuple[int, int]:
    """Return the pair with the largest class-balanced conditional log-loss."""
    scores, target = _validate_scores(scores, target)
    weight = _validate_weight(sample_weight, len(target))
    selected_rows = np.arange(len(target)) if rows is None else np.asarray(rows, dtype=np.int64)
    if selected_rows.ndim != 1:
        raise ValueError("rows must be one-dimensional")
    loss_sum, mass = _class_loss_sums(scores, target, weight, selected_rows)
    return _hardest_pair_from_sums(loss_sum, mass)


def _class_loss_sums(
    scores: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    selected_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    class_loss_sum = np.zeros((scores.shape[1], scores.shape[1]), dtype=float)
    class_mass = np.zeros(scores.shape[1], dtype=float)
    for target_class in range(scores.shape[1]):
        class_rows = selected_rows[target[selected_rows] == target_class]
        if not len(class_rows):
            continue
        selected_scores = scores[class_rows]
        losses = (
            np.logaddexp(selected_scores[:, [target_class]], selected_scores)
            - selected_scores[:, [target_class]]
        )
        if weight is None:
            class_loss_sum[target_class] = losses.sum(axis=0)
            class_mass[target_class] = len(class_rows)
            continue
        class_weight = weight[class_rows]
        total_weight = float(class_weight.sum())
        if total_weight <= 0.0:
            class_loss_sum[target_class] = losses.sum(axis=0)
            class_mass[target_class] = len(class_rows)
        else:
            class_loss_sum[target_class] = np.sum(losses * class_weight[:, None], axis=0)
            class_mass[target_class] = total_weight
    return class_loss_sum, class_mass


def _hardest_pair_from_sums(loss_sum: np.ndarray, mass: np.ndarray) -> tuple[int, int]:
    class_losses = np.divide(
        loss_sum,
        mass[:, None],
        out=np.full_like(loss_sum, np.nan),
        where=mass[:, None] > 0.0,
    )
    best_pair = (0, 1)
    best_loss = -np.inf
    for left in range(loss_sum.shape[1] - 1):
        for right in range(left + 1, loss_sum.shape[1]):
            loss = 0.5 * (class_losses[left, right] + class_losses[right, left])
            if np.isfinite(loss) and loss > best_loss:
                best_pair = (left, right)
                best_loss = float(loss)
    return best_pair


def _hardest_pair_views(
    scores: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    halves: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    half_statistics = tuple(_class_loss_sums(scores, target, weight, rows) for rows in halves)
    half_pairs = tuple(_hardest_pair_from_sums(*statistics) for statistics in half_statistics)
    full_sum = half_statistics[0][0] + half_statistics[1][0]
    full_mass = half_statistics[0][1] + half_statistics[1][1]
    return _hardest_pair_from_sums(full_sum, full_mass), half_pairs  # type: ignore[return-value]


def _pair_loss_change(
    before: np.ndarray,
    after: np.ndarray,
    target: np.ndarray,
    pair: tuple[int, int],
    weight: np.ndarray | None,
    rows: np.ndarray,
) -> tuple[float, float]:
    means: list[float] = []
    variances: list[float] = []
    for target_class, other_class in (pair, pair[::-1]):
        class_rows = rows[target[rows] == target_class]
        change = _class_loss(after[class_rows], target_class, other_class) - _class_loss(
            before[class_rows], target_class, other_class
        )
        class_weight = None if weight is None else weight[class_rows]
        mean, standard_error = _weighted_mean_se(change, class_weight)
        means.append(mean)
        variances.append(standard_error * standard_error)
    if not np.isfinite(means).all():
        return float("nan"), float("inf")
    return 0.5 * (means[0] + means[1]), 0.5 * np.sqrt(sum(variances))


def _stratified_halves(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    halves: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
    for target_class in np.unique(target):
        rows = np.flatnonzero(target == target_class)
        halves[0].append(rows[::2])
        halves[1].append(rows[1::2])
    return np.sort(np.concatenate(halves[0])), np.sort(np.concatenate(halves[1]))


def _center_logits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - values.mean(axis=1, keepdims=True)


def _norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.square(values))))


def _alignment(first: np.ndarray | None, second: np.ndarray) -> float:
    if first is None:
        return 0.0
    denominator = _norm(first) * _norm(second)
    return 0.0 if denominator <= _EPSILON else float(np.sum(first * second) / denominator)


def _persistent_on_rows(
    previous: np.ndarray | None,
    current: np.ndarray,
    rows: np.ndarray,
    weight: np.ndarray | None,
) -> bool:
    if previous is None or not len(rows):
        return False
    contribution = np.sum(previous[rows] * current[rows], axis=1)
    row_weight = None if weight is None else weight[rows]
    mean, standard_error = _weighted_mean_se(contribution, row_weight)
    return bool(np.isfinite(standard_error) and mean - _CONFIDENCE_Z * standard_error > 0.0)


def _effective_rank(updates: Sequence[np.ndarray]) -> float:
    if not updates:
        return 0.0
    gram = np.asarray([[np.sum(left * right) for right in updates] for left in updates], dtype=float)
    trace = float(np.trace(gram))
    squared_trace = float(np.sum(np.square(gram)))
    return 0.0 if squared_trace <= _EPSILON else trace * trace / squared_trace


def _innovation_ratio(previous: Sequence[np.ndarray], current: np.ndarray) -> float:
    current_energy = float(np.sum(np.square(current)))
    if not previous or current_energy <= _EPSILON:
        return 1.0
    gram = np.asarray([[np.sum(left * right) for right in previous] for left in previous], dtype=float)
    cross = np.asarray([np.sum(update * current) for update in previous], dtype=float)
    explained = float(cross @ np.linalg.pinv(gram, hermitian=True) @ cross)
    residual = max(0.0, current_energy - min(current_energy, explained))
    return float(np.sqrt(residual / current_energy))


def _logloss(scores: np.ndarray, target: np.ndarray, weight: np.ndarray | None) -> float:
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    loss = log_normalizer - shifted[np.arange(len(target)), target]
    return float(loss.mean() if weight is None else np.average(loss, weights=weight))


class ResidualDynamicsTracker:
    """Measure verifier dynamics and expose the next transparent growth phase."""

    def __init__(
        self,
        initial_scores: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
        detailed: bool = True,
    ) -> None:
        scores, target = _validate_scores(initial_scores, target)
        self._target = target.copy()
        self._weight = _validate_weight(sample_weight, len(target))
        self._halves = _stratified_halves(target)
        self._scores = scores.copy()
        self._previous_update: np.ndarray | None = None
        self._updates: list[np.ndarray] = []
        self._previous_pair: tuple[int, int] | None = None
        self._active_pair: tuple[int, int] | None = None
        self._next_pair: tuple[int, int] | tuple[()] = ()
        self._detailed = bool(detailed)
        self._round = 0
        self.records: list[dict[str, object]] = []

    @property
    def next_pair(self) -> tuple[int, int] | tuple[()]:
        """Class pair that should receive best-first capacity next round."""
        return self._next_pair

    def observe(
        self,
        scores: np.ndarray,
        update: np.ndarray,
        growth_pair: tuple[int, int] | tuple[()] | None,
        validation_scores: dict[str, float] | None = None,
    ) -> tuple[int, int] | tuple[()]:
        """Record one completed round and return the next round's selected pair."""
        scores, target = _validate_scores(scores, self._target)
        update = np.asarray(update, dtype=float)
        if update.shape != scores.shape:
            raise ValueError("update must have the same shape as scores")
        centered_update = _center_logits(update)
        pair, half_pairs = _hardest_pair_views(scores, target, self._weight, self._halves)
        pair_consensus = bool(all(candidate == pair for candidate in half_pairs))
        pair_stable = bool(pair_consensus and pair == self._previous_pair)

        full_change, full_change_se = _pair_loss_change(
            self._scores,
            scores,
            target,
            pair,
            self._weight,
            np.arange(len(target)),
        )
        pair_loss_stalled = bool(
            np.isfinite(full_change)
            and np.isfinite(full_change_se)
            and full_change + _CONFIDENCE_Z * full_change_se >= -1e-12
        )
        stalled_halves = (
            tuple(
                bool(
                    np.isfinite(change)
                    and np.isfinite(standard_error)
                    and change + _CONFIDENCE_Z * standard_error >= -1e-12
                )
                for change, standard_error in (
                    _pair_loss_change(self._scores, scores, target, pair, self._weight, rows)
                    for rows in self._halves
                )
            )
            if self._detailed
            else ()
        )
        persistent_halves = tuple(
            _persistent_on_rows(self._previous_update, centered_update, rows, self._weight)
            for rows in self._halves
        )
        recent = self._updates[-(_DYNAMICS_WINDOW - 1) :]
        innovation_ratio = _innovation_ratio(recent, centered_update)
        redundant_update = bool(
            recent and 1.0 - innovation_ratio * innovation_ratio >= _REDUNDANT_EXPLAINED_ENERGY
        )
        continuing_pair = bool(self._active_pair is not None and pair_consensus and pair == self._active_pair)
        initial_stall = bool(self._round == 0 and pair_consensus and pair_loss_stalled)
        stable_bottleneck = bool(
            pair_stable and all(persistent_halves) and (pair_loss_stalled or redundant_update)
        )
        activate = bool(
            _norm(centered_update) > _EPSILON
            and (continuing_pair or initial_stall or (pair_consensus and stable_bottleneck))
        )

        if self._detailed:
            activation_reason = None
            if continuing_pair:
                activation_reason = "continuing_verified_pair"
            elif initial_stall:
                activation_reason = "initial_pair_stall"
            elif stable_bottleneck and pair_loss_stalled:
                activation_reason = "stable_pair_stall"
            elif stable_bottleneck and redundant_update:
                activation_reason = "low_innovation"

            blocked_by: list[str] = []
            if not activate:
                if not pair_consensus:
                    blocked_by.append("verifier_pair_disagreement")
                if not pair_stable:
                    blocked_by.append("pair_not_stable")
                if not pair_loss_stalled:
                    blocked_by.append("pair_loss_improving")
                if not redundant_update:
                    blocked_by.append("update_has_innovation")
                if not all(persistent_halves):
                    blocked_by.append("update_direction_not_persistent")

            previous_norm = _norm(self._previous_update) if self._previous_update is not None else 0.0
            curvature = (
                0.0
                if self._previous_update is None or previous_norm <= _EPSILON
                else _norm(centered_update - self._previous_update) / previous_norm
            )
            pair_rows = np.flatnonzero((target == pair[0]) | (target == pair[1]))
            margin_update = centered_update[pair_rows, pair[0]] - centered_update[pair_rows, pair[1]]
            loss, loss_se = pair_loss(scores, target, pair, self._weight)
            velocity = _norm(centered_update) / np.sqrt(centered_update.size)
            state_norm = _norm(_center_logits(scores))
            self.records.append(
                {
                    "round": self._round,
                    "growth_phase": "hard_pair" if growth_pair and len(growth_pair) == 2 else "depth_wise",
                    "growth_pair": tuple(growth_pair) if growth_pair else (),
                    "hard_pair": pair,
                    "half_pairs": half_pairs,
                    "pair_consensus": pair_consensus,
                    "pair_stable": pair_stable,
                    "pair_loss": float(loss),
                    "pair_loss_se": float(loss_se),
                    "pair_loss_change": float(full_change),
                    "pair_loss_change_se": float(full_change_se),
                    "pair_loss_stalled": pair_loss_stalled,
                    "velocity": float(velocity),
                    "relative_update": float(_norm(centered_update) / max(state_norm, _EPSILON)),
                    "alignment": _alignment(self._previous_update, centered_update),
                    "curvature": float(curvature),
                    "effective_rank": float(_effective_rank([*recent, centered_update])),
                    "innovation_ratio": float(innovation_ratio),
                    "redundant_update": redundant_update,
                    "pair_margin_velocity": float(np.sqrt(np.mean(np.square(margin_update)))),
                    "stalled_halves": stalled_halves,
                    "persistent_halves": persistent_halves,
                    "next_phase": "hard_pair" if activate else "depth_wise",
                    "next_pair": pair if activate else (),
                    "activation_reason": activation_reason,
                    "blocked_by": tuple(blocked_by),
                    "validation_logloss": _logloss(scores, target, self._weight),
                    "validation_scores": dict(validation_scores or {}),
                }
            )

        self._scores = scores.copy()
        self._previous_update = centered_update.copy()
        self._updates.append(centered_update.copy())
        self._updates = self._updates[-_DYNAMICS_WINDOW:]
        self._previous_pair = pair if pair_consensus else None
        self._active_pair = pair if activate else None
        self._next_pair = pair if activate else ()
        self._round += 1
        return self._next_pair


def summarize_dynamics(records: Sequence[dict[str, object]]) -> dict[str, float | int]:
    """Return compact aggregate evidence suitable for benchmark artifacts."""
    if not records:
        return {
            "rounds": 0,
            "hard_pair_rounds": 0,
            "activation_rate": 0.0,
            "mean_effective_rank": 0.0,
            "mean_innovation_ratio": 0.0,
            "final_validation_logloss": float("nan"),
        }
    hard_pair_rounds = sum(record.get("growth_phase") == "hard_pair" for record in records)
    return {
        "rounds": len(records),
        "hard_pair_rounds": int(hard_pair_rounds),
        "activation_rate": float(hard_pair_rounds / len(records)),
        "mean_effective_rank": float(np.mean([cast(float, record["effective_rank"]) for record in records])),
        "mean_innovation_ratio": float(
            np.mean([cast(float, record["innovation_ratio"]) for record in records])
        ),
        "final_validation_logloss": cast(float, records[-1]["validation_logloss"]),
    }
