"""Validation geometry for ordered event data.

Random cross-validation answers an exchangeable-data question. Event models
instead deploy into the future, so their architecture choices must be measured
on rows strictly later than every row used to fit a candidate. This module owns
that boundary and keeps equal-timestamp rows on the same side of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FutureValidation:
    """Timestamp groups and deterministic, tie-safe future holdouts."""

    groups: NDArray[np.int64]
    _order: NDArray[np.int64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        groups = np.asarray(self.groups, dtype=np.int64)
        if groups.ndim != 1:
            raise ValueError("future-validation groups must be one-dimensional")
        if len(groups) < 2:
            raise ValueError("future validation requires at least two rows")
        groups = groups.copy()
        groups.setflags(write=False)
        rows: NDArray[np.int64] = np.arange(len(groups), dtype=np.int64)
        order = (
            rows
            if np.all(groups[1:] >= groups[:-1])
            else np.lexsort((rows, groups)).astype(np.int64, copy=False)
        )
        order.setflags(write=False)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "_order", order)

    @classmethod
    def from_timestamps(cls, timestamps: Any) -> Self:
        """Parse timestamp-like values into nanosecond validation groups."""
        import pandas as pd

        series = pd.Series(timestamps)
        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_datetime64_any_dtype(
            series.dtype
        ):
            raise TypeError("event validation timestamps need an explicit pandas datetime dtype")
        options = {"format": "mixed"} if series.dtype == object else {}
        parsed = pd.to_datetime(series, errors="coerce", utc=True, **options)
        missing = parsed.isna().to_numpy()
        if missing.any():
            raise ValueError(
                f"event validation timestamps contain {int(missing.sum())} missing or invalid values"
            )
        return cls(parsed.astype("int64").to_numpy(dtype=np.int64, copy=False))

    @property
    def order(self) -> NDArray[np.int64]:
        """Stable row order from oldest to newest timestamp."""
        return self._order

    def take(self, rows: Any) -> Self:
        """Return aligned timestamp groups for a row subset."""
        selected = np.asarray(rows, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("future-validation rows must be one-dimensional")
        return type(self)(self.groups[selected])

    def sorted(self) -> tuple[Self, NDArray[np.int64]]:
        """Return this context in chronological row order plus the source order."""
        order = self.order
        return type(self)(self.groups[order]), order

    def bounded_rows(self, cap: int) -> NDArray[np.int64]:
        """Select at most ``cap`` most-recent rows without bisecting the first tie."""
        cap = int(cap)
        if cap < 2:
            raise ValueError("future-validation row cap must be at least two")
        order = self.order
        if len(order) <= cap:
            return order
        ordered_groups = self.groups[order]
        requested = len(order) - cap
        boundary = ordered_groups[requested]
        left = int(np.searchsorted(ordered_groups, boundary, side="left"))
        right = int(np.searchsorted(ordered_groups, boundary, side="right"))
        start = left if len(order) - left <= cap else right
        # A timestamp batch larger than the whole budget cannot be retained
        # without violating the hard cap. If the contiguous tail leaves fewer
        # than two timestamps, retain a deterministic within-batch sample from
        # several recent timestamps. Selected equal-time rows remain atomic at
        # the later split boundary.
        tail = order[start:]
        if len(tail) >= 2 and np.unique(self.groups[tail]).size >= 2:
            return tail
        unique_groups = np.unique(ordered_groups)
        group_count = min(len(unique_groups), min(32, cap))
        if group_count < 2:
            raise ValueError("bounded future evidence needs at least two timestamp groups")
        selected_groups = unique_groups[-group_count:]
        quota, remainder = divmod(cap, group_count)
        selected = []
        for index, group in enumerate(selected_groups):
            candidates = order[ordered_groups == group]
            take = min(len(candidates), quota + (index >= group_count - remainder))
            positions = np.linspace(0, len(candidates) - 1, num=take, dtype=np.int64)
            selected.append(candidates[positions])
        rows = np.concatenate(selected).astype(np.int64, copy=False)
        return rows[np.lexsort((rows, self.groups[rows]))]

    def split(
        self,
        y: Any | None = None,
        *,
        holdout: float = 0.25,
        min_train: int = 100,
        min_valid: int = 40,
        require_class_coverage: bool = False,
        min_valid_class_counts: Mapping[Any, int] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return one strict past/future partition nearest the requested size.

        Candidate boundaries are only positions between distinct timestamps.
        For classification, both sides can be required to contain every source
        class so rank metrics cannot silently evaluate a reduced task.
        """
        if not 0.0 < float(holdout) < 1.0:
            raise ValueError("future-validation holdout must lie between zero and one")
        order = self.order
        ordered_groups = self.groups[order]
        changes: NDArray[np.int64] = (
            np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]).astype(np.int64, copy=False) + 1
        )
        minimum_train = max(1, int(min_train))
        minimum_valid = max(1, int(min_valid))
        candidates = changes[(changes >= minimum_train) & (changes <= len(order) - minimum_valid)]
        if not len(candidates):
            raise ValueError("no distinct timestamp boundary can support the future holdout")

        target = int(round((1.0 - float(holdout)) * len(order)))
        candidates = candidates[np.lexsort((candidates, np.abs(candidates.astype(np.int64) - target)))]
        target_values: NDArray[Any] | None = None if y is None else np.asarray(y)
        minimum_counts = {label: int(count) for label, count in (min_valid_class_counts or {}).items()}
        if any(count < 0 for count in minimum_counts.values()):
            raise ValueError("future-validation class-count floors cannot be negative")
        if minimum_counts and target_values is None:
            raise ValueError("future-validation class-count floors require a target")
        if target_values is not None:
            if target_values.ndim != 1 or len(target_values) != len(order):
                raise ValueError("future-validation target must align with timestamp groups")
            source_classes = np.unique(target_values) if require_class_coverage else None
        else:
            source_classes = None

        for cutoff in candidates:
            train = order[: int(cutoff)]
            valid = order[int(cutoff) :]
            if source_classes is not None:
                assert target_values is not None
                if not np.array_equal(np.unique(target_values[train]), source_classes):
                    continue
                if not np.array_equal(np.unique(target_values[valid]), source_classes):
                    continue
            if target_values is not None and any(
                np.count_nonzero(target_values[valid] == label) < count
                for label, count in minimum_counts.items()
            ):
                continue
            if self.groups[train].max() >= self.groups[valid].min():
                continue
            return train, valid
        constraints = []
        if source_classes is not None:
            constraints.append("full class coverage")
        if minimum_counts:
            constraints.append("validation class-count floors")
        reason = f" with {' and '.join(constraints)}" if constraints else ""
        raise ValueError(f"no strict future holdout exists{reason}")

    def expanding_splits(
        self,
        y: Any | None = None,
        *,
        folds: int = 3,
        warmup: float = 0.4,
        min_train: int = 20,
        min_valid: int = 5,
        require_train_class_coverage: bool = False,
    ) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Return expanding-past, disjoint-future validation blocks.

        The first block starts near ``warmup`` of the rows. Later blocks add
        every previously observed block to their training prefix. Boundaries
        always fall between distinct timestamps; fewer folds are returned only
        when the requested geometry cannot satisfy the row floors.
        """
        requested_folds = int(folds)
        if requested_folds < 1:
            raise ValueError("future-validation folds must be positive")
        if not 0.0 < float(warmup) < 1.0:
            raise ValueError("future-validation warmup must lie between zero and one")
        minimum_train = max(1, int(min_train))
        minimum_valid = max(1, int(min_valid))
        order = self.order
        ordered_groups = self.groups[order]
        changes: NDArray[np.int64] = (
            np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]).astype(np.int64, copy=False) + 1
        )
        target_values: NDArray[Any] | None = None if y is None else np.asarray(y)
        if target_values is not None and (target_values.ndim != 1 or len(target_values) != len(order)):
            raise ValueError("future-validation target must align with timestamp groups")
        if require_train_class_coverage and target_values is None:
            raise ValueError("future-validation class coverage requires a target")
        source_classes = (
            np.unique(target_values) if require_train_class_coverage and target_values is not None else None
        )
        warmup_target = int(round(float(warmup) * len(order)))

        for fold_count in range(requested_folds, 0, -1):
            latest_start = len(order) - fold_count * minimum_valid
            starts = changes[(changes >= minimum_train) & (changes <= latest_start)]
            if source_classes is not None:
                assert target_values is not None
                starts = np.asarray(
                    [
                        cutoff
                        for cutoff in starts
                        if np.array_equal(np.unique(target_values[order[: int(cutoff)]]), source_classes)
                    ],
                    dtype=np.int64,
                )
            if not len(starts):
                continue
            starts = starts[np.lexsort((starts, np.abs(starts.astype(np.int64) - warmup_target)))]
            for first in starts:
                endpoints: list[int] = []
                cursor = int(first)
                feasible = True
                for step in range(1, fold_count):
                    remaining_blocks = fold_count - step
                    desired = int(first) + int(round((len(order) - int(first)) * step / fold_count))
                    candidates = changes[
                        (changes >= cursor + minimum_valid)
                        & (changes <= len(order) - remaining_blocks * minimum_valid)
                    ]
                    if not len(candidates):
                        feasible = False
                        break
                    endpoint = int(
                        candidates[np.lexsort((candidates, np.abs(candidates.astype(np.int64) - desired)))[0]]
                    )
                    endpoints.append(endpoint)
                    cursor = endpoint
                if not feasible or len(order) - cursor < minimum_valid:
                    continue
                endpoints.append(len(order))
                splits = []
                start = int(first)
                for endpoint in endpoints:
                    train = order[:start]
                    valid = order[start:endpoint]
                    if self.groups[train].max() >= self.groups[valid].min():
                        feasible = False
                        break
                    splits.append((train, valid))
                    start = endpoint
                if feasible:
                    return splits
        raise ValueError("no expanding future-validation geometry satisfies the requested row floors")

    def report(self) -> dict[str, Any]:
        """Return compact serializable metadata without retaining row timestamps."""
        ordered = self.groups[self.order]
        return {
            "mode": "strict_future_holdout",
            "source_rows": int(len(ordered)),
            "timestamp_groups": int(np.unique(ordered).size),
            "first_timestamp_ns": int(ordered[0]),
            "last_timestamp_ns": int(ordered[-1]),
            "same_timestamp_rows_are_atomic": True,
        }


__all__ = ["FutureValidation"]
