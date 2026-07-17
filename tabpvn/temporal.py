"""Causal, scale-adaptive temporal evidence for event tables.

The map in this module is a schema compiler, not a predictor. It samples the
Laplace transform of each entity's strictly prior event history with a bounded
bank of leaky integrators. No labels, fitted weights, or user-selected windows
enter the representation.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Self

import numpy as np
from numba import njit
from numpy.typing import NDArray

_NAT_NS = np.iinfo(np.int64).min
_NS_PER_SECOND = 1_000_000_000.0
_MAX_SCALES = 8
_MAX_VALUE_COLUMNS = 2
_MAX_EVIDENCE_FEATURES = 32
_MARK_CLIP = 8.0


@njit(cache=True, nogil=True)
def _causal_laplace_scan(
    order: NDArray[np.int64],
    entity_codes: NDArray[np.int64],
    timestamps_ns: NDArray[np.int64],
    event_weights: NDArray[np.float64],
    scales_seconds: NDArray[np.float64],
    initial_states: NDArray[np.float64],
    initial_times_ns: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Scan entity/time-sorted rows and emit state before each timestamp batch."""
    states = initial_states.copy()
    last_times = initial_times_ns.copy()
    channel_count = event_weights.shape[1]
    scale_count = len(scales_seconds)
    channel_width = 2 * scale_count - 1
    output: NDArray[np.float64] = np.zeros(
        (len(entity_codes), channel_count * channel_width), dtype=np.float64
    )
    cursor = 0
    while cursor < len(order):
        row = order[cursor]
        entity = entity_codes[row]
        timestamp = timestamps_ns[row]
        previous = last_times[entity]
        if previous == _NAT_NS:
            last_times[entity] = timestamp
        else:
            elapsed_seconds = (float(timestamp) - float(previous)) / _NS_PER_SECOND
            for channel in range(channel_count):
                for scale in range(scale_count):
                    states[entity, channel, scale] *= np.exp(-elapsed_seconds / scales_seconds[scale])

        end = cursor + 1
        while end < len(order):
            candidate = order[end]
            if entity_codes[candidate] != entity or timestamps_ns[candidate] != timestamp:
                break
            end += 1

        for position in range(cursor, end):
            target_row = order[position]
            for channel in range(channel_count):
                offset = channel * channel_width
                for scale in range(scale_count):
                    output[target_row, offset + scale] = states[entity, channel, scale]
                for scale in range(scale_count - 1):
                    output[target_row, offset + scale_count + scale] = (
                        states[entity, channel, scale + 1] - states[entity, channel, scale]
                    )

        for position in range(cursor, end):
            source_row = order[position]
            for channel in range(channel_count):
                contribution = event_weights[source_row, channel]
                for scale in range(scale_count):
                    states[entity, channel, scale] += contribution
        last_times[entity] = timestamp
        cursor = end
    return output, states, last_times


class TemporalLaplaceMap:
    """Compile strictly prior event counts into bounded Laplace-history facts.

    Parameters identify semantic columns, not tuning choices. Up to eight
    geometric timescales are derived from positive within-entity gaps and
    entity history spans. ``fit_transform`` returns training-row history and
    stores each entity's final state. ``transform`` starts from that immutable
    fitted state, so repeated prediction calls are deterministic.
    """

    def __init__(
        self,
        entity: Hashable,
        timestamp: Hashable,
        value_columns: Sequence[Hashable] | None = None,
    ) -> None:
        if not isinstance(entity, Hashable) or not isinstance(timestamp, Hashable):
            raise TypeError("entity and timestamp must be hashable DataFrame column labels")
        if entity == timestamp:
            raise ValueError("entity and timestamp must identify different columns")
        if isinstance(value_columns, (str, bytes)):
            raise TypeError("value_columns must be a sequence of DataFrame column labels, not a string")
        values = () if value_columns is None else tuple(value_columns)
        if len(values) > _MAX_VALUE_COLUMNS:
            raise ValueError(f"value_columns supports at most {_MAX_VALUE_COLUMNS} marked channels")
        if any(not isinstance(column, Hashable) for column in values):
            raise TypeError("every marked value column label must be hashable")
        if len(values) != len(set(values)):
            raise ValueError("value_columns must not contain duplicates")
        if entity in values or timestamp in values:
            raise ValueError("value_columns must not repeat the entity or timestamp column")
        self.entity = entity
        self.timestamp = timestamp
        self.value_columns = values

    @staticmethod
    def _as_frame(events: Any) -> Any:
        import pandas as pd

        if not isinstance(events, pd.DataFrame):
            raise TypeError("TemporalLaplaceMap requires a pandas DataFrame")
        if events.columns.duplicated().any():
            duplicates = list(events.columns[events.columns.duplicated()])
            raise ValueError(f"event DataFrame has duplicate columns: {duplicates[:5]}")
        if len(events) == 0:
            raise ValueError("event DataFrame must contain at least one row")
        return events

    def _extract(self, events: Any) -> tuple[Any, NDArray[np.int64], tuple[Hashable, ...], NDArray[np.int64]]:
        import pandas as pd

        frame = self._as_frame(events)
        missing_columns = [column for column in (self.entity, self.timestamp) if column not in frame]
        if missing_columns:
            raise ValueError(f"event DataFrame is missing semantic columns: {missing_columns}")

        entities = frame[self.entity]
        if entities.isna().any():
            raise ValueError(f"entity column {self.entity!r} contains missing values")
        try:
            codes, unique_entities = pd.factorize(entities, sort=False)
        except TypeError as error:
            raise ValueError(f"entity column {self.entity!r} contains unhashable values") from error
        if np.any(codes < 0):
            raise ValueError(f"entity column {self.entity!r} contains unsupported values")

        raw_time = frame[self.timestamp]
        if pd.api.types.is_numeric_dtype(raw_time.dtype) and not pd.api.types.is_datetime64_any_dtype(
            raw_time.dtype
        ):
            raise TypeError(
                f"timestamp column {self.timestamp!r} is numeric and has ambiguous units; "
                "convert it to a pandas datetime dtype first"
            )
        parse_options = {"format": "mixed"} if raw_time.dtype == object else {}
        timestamps = pd.to_datetime(raw_time, errors="coerce", utc=True, **parse_options)
        invalid = timestamps.isna().to_numpy()
        if invalid.any():
            raise ValueError(
                f"timestamp column {self.timestamp!r} contains {int(invalid.sum())} missing or invalid values"
            )
        timestamps_ns = timestamps.astype("int64").to_numpy(dtype=np.int64, copy=False)
        keys = tuple(unique_entities.tolist())
        return frame, codes.astype(np.int64, copy=False), keys, timestamps_ns

    @staticmethod
    def _ordered_rows(entity_codes: NDArray[np.int64], timestamps_ns: NDArray[np.int64]) -> NDArray[np.int64]:
        row_ids: NDArray[np.int64] = np.arange(len(entity_codes), dtype=np.int64)
        return np.lexsort((row_ids, timestamps_ns, entity_codes)).astype(np.int64, copy=False)

    @staticmethod
    def _derive_scales(
        entity_codes: NDArray[np.int64],
        timestamps_ns: NDArray[np.int64],
        max_scales: int = _MAX_SCALES,
    ) -> NDArray[np.float64]:
        order = TemporalLaplaceMap._ordered_rows(entity_codes, timestamps_ns)
        ordered_entities = entity_codes[order]
        ordered_times = timestamps_ns[order]
        same_entity = ordered_entities[1:] == ordered_entities[:-1]
        gaps_ns = ordered_times[1:].astype(np.float64) - ordered_times[:-1].astype(np.float64)
        positive_gaps = gaps_ns[same_entity & (gaps_ns > 0.0)]
        if len(positive_gaps) == 0:
            raise ValueError(
                "TemporalLaplaceMap needs at least one entity observed at two distinct timestamps"
            )

        starts = np.r_[0, np.flatnonzero(ordered_entities[1:] != ordered_entities[:-1]) + 1]
        ends = np.r_[starts[1:] - 1, len(order) - 1]
        spans_ns = ordered_times[ends].astype(np.float64) - ordered_times[starts].astype(np.float64)
        positive_spans = spans_ns[spans_ns > 0.0]
        minimum = float(np.quantile(positive_gaps, 0.10) / _NS_PER_SECOND)
        maximum = float(np.quantile(positive_spans, 0.95) / _NS_PER_SECOND)
        maximum = max(maximum, minimum)
        ratio = maximum / minimum
        if ratio <= 1.0 + 1e-12:
            return np.array([minimum], dtype=np.float64)
        scale_count = min(max_scales, max(2, int(np.ceil(np.log2(ratio))) + 1))
        return np.geomspace(minimum, maximum, num=scale_count).astype(np.float64)

    def _fit_event_weights(self, frame: Any) -> NDArray[np.float64]:
        import pandas as pd

        missing = [column for column in self.value_columns if column not in frame]
        if missing:
            raise ValueError(f"event DataFrame is missing marked value columns: {missing}")
        self.value_scales_ = {}
        self.value_missing_rates_ = {}
        for column in self.value_columns:
            series = frame[column]
            if not pd.api.types.is_numeric_dtype(series.dtype):
                raise TypeError(f"marked value column {column!r} must have a numeric dtype")
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            if len(np.unique(values[finite])) < 2:
                raise ValueError(f"marked value column {column!r} must contain at least two finite values")
            nonzero = np.abs(values[finite & (values != 0.0)])
            if len(nonzero) == 0:
                raise ValueError(f"marked value column {column!r} has no non-zero finite values")
            self.value_scales_[column] = float(np.quantile(nonzero, 0.50))
            self.value_missing_rates_[column] = float(1.0 - finite.mean())
        self.channel_names_ = ("event_count",) + self.value_columns
        return self._event_weights(frame)

    def _event_weights(self, frame: Any) -> NDArray[np.float64]:
        import pandas as pd

        missing = [column for column in self.value_columns if column not in frame]
        if missing:
            raise ValueError(f"event DataFrame is missing marked value columns: {missing}")
        weights: NDArray[np.float64] = np.ones((len(frame), 1 + len(self.value_columns)), dtype=np.float64)
        for channel, column in enumerate(self.value_columns, start=1):
            if not pd.api.types.is_numeric_dtype(frame[column].dtype):
                raise TypeError(f"marked value column {column!r} must have a numeric dtype")
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            normalized: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
            normalized[finite] = np.clip(values[finite] / self.value_scales_[column], -_MARK_CLIP, _MARK_CLIP)
            weights[:, channel] = normalized
        return weights

    def _maximum_scale_count(self) -> int:
        channel_count = 1 + len(self.value_columns)
        per_channel_budget = _MAX_EVIDENCE_FEATURES // channel_count
        return min(_MAX_SCALES, max(1, (per_channel_budget + 1) // 2))

    def _feature_declarations(self) -> tuple[list[str], list[dict[str, Any]]]:
        names = []
        metadata = []
        for channel, column in enumerate(self.channel_names_):
            is_count = channel == 0
            prefix = str(self.entity) if is_count else str(column)
            state_kind = "decayed_event_count" if is_count else "decayed_value_sum"
            band_kind = "event_count_age_band" if is_count else "value_age_band"
            for scale_index, scale in enumerate(self.scales_seconds_):
                name = f"{prefix}__history_laplace_{'count' if is_count else 'sum'}_s{scale_index}"
                item = {
                    "name": name,
                    "kind": state_kind,
                    "channel": str(column),
                    "tau_seconds": float(scale),
                    "causal_scope": "strictly_prior_timestamp",
                }
                if not is_count:
                    item.update(
                        {
                            "value_scale": float(self.value_scales_[column]),
                            "value_clip": _MARK_CLIP,
                            "normalization": "clip(value / value_scale)",
                            "missing_contribution": 0.0,
                        }
                    )
                names.append(name)
                metadata.append(item)
            for scale_index in range(len(self.scales_seconds_) - 1):
                name = (
                    f"{prefix}__history_laplace_{'count' if is_count else 'sum'}"
                    f"_band_s{scale_index}_s{scale_index + 1}"
                )
                item = {
                    "name": name,
                    "kind": band_kind,
                    "channel": str(column),
                    "tau_short_seconds": float(self.scales_seconds_[scale_index]),
                    "tau_long_seconds": float(self.scales_seconds_[scale_index + 1]),
                    "causal_scope": "strictly_prior_timestamp",
                }
                if not is_count:
                    item.update(
                        {
                            "value_scale": float(self.value_scales_[column]),
                            "value_clip": _MARK_CLIP,
                            "normalization": "clip(value / value_scale)",
                            "missing_contribution": 0.0,
                        }
                    )
                names.append(name)
                metadata.append(item)
        return names, metadata

    def _set_fitted_state(
        self,
        keys: tuple[Hashable, ...],
        states: NDArray[np.float64],
        times_ns: NDArray[np.int64],
        n_rows: int,
    ) -> None:
        self.entity_keys_ = keys
        self._entity_index_ = {key: index for index, key in enumerate(keys)}
        self._states_ = states
        self._state_times_ns_ = times_ns
        feature_names, metadata = self._feature_declarations()
        self.feature_names_ = np.asarray(feature_names, dtype=object)
        self.feature_metadata_ = tuple(metadata)
        self.report_ = {
            "rows": int(n_rows),
            "entities": int(len(keys)),
            "scales_seconds": [float(scale) for scale in self.scales_seconds_],
            "features": int(len(self.feature_names_)),
            "channels": [str(channel) for channel in self.channel_names_],
            "marked_value_scales": {
                str(column): float(scale) for column, scale in self.value_scales_.items()
            },
            "marked_value_missing_rates": {
                str(column): float(rate) for column, rate in self.value_missing_rates_.items()
            },
            "same_timestamp_policy": "emit_then_update_batch",
        }

    def fit_transform(self, events: Any, y: Any = None) -> NDArray[np.float64]:
        """Fit automatic scales and return history available before each row."""
        del y
        frame, entity_codes, keys, timestamps_ns = self._extract(events)
        event_weights = self._fit_event_weights(frame)
        self.scales_seconds_ = self._derive_scales(
            entity_codes,
            timestamps_ns,
            max_scales=self._maximum_scale_count(),
        )
        order = self._ordered_rows(entity_codes, timestamps_ns)
        initial_states: NDArray[np.float64] = np.zeros(
            (len(keys), len(self.channel_names_), len(self.scales_seconds_)), dtype=np.float64
        )
        initial_times: NDArray[np.int64] = np.full(len(keys), _NAT_NS, dtype=np.int64)
        output, states, times_ns = _causal_laplace_scan(
            order,
            entity_codes,
            timestamps_ns,
            event_weights,
            self.scales_seconds_,
            initial_states,
            initial_times,
        )
        self._set_fitted_state(keys, states, times_ns, len(entity_codes))
        return output

    def fit(self, events: Any, y: Any = None) -> Self:
        """Fit scales and final history state; use ``fit_transform`` for training facts."""
        self.fit_transform(events, y=y)
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "scales_seconds_"):
            raise RuntimeError("TemporalLaplaceMap is not fitted; call fit or fit_transform first")

    def transform(self, events: Any) -> NDArray[np.float64]:
        """Return causal history facts for rows strictly after fitted entity history."""
        self._require_fitted()
        frame, entity_codes, keys, timestamps_ns = self._extract(events)
        event_weights = self._event_weights(frame)
        initial_states: NDArray[np.float64] = np.zeros(
            (len(keys), len(self.channel_names_), len(self.scales_seconds_)), dtype=np.float64
        )
        initial_times: NDArray[np.int64] = np.full(len(keys), _NAT_NS, dtype=np.int64)
        known: NDArray[np.bool_] = np.zeros(len(keys), dtype=bool)
        for query_index, key in enumerate(keys):
            fitted_index = self._entity_index_.get(key)
            if fitted_index is None:
                continue
            known[query_index] = True
            initial_states[query_index] = self._states_[fitted_index]
            initial_times[query_index] = self._state_times_ns_[fitted_index]

        order = self._ordered_rows(entity_codes, timestamps_ns)
        first_for_entity = np.r_[True, entity_codes[order][1:] != entity_codes[order][:-1]]
        first_rows = order[first_for_entity]
        for row in first_rows:
            entity_index = int(entity_codes[row])
            if known[entity_index] and timestamps_ns[row] <= initial_times[entity_index]:
                raise ValueError(
                    "prediction events must be strictly later than fitted history for known entities; "
                    f"entity {keys[entity_index]!r} overlaps the fitted time boundary"
                )

        output, _, _ = _causal_laplace_scan(
            order,
            entity_codes,
            timestamps_ns,
            event_weights,
            self.scales_seconds_,
            initial_states,
            initial_times,
        )
        return output

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Return stable output names; ``input_features`` is accepted for sklearn parity."""
        del input_features
        self._require_fitted()
        return self.feature_names_.copy()

    def _append(self, events: Any, evidence: NDArray[np.float64]) -> Any:
        frame = self._as_frame(events)
        collisions = [name for name in self.feature_names_ if name in frame.columns]
        if collisions:
            raise ValueError(f"temporal evidence columns already exist: {collisions[:5]}")
        augmented = frame.copy(deep=False)
        for index, name in enumerate(self.feature_names_):
            augmented[name] = evidence[:, index]
        return augmented

    def fit_augment(self, events: Any, y: Any = None) -> Any:
        """Fit and return a non-mutating DataFrame augmented with training history."""
        return self._append(events, self.fit_transform(events, y=y))

    def augment(self, events: Any) -> Any:
        """Return a non-mutating DataFrame augmented from fitted history."""
        return self._append(events, self.transform(events))


__all__ = ["TemporalLaplaceMap"]
