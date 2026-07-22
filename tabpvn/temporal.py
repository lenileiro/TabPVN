"""Causal, scale-adaptive temporal evidence and finite context for event tables.

The map in this module is a schema compiler, not a predictor. It samples the
Laplace transform of each entity's strictly prior event history with a bounded
bank of leaky integrators, records a depth-two state of exact prior event
transitions, and can compile an MDL-selected probabilistic suffix tree. No
labels, fitted weights, or user-selected windows enter the representation.
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
_CONTEXT_DEPTH = 2
_CONTEXT_MARK_WIDTH = _CONTEXT_DEPTH + 1
_SUFFIX_TREE_DEPTH = 3
_SUFFIX_TREE_ALPHABET = 3
_SUFFIX_TREE_WIDTH = 4
_SUFFIX_TREE_CONTEXTS = sum(_SUFFIX_TREE_ALPHABET**depth for depth in range(_SUFFIX_TREE_DEPTH + 1))
_SUFFIX_TREE_PRIOR = 0.5
_SUFFIX_TREE_MIN_SUPPORT = 8
_SUFFIX_TREE_DEAD_ZONE = 0.25


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


@njit(cache=True, nogil=True)
def _causal_context_scan(
    order: NDArray[np.int64],
    entity_codes: NDArray[np.int64],
    timestamps_ns: NDArray[np.int64],
    event_marks: NDArray[np.float64],
    time_scale_seconds: float,
    initial_lag1: NDArray[np.float64],
    initial_lag2: NDArray[np.float64],
    initial_previous_gaps: NDArray[np.float64],
    initial_times_ns: NDArray[np.int64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Emit depth-two entity context before each atomic timestamp update."""
    lag1 = initial_lag1.copy()
    lag2 = initial_lag2.copy()
    previous_gaps = initial_previous_gaps.copy()
    last_times = initial_times_ns.copy()
    mark_count = event_marks.shape[1]
    output: NDArray[np.float64] = np.zeros(
        (len(entity_codes), 2 + _CONTEXT_MARK_WIDTH * mark_count),
        dtype=np.float64,
    )
    cursor = 0
    while cursor < len(order):
        row = order[cursor]
        entity = entity_codes[row]
        timestamp = timestamps_ns[row]
        previous = last_times[entity]
        elapsed_seconds = 0.0
        if previous != _NAT_NS:
            elapsed_seconds = max(0.0, (float(timestamp) - float(previous)) / _NS_PER_SECOND)

        end = cursor + 1
        while end < len(order):
            candidate = order[end]
            if entity_codes[candidate] != entity or timestamps_ns[candidate] != timestamp:
                break
            end += 1

        for position in range(cursor, end):
            target_row = order[position]
            if previous != _NAT_NS:
                output[target_row, 0] = np.log1p(elapsed_seconds / time_scale_seconds)
                output[target_row, 1] = np.log1p(previous_gaps[entity] / time_scale_seconds)
            for channel in range(mark_count):
                offset = 2 + _CONTEXT_MARK_WIDTH * channel
                output[target_row, offset] = lag1[entity, channel]
                output[target_row, offset + 1] = lag2[entity, channel]
                output[target_row, offset + 2] = lag1[entity, channel] * lag2[entity, channel]

        if previous != _NAT_NS:
            previous_gaps[entity] = elapsed_seconds
        for channel in range(mark_count):
            total = 0.0
            for position in range(cursor, end):
                total += event_marks[order[position], channel]
            lag2[entity, channel] = lag1[entity, channel]
            lag1[entity, channel] = total / float(end - cursor)
        last_times[entity] = timestamp
        cursor = end
    return output, lag1, lag2, previous_gaps, last_times


@njit(cache=True, nogil=True)
def _suffix_context_index(history: NDArray[np.int8], depth: int) -> int:
    """Encode the newest-first suffix into its fixed trie layer."""
    offset = 0
    width = 1
    for _level in range(depth):
        offset += width
        width *= _SUFFIX_TREE_ALPHABET
    code = 0
    for lag in range(depth):
        symbol = int(history[lag])
        if symbol < 0:
            return -1
        code = code * _SUFFIX_TREE_ALPHABET + symbol
    return offset + code


@njit(cache=True, nogil=True)
def _suffix_tree_read(
    counts: NDArray[np.int64],
    history: NDArray[np.int8],
) -> tuple[float, float, float, float]:
    """Read the deepest local suffix whose likelihood gain pays its MDL cost."""
    root = counts[0]
    root_support = float(root.sum())
    posterior: NDArray[np.float64] = np.empty(_SUFFIX_TREE_ALPHABET, dtype=np.float64)
    for symbol in range(_SUFFIX_TREE_ALPHABET):
        posterior[symbol] = (root[symbol] + _SUFFIX_TREE_PRIOR) / (
            root_support + _SUFFIX_TREE_ALPHABET * _SUFFIX_TREE_PRIOR
        )
    selected = posterior.copy()
    selected_depth = 0
    selected_support = root_support
    for depth in range(1, _SUFFIX_TREE_DEPTH + 1):
        context = _suffix_context_index(history, depth)
        if context < 0:
            break
        bucket = counts[context]
        support = float(bucket.sum())
        child: NDArray[np.float64] = np.empty(_SUFFIX_TREE_ALPHABET, dtype=np.float64)
        for symbol in range(_SUFFIX_TREE_ALPHABET):
            child[symbol] = (bucket[symbol] + posterior[symbol]) / (support + 1.0)
        if support >= _SUFFIX_TREE_MIN_SUPPORT:
            likelihood_gain = 0.0
            for symbol in range(_SUFFIX_TREE_ALPHABET):
                if bucket[symbol] > 0:
                    empirical = bucket[symbol] / support
                    likelihood_gain += bucket[symbol] * np.log(empirical / max(posterior[symbol], 1e-15))
            mdl_penalty = 0.5 * (_SUFFIX_TREE_ALPHABET - 1) * np.log(support)
            if likelihood_gain > mdl_penalty:
                selected = child.copy()
                selected_depth = depth
                selected_support = support
        posterior = child

    entropy = 0.0
    for symbol in range(_SUFFIX_TREE_ALPHABET):
        entropy -= selected[symbol] * np.log(max(selected[symbol], 1e-15))
    return (
        selected[2] - selected[0],
        entropy / np.log(float(_SUFFIX_TREE_ALPHABET)),
        selected_support / (selected_support + 1.0),
        selected_depth / float(_SUFFIX_TREE_DEPTH),
    )


@njit(cache=True, nogil=True)
def _suffix_tree_update(
    counts: NDArray[np.int64],
    history: NDArray[np.int8],
    mean_mark: float,
) -> None:
    """Append one timestamp-atomic transition to every available suffix."""
    symbol = 1
    if mean_mark < -_SUFFIX_TREE_DEAD_ZONE:
        symbol = 0
    elif mean_mark > _SUFFIX_TREE_DEAD_ZONE:
        symbol = 2
    counts[0, symbol] += 1
    for depth in range(1, _SUFFIX_TREE_DEPTH + 1):
        context = _suffix_context_index(history, depth)
        if context < 0:
            break
        counts[context, symbol] += 1
    for lag in range(_SUFFIX_TREE_DEPTH - 1, 0, -1):
        history[lag] = history[lag - 1]
    history[0] = symbol


@njit(cache=True, nogil=True)
def _causal_suffix_tree_scan(
    order: NDArray[np.int64],
    entity_codes: NDArray[np.int64],
    timestamps_ns: NDArray[np.int64],
    event_marks: NDArray[np.float64],
    initial_counts: NDArray[np.int64],
    initial_history: NDArray[np.int8],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int8]]:
    """Emit a prequential variable-order context posterior before each update."""
    counts = initial_counts.copy()
    history = initial_history.copy()
    mark_count = event_marks.shape[1]
    output: NDArray[np.float64] = np.zeros(
        (len(entity_codes), mark_count * _SUFFIX_TREE_WIDTH),
        dtype=np.float64,
    )
    cursor = 0
    while cursor < len(order):
        timestamp = timestamps_ns[order[cursor]]
        end = cursor + 1
        while end < len(order) and timestamps_ns[order[end]] == timestamp:
            end += 1

        # All entities at one timestamp read the same pre-update population
        # trie. Rows sharing an entity/timestamp also read the same suffix.
        for position in range(cursor, end):
            row = order[position]
            entity = entity_codes[row]
            for channel in range(mark_count):
                offset = channel * _SUFFIX_TREE_WIDTH
                read = _suffix_tree_read(counts[channel], history[entity, channel])
                for feature in range(_SUFFIX_TREE_WIDTH):
                    output[row, offset + feature] = read[feature]

        # Timestamp batches update only after every row has been emitted. The
        # batch mean becomes one transition symbol per entity and channel.
        position = cursor
        while position < end:
            entity = entity_codes[order[position]]
            entity_end = position + 1
            while entity_end < end and entity_codes[order[entity_end]] == entity:
                entity_end += 1
            for channel in range(mark_count):
                mean_mark = 0.0
                for batch_position in range(position, entity_end):
                    mean_mark += event_marks[order[batch_position], channel]
                mean_mark /= float(entity_end - position)
                _suffix_tree_update(
                    counts[channel],
                    history[entity, channel],
                    mean_mark,
                )
            position = entity_end
        cursor = end
    return output, counts, history


class TemporalLaplaceMap:
    """Compile strictly prior events into bounded Laplace and context facts.

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
        *,
        _context_state: bool = True,
        _context_tree: bool = False,
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
        self._context_state_enabled = bool(_context_state)
        self._context_tree_enabled = bool(_context_tree)
        if self._context_tree_enabled and not self._context_state_enabled:
            raise ValueError("probabilistic suffix context requires the finite context state")
        if self._context_tree_enabled and not self.value_columns:
            raise ValueError("probabilistic suffix context requires at least one marked value column")

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
        context_width = self._context_feature_count()
        for scale_count in range(_MAX_SCALES, 0, -1):
            laplace_width = channel_count * (2 * scale_count - 1)
            if laplace_width + context_width <= _MAX_EVIDENCE_FEATURES:
                return scale_count
        raise RuntimeError("temporal evidence budget cannot represent one scale")

    def _context_feature_count(self) -> int:
        if not self._context_state_enabled:
            return 0
        suffix_width = _SUFFIX_TREE_WIDTH * len(self.value_columns) if self._context_tree_enabled else 0
        return 2 + _CONTEXT_MARK_WIDTH * len(self.value_columns) + suffix_width

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
        if self._context_state_enabled:
            time_scale = float(self.scales_seconds_[0])
            for suffix, kind in (
                ("time_since_last", "context_current_gap"),
                ("previous_interarrival", "context_previous_gap"),
            ):
                name = f"{self.entity}__history_{suffix}"
                names.append(name)
                metadata.append(
                    {
                        "name": name,
                        "kind": kind,
                        "time_scale_seconds": time_scale,
                        "normalization": "log1p(seconds / time_scale)",
                        "causal_scope": "strictly_prior_timestamp",
                    }
                )
            for column in self.value_columns:
                for lag in range(1, _CONTEXT_DEPTH + 1):
                    name = f"{column}__history_context_lag{lag}"
                    names.append(name)
                    metadata.append(
                        {
                            "name": name,
                            "kind": "context_mark_lag",
                            "channel": str(column),
                            "lag": lag,
                            "value_scale": float(self.value_scales_[column]),
                            "value_clip": _MARK_CLIP,
                            "normalization": "timestamp_batch_mean(clip(value / value_scale))",
                            "causal_scope": "strictly_prior_timestamp",
                        }
                    )
                name = f"{column}__history_context_transition"
                names.append(name)
                metadata.append(
                    {
                        "name": name,
                        "kind": "context_mark_transition",
                        "channel": str(column),
                        "lags": [1, 2],
                        "value_scale": float(self.value_scales_[column]),
                        "value_clip": _MARK_CLIP,
                        "normalization": "normalized_lag1 * normalized_lag2",
                        "causal_scope": "strictly_prior_timestamp",
                    }
                )
            if self._context_tree_enabled:
                for column in self.value_columns:
                    prefix = f"{column}__history_suffix"
                    declarations = (
                        ("expectation", "probabilistic_suffix_expectation"),
                        ("entropy", "probabilistic_suffix_entropy"),
                        ("confidence", "probabilistic_suffix_confidence"),
                        ("depth", "probabilistic_suffix_depth"),
                    )
                    for suffix, kind in declarations:
                        name = f"{prefix}_{suffix}"
                        names.append(name)
                        metadata.append(
                            {
                                "name": name,
                                "kind": kind,
                                "channel": str(column),
                                "maximum_depth": _SUFFIX_TREE_DEPTH,
                                "alphabet": ["negative", "neutral", "positive"],
                                "dead_zone": _SUFFIX_TREE_DEAD_ZONE,
                                "selection": "local_bic_mdl_gain",
                                "minimum_support": _SUFFIX_TREE_MIN_SUPPORT,
                                "causal_scope": "strictly_prior_global_timestamp",
                            }
                        )
        return names, metadata

    def _set_fitted_state(
        self,
        keys: tuple[Hashable, ...],
        states: NDArray[np.float64],
        times_ns: NDArray[np.int64],
        context_lag1: NDArray[np.float64],
        context_lag2: NDArray[np.float64],
        context_previous_gaps: NDArray[np.float64],
        suffix_counts: NDArray[np.int64],
        suffix_history: NDArray[np.int8],
        suffix_time_ns: int,
        n_rows: int,
    ) -> None:
        self.entity_keys_ = keys
        self._entity_index_ = {key: index for index, key in enumerate(keys)}
        self._states_ = states
        self._state_times_ns_ = times_ns
        self._context_lag1_ = context_lag1
        self._context_lag2_ = context_lag2
        self._context_previous_gaps_ = context_previous_gaps
        self._suffix_counts_ = suffix_counts
        self._suffix_history_ = suffix_history
        self._suffix_time_ns_ = int(suffix_time_ns)
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
            "context_state": {
                "enabled": self._context_state_enabled,
                "depth": _CONTEXT_DEPTH if self._context_state_enabled else 0,
                "features": self._context_feature_count(),
                "time_scale_seconds": (
                    float(self.scales_seconds_[0]) if self._context_state_enabled else None
                ),
            },
            "context_tree": {
                "enabled": self._context_tree_enabled,
                "maximum_depth": _SUFFIX_TREE_DEPTH if self._context_tree_enabled else 0,
                "alphabet_size": _SUFFIX_TREE_ALPHABET if self._context_tree_enabled else 0,
                "contexts_per_channel": _SUFFIX_TREE_CONTEXTS if self._context_tree_enabled else 0,
                "features": (
                    _SUFFIX_TREE_WIDTH * len(self.value_columns) if self._context_tree_enabled else 0
                ),
                "minimum_support": _SUFFIX_TREE_MIN_SUPPORT if self._context_tree_enabled else 0,
                "selection": "local_bic_mdl_gain" if self._context_tree_enabled else None,
            },
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
        mark_count = len(self.value_columns)
        if self._context_state_enabled:
            empty_marks: NDArray[np.float64] = np.zeros((len(keys), mark_count), dtype=np.float64)
            empty_gaps: NDArray[np.float64] = np.zeros(len(keys), dtype=np.float64)
            context, lag1, lag2, previous_gaps, context_times = _causal_context_scan(
                order,
                entity_codes,
                timestamps_ns,
                event_weights[:, 1:],
                float(self.scales_seconds_[0]),
                empty_marks,
                empty_marks,
                empty_gaps,
                initial_times,
            )
            if not np.array_equal(context_times, times_ns):
                raise RuntimeError("temporal context and Laplace scans diverged")
            output = np.column_stack((output, context))
        else:
            lag1 = np.zeros((len(keys), mark_count), dtype=np.float64)
            lag2 = np.zeros_like(lag1)
            previous_gaps = np.zeros(len(keys), dtype=np.float64)
        if self._context_tree_enabled:
            suffix_order = np.lexsort(
                (np.arange(len(entity_codes), dtype=np.int64), entity_codes, timestamps_ns)
            ).astype(np.int64, copy=False)
            initial_suffix_counts: NDArray[np.int64] = np.zeros(
                (mark_count, _SUFFIX_TREE_CONTEXTS, _SUFFIX_TREE_ALPHABET),
                dtype=np.int64,
            )
            initial_suffix_history: NDArray[np.int8] = np.full(
                (len(keys), mark_count, _SUFFIX_TREE_DEPTH),
                -1,
                dtype=np.int8,
            )
            suffix, suffix_counts, suffix_history = _causal_suffix_tree_scan(
                suffix_order,
                entity_codes,
                timestamps_ns,
                event_weights[:, 1:],
                initial_suffix_counts,
                initial_suffix_history,
            )
            output = np.column_stack((output, suffix))
            suffix_time_ns = int(timestamps_ns.max())
        else:
            suffix_counts = np.zeros(
                (mark_count, _SUFFIX_TREE_CONTEXTS, _SUFFIX_TREE_ALPHABET),
                dtype=np.int64,
            )
            suffix_history = np.full(
                (len(keys), mark_count, _SUFFIX_TREE_DEPTH),
                -1,
                dtype=np.int8,
            )
            suffix_time_ns = _NAT_NS
        self._set_fitted_state(
            keys,
            states,
            times_ns,
            lag1,
            lag2,
            previous_gaps,
            suffix_counts,
            suffix_history,
            suffix_time_ns,
            len(entity_codes),
        )
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

        output, _, output_times = _causal_laplace_scan(
            order,
            entity_codes,
            timestamps_ns,
            event_weights,
            self.scales_seconds_,
            initial_states,
            initial_times,
        )
        if self._context_state_enabled:
            mark_count = len(self.value_columns)
            initial_lag1: NDArray[np.float64] = np.zeros((len(keys), mark_count), dtype=np.float64)
            initial_lag2: NDArray[np.float64] = np.zeros_like(initial_lag1)
            initial_previous_gaps: NDArray[np.float64] = np.zeros(len(keys), dtype=np.float64)
            for query_index, key in enumerate(keys):
                fitted_index = self._entity_index_.get(key)
                if fitted_index is None:
                    continue
                initial_lag1[query_index] = self._context_lag1_[fitted_index]
                initial_lag2[query_index] = self._context_lag2_[fitted_index]
                initial_previous_gaps[query_index] = self._context_previous_gaps_[fitted_index]
            context, _, _, _, context_times = _causal_context_scan(
                order,
                entity_codes,
                timestamps_ns,
                event_weights[:, 1:],
                float(self.scales_seconds_[0]),
                initial_lag1,
                initial_lag2,
                initial_previous_gaps,
                initial_times,
            )
            if not np.array_equal(context_times, output_times):
                raise RuntimeError("temporal context and Laplace scans diverged")
            output = np.column_stack((output, context))
        if self._context_tree_enabled:
            if int(timestamps_ns.min()) <= self._suffix_time_ns_:
                raise ValueError(
                    "probabilistic suffix-tree prediction events must be strictly later "
                    "than the fitted global timestamp boundary"
                )
            mark_count = len(self.value_columns)
            initial_suffix_history: NDArray[np.int8] = np.full(
                (len(keys), mark_count, _SUFFIX_TREE_DEPTH),
                -1,
                dtype=np.int8,
            )
            for query_index, key in enumerate(keys):
                fitted_index = self._entity_index_.get(key)
                if fitted_index is not None:
                    initial_suffix_history[query_index] = self._suffix_history_[fitted_index]
            suffix_order = np.lexsort(
                (np.arange(len(entity_codes), dtype=np.int64), entity_codes, timestamps_ns)
            ).astype(np.int64, copy=False)
            suffix, _, _ = _causal_suffix_tree_scan(
                suffix_order,
                entity_codes,
                timestamps_ns,
                event_weights[:, 1:],
                self._suffix_counts_,
                initial_suffix_history,
            )
            output = np.column_stack((output, suffix))
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
