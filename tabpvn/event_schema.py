"""Deterministic, bounded discovery of entity-event table semantics.

Discovery is deliberately unlabeled. It proposes a small number of plausible
``(entity, timestamp, marked values)`` schemas; the temporal proposer remains
responsible for deciding whether any proposal improves across future windows.
"""

from __future__ import annotations

import re
from collections.abc import Hashable
from dataclasses import dataclass
from math import log1p
from typing import Any

import numpy as np
from numpy.typing import NDArray

_DISCOVERY_MIN_ROWS = 200
_DISCOVERY_MAX_ROWS = 100_000
_DISCOVERY_MAX_CELLS = 5_000_000
_MAX_TIMESTAMP_CANDIDATES = 2
_MAX_ENTITY_PAIRS = 2
_MIN_ELIGIBLE_COVERAGE = 0.20

_TIMESTAMP_NAMES = {
    "date",
    "datetime",
    "event_date",
    "event_datetime",
    "event_time",
    "event_timestamp",
    "occurred_at",
    "time",
    "timestamp",
}
_TIMESTAMP_SUFFIXES = ("_at", "_date", "_datetime", "_time", "_timestamp")
_ENTITY_NAMES = {
    "account",
    "account_id",
    "card_id",
    "client_id",
    "customer",
    "customer_id",
    "device_id",
    "entity",
    "entity_id",
    "ip",
    "ip_address",
    "member_id",
    "merchant_id",
    "patient_id",
    "session_id",
    "user",
    "user_id",
}
_VALUE_TOKENS = {
    "amount",
    "balance",
    "cost",
    "count",
    "duration",
    "failed",
    "price",
    "quantity",
    "qty",
    "score",
    "size",
    "total",
    "value",
    "volume",
}


def _normalized_name(column: Hashable) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).casefold()).strip("_")


def _timestamp_name_score(column: Hashable) -> float:
    name = _normalized_name(column)
    if name.endswith("up_to_date"):
        return 0.0
    if name in _TIMESTAMP_NAMES:
        return 2.0
    return 1.0 if name.endswith(_TIMESTAMP_SUFFIXES) else 0.0


def _entity_name_score(column: Hashable) -> float:
    name = _normalized_name(column)
    if name in _ENTITY_NAMES or name.endswith(("_entity_id", "_id")):
        return 2.0
    tokens = set(name.split("_"))
    return 1.0 if tokens & {"account", "customer", "device", "entity", "session", "user"} else 0.0


def _value_name_score(column: Hashable) -> float:
    return float(len(set(_normalized_name(column).split("_")) & _VALUE_TOKENS) > 0)


@dataclass(frozen=True, slots=True)
class EventSchemaCandidate:
    """One bounded schema proposal awaiting future-window evaluation."""

    entity: Hashable
    timestamp: Hashable
    value_columns: tuple[Hashable, ...]
    pair_rank: int
    structural_score: float
    eligible_rows: int
    eligible_entities: int
    sample_rows: int
    timestamp_source: str

    def asdict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "timestamp": self.timestamp,
            "value_columns": self.value_columns,
            "pair_rank": int(self.pair_rank),
            "structural_score": float(self.structural_score),
            "eligible_rows": int(self.eligible_rows),
            "eligible_entities": int(self.eligible_entities),
            "sample_rows": int(self.sample_rows),
            "timestamp_source": self.timestamp_source,
        }


@dataclass(frozen=True, slots=True)
class _TimestampCandidate:
    column: Hashable
    values_ns: NDArray[np.int64]
    score: float
    source: str
    column_index: int


def _bounded_row_limit(row_count: int, column_count: int) -> int:
    cell_limited_rows = max(_DISCOVERY_MIN_ROWS, _DISCOVERY_MAX_CELLS // max(column_count, 1))
    return min(row_count, _DISCOVERY_MAX_ROWS, cell_limited_rows)


def _bounded_rows(row_count: int, column_count: int) -> NDArray[np.int64]:
    maximum_rows = _bounded_row_limit(row_count, column_count)
    if row_count <= maximum_rows:
        return np.arange(row_count, dtype=np.int64)
    return np.linspace(0, row_count - 1, num=maximum_rows, dtype=np.int64)


def _bounded_sample(frame: Any) -> Any:
    return frame.iloc[_bounded_rows(len(frame), len(frame.columns))]


def bounded_event_gate(
    data: Any,
    target: Any,
    *,
    timestamp: Hashable | None = None,
) -> tuple[Any, np.ndarray]:
    """Return a bounded gate table, using a dense recent stream when possible.

    Schema discovery needs broad table coverage and therefore uses spread rows.
    Causal evidence instead needs the original event density, so a discovered
    timestamp selects the latest timestamp-atomic window.
    """
    target_values = np.asarray(target)
    if target_values.ndim != 1 or len(target_values) != len(data):
        raise ValueError("automatic event gate target must be one-dimensional and aligned")
    if timestamp is None or len(data) <= _DISCOVERY_MIN_ROWS:
        rows = _bounded_rows(len(data), len(data.columns))
        return data.iloc[rows], target_values[rows]

    import pandas as pd

    series = data[timestamp]
    options = {} if pd.api.types.is_datetime64_any_dtype(series.dtype) else {"format": "mixed"}
    parsed = pd.to_datetime(series, errors="coerce", utc=True, **options)
    if parsed.isna().any():
        rows = _bounded_rows(len(data), len(data.columns))
        return data.iloc[rows], target_values[rows]
    timestamps_ns = parsed.astype("int64").to_numpy(dtype=np.int64, copy=False)
    row_ids: NDArray[np.int64] = np.arange(len(data), dtype=np.int64)
    order = row_ids[np.lexsort((row_ids, timestamps_ns))]
    maximum_rows = _bounded_row_limit(len(data), len(data.columns))
    if len(order) > maximum_rows:
        ordered_times = timestamps_ns[order]
        requested = len(order) - maximum_rows
        boundary = ordered_times[requested]
        left = int(np.searchsorted(ordered_times, boundary, side="left"))
        right = int(np.searchsorted(ordered_times, boundary, side="right"))
        start = left if len(order) - left <= maximum_rows else right
        order = order[start:]
    return data.iloc[order], target_values[order]


def _could_contain_timestamp(frame: Any) -> bool:
    import pandas as pd

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return True
        if _timestamp_name_score(column) > 0.0 and not pd.api.types.is_numeric_dtype(series.dtype):
            return True
    return False


def _timestamp_candidates(frame: Any) -> tuple[_TimestampCandidate, ...]:
    import pandas as pd

    candidates = []
    for column_index, column in enumerate(frame.columns):
        series = frame[column]
        native = pd.api.types.is_datetime64_any_dtype(series.dtype)
        name_score = _timestamp_name_score(column)
        if not native and (name_score == 0.0 or pd.api.types.is_numeric_dtype(series.dtype)):
            continue
        options = {} if native else {"format": "mixed"}
        parsed = pd.to_datetime(series, errors="coerce", utc=True, **options)
        if parsed.isna().any() or parsed.nunique(dropna=True) < 3:
            continue
        values_ns = parsed.astype("int64").to_numpy(dtype=np.int64, copy=False)
        if int(values_ns.max()) <= int(values_ns.min()):
            continue
        candidates.append(
            _TimestampCandidate(
                column=column,
                values_ns=values_ns,
                score=(3.0 if native else 1.0) + name_score,
                source="datetime_dtype" if native else "named_parseable_text",
                column_index=column_index,
            )
        )
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.column_index))
    return tuple(candidates[:_MAX_TIMESTAMP_CANDIDATES])


def _entity_eligible(series: Any) -> bool:
    import pandas as pd

    if series.isna().any() or pd.api.types.is_bool_dtype(series.dtype):
        return False
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return False
    if pd.api.types.is_float_dtype(series.dtype):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return bool(np.isfinite(values).all() and np.equal(values, np.floor(values)).all())
    return bool(
        pd.api.types.is_integer_dtype(series.dtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _entity_pair_score(
    series: Any,
    timestamps_ns: np.ndarray,
    column: Hashable,
) -> tuple[float, int, int] | None:
    import pandas as pd

    if not _entity_eligible(series):
        return None
    try:
        codes, entities = pd.factorize(series, sort=False)
    except TypeError:
        return None
    entity_count = len(entities)
    if entity_count < 2 or entity_count > len(series) // 2 or np.any(codes < 0):
        return None
    counts = np.bincount(codes, minlength=entity_count)
    minimum: NDArray[np.int64] = np.full(entity_count, np.iinfo(np.int64).max, dtype=np.int64)
    maximum: NDArray[np.int64] = np.full(entity_count, np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(minimum, codes, timestamps_ns)
    np.maximum.at(maximum, codes, timestamps_ns)
    eligible = (counts >= 3) & (maximum > minimum)
    eligible_entities = int(eligible.sum())
    eligible_rows = int(counts[eligible].sum())
    required_rows = max(_DISCOVERY_MIN_ROWS, int(np.ceil(_MIN_ELIGIBLE_COVERAGE * len(series))))
    if eligible_entities < 2 or eligible_rows < required_rows:
        return None
    coverage = eligible_rows / len(series)
    diversity = log1p(eligible_entities) / log1p(len(series))
    repeat_balance = min(1.0, len(series) / max(3.0 * entity_count, 1.0))
    score = 4.0 * coverage + 2.0 * diversity + repeat_balance + 2.0 * _entity_name_score(column)
    return score, eligible_rows, eligible_entities


def _marked_values(frame: Any, *, entity: Hashable, timestamp: Hashable) -> tuple[Hashable, ...]:
    import pandas as pd

    ranked = []
    for column_index, column in enumerate(frame.columns):
        if column in {entity, timestamp}:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(series.dtype):
            continue
        if _entity_name_score(column) >= 2.0:
            continue
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.mean() < 0.80 or len(np.unique(values[finite])) < 3 or not np.any(values[finite] != 0.0):
            continue
        name_score = _value_name_score(column)
        integer_coded = np.equal(values[finite], np.floor(values[finite])).all()
        if name_score == 0.0 and integer_coded:
            continue
        diversity = log1p(len(np.unique(values[finite]))) / log1p(len(values) + 1)
        score = 2.0 * name_score + diversity + float(finite.mean())
        ranked.append((score, name_score, column_index, column))
    ranked.sort(key=lambda item: (-item[0], item[2]))
    if not ranked:
        return ()
    selected = [ranked[0][3]]
    if len(ranked) > 1 and ranked[1][1] > 0.0:
        selected.append(ranked[1][3])
    return tuple(selected)


def discover_event_schemas(data: Any) -> tuple[EventSchemaCandidate, ...]:
    """Return at most three deterministic, unlabeled event-schema proposals."""
    import pandas as pd

    if not isinstance(data, pd.DataFrame) or len(data) < _DISCOVERY_MIN_ROWS:
        return ()
    if data.columns.duplicated().any():
        return ()
    if len(data.columns) * _DISCOVERY_MIN_ROWS > _DISCOVERY_MAX_CELLS:
        return ()
    if not _could_contain_timestamp(data):
        return ()
    frame = _bounded_sample(data)
    pair_candidates = []
    for timestamp in _timestamp_candidates(frame):
        for column_index, entity in enumerate(frame.columns):
            if entity == timestamp.column:
                continue
            scored = _entity_pair_score(frame[entity], timestamp.values_ns, entity)
            if scored is None:
                continue
            score, eligible_rows, eligible_entities = scored
            pair_candidates.append(
                (
                    -(score + timestamp.score),
                    timestamp.column_index,
                    column_index,
                    entity,
                    timestamp,
                    eligible_rows,
                    eligible_entities,
                )
            )
    pair_candidates.sort(key=lambda item: item[:3])
    proposals = []
    for pair_rank, pair in enumerate(pair_candidates[:_MAX_ENTITY_PAIRS]):
        neg_score, _, _, entity, timestamp, eligible_rows, eligible_entities = pair
        common = {
            "entity": entity,
            "timestamp": timestamp.column,
            "pair_rank": pair_rank,
            "structural_score": -float(neg_score),
            "eligible_rows": eligible_rows,
            "eligible_entities": eligible_entities,
            "sample_rows": len(frame),
            "timestamp_source": timestamp.source,
        }
        proposals.append(EventSchemaCandidate(value_columns=(), **common))
        if pair_rank == 0:
            marked = _marked_values(frame, entity=entity, timestamp=timestamp.column)
            if marked:
                proposals.append(EventSchemaCandidate(value_columns=marked, **common))
    return tuple(proposals)


__all__ = ["EventSchemaCandidate", "bounded_event_gate", "discover_event_schemas"]
