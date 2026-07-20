"""Deterministic verifier-relative allocation for bounded candidate searches.

Candidate models are still selected by an absolute held-out metric.  This module
only decides which candidates receive the next search budget.  It compares
paired metric blocks without variance normalization, retains the certified
baseline, and can remove a candidate only when another candidate is at least as
good on every paired evidence block.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_MIN_GROUP_RELATIVE_EVIDENCE_UNITS = 4


@dataclass(frozen=True)
class VerifierScore:
    """A legacy-compatible score tuple with paired primary-metric evidence."""

    values: tuple[Any, ...]
    paired_primary: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("verifier score requires at least one value")

    def __getitem__(self, index: int) -> Any:
        return self.values[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class AllocationDecision:
    """Candidates promoted from one rung and aggregate verifier diagnostics."""

    promoted: tuple[int, ...]
    report: dict[str, Any]


def verification_blocks(
    rows: int,
    labels: np.ndarray | None = None,
    *,
    max_blocks: int = 4,
    min_block_rows: int = 64,
) -> tuple[np.ndarray, ...]:
    """Partition validation rows into deterministic paired metric blocks.

    Classification blocks are stratified so every returned block contains every
    observed class.  The blocks reuse predictions from one fit; they do not add
    training or inference work.
    """

    if rows < 1:
        raise ValueError("verification blocks require at least one row")
    if max_blocks < 1 or min_block_rows < 1:
        raise ValueError("verification block limits must be positive")

    block_count = min(max_blocks, max(1, rows // min_block_rows))
    encoded = None if labels is None else np.asarray(labels)
    if encoded is not None:
        if encoded.ndim != 1 or len(encoded) != rows:
            raise ValueError("labels must contain one value per verification row")
        _classes, counts = np.unique(encoded, return_counts=True)
        block_count = min(block_count, int(counts.min()))
    if block_count < 2:
        return (np.arange(rows, dtype=np.int64),)

    if encoded is None:
        return tuple(np.arange(block, rows, block_count, dtype=np.int64) for block in range(block_count))

    blocks: list[list[np.ndarray]] = [[] for _ in range(block_count)]
    for class_value in np.unique(encoded):
        class_rows = np.flatnonzero(encoded == class_value)
        for block in range(block_count):
            blocks[block].append(class_rows[block::block_count])
    return tuple(np.sort(np.concatenate(parts)).astype(np.int64, copy=False) for parts in blocks)


def score_values(score: Any) -> tuple[Any, ...]:
    """Return the tuple used by the existing absolute score ordering."""

    if isinstance(score, VerifierScore):
        return score.values
    if isinstance(score, tuple):
        return score
    if isinstance(score, list):
        return tuple(score)
    return (score,)


def best_candidate(scores: Mapping[int, Any], candidates: list[int], *, maximize: bool) -> int:
    """Select by the original absolute objective and deterministic tie-breaks."""

    if not candidates:
        raise ValueError("best-candidate selection requires at least one candidate")
    selector = max if maximize else min
    return selector(candidates, key=lambda index: score_values(scores[index]))


def _paired_matrix(scores: Mapping[int, Any], candidates: list[int]) -> np.ndarray:
    paired: list[tuple[float, ...]] = []
    for index in candidates:
        score = scores[index]
        values = score.paired_primary if isinstance(score, VerifierScore) else ()
        if not values:
            values = (float(score_values(score)[0]),)
        paired.append(tuple(float(value) for value in values))

    lengths = {len(values) for values in paired}
    if len(lengths) != 1:
        return np.asarray([[float(score_values(scores[index])[0])] for index in candidates], dtype=float)
    matrix = np.asarray(paired, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        return np.asarray([[float(score_values(scores[index])[0])] for index in candidates], dtype=float)
    return matrix


def _dominated_candidates(
    utility: np.ndarray,
    candidates: list[int],
    baseline_index: int,
) -> set[int]:
    """Return candidates consistently Pareto-dominated on paired evidence."""

    scale = max(1.0, float(np.max(np.abs(utility))))
    tolerance = 32.0 * float(np.finfo(float).eps) * scale
    dominated: set[int] = set()
    for candidate_position, candidate_index in enumerate(candidates):
        if candidate_index == baseline_index:
            continue
        candidate_utility = utility[candidate_position]
        for challenger_position, challenger_index in enumerate(candidates):
            if challenger_index == candidate_index:
                continue
            challenger_utility = utility[challenger_position]
            if np.all(challenger_utility >= candidate_utility - tolerance) and np.any(
                challenger_utility > candidate_utility + tolerance
            ):
                dominated.add(candidate_index)
                break
    return dominated


def _block_winner_counts(utility: np.ndarray) -> np.ndarray:
    """Count verifier blocks won by each candidate, including numerical ties."""

    scale = max(1.0, float(np.max(np.abs(utility))))
    tolerance = 32.0 * float(np.finfo(float).eps) * scale
    block_spread = np.ptp(utility, axis=0)
    informative = block_spread > tolerance
    if not np.any(informative):
        return np.zeros(utility.shape[0], dtype=np.int64)
    best = np.max(utility[:, informative], axis=0)
    winners = utility[:, informative] >= best - tolerance
    return np.sum(winners, axis=1, dtype=np.int64)


def allocate_candidates(
    scores: Mapping[int, Any],
    candidates: list[int],
    baseline_index: int,
    *,
    keep: int,
    maximize: bool,
    prune_dominated: bool,
) -> AllocationDecision:
    """Promote candidates using paired group-relative evidence.

    Relative evidence controls search allocation only.  The final winner remains
    subject to the caller's absolute metric and baseline-improvement gate.
    """

    if baseline_index not in candidates:
        raise ValueError("the certified baseline must be evaluated in every rung")
    if set(candidates) != set(scores):
        raise ValueError("scores must align exactly with the evaluated candidates")
    keep = min(max(1, int(keep)), len(candidates))

    legacy_ranked = sorted(
        candidates,
        key=lambda index: score_values(scores[index]),
        reverse=maximize,
    )
    paired = _paired_matrix(scores, candidates)
    utility = paired if maximize else -paired
    relative_active = paired.shape[1] >= _MIN_GROUP_RELATIVE_EVIDENCE_UNITS
    if relative_active:
        group_center = np.median(utility, axis=0)
        relative = utility - group_center
        relative_advantage = np.median(relative, axis=1)
        group_support = np.mean(relative >= 0.0, axis=1)
        position = {candidate: offset for offset, candidate in enumerate(candidates)}

        # Stable sorting preserves the legacy absolute ordering when relative
        # evidence is tied.
        ranked = sorted(
            legacy_ranked,
            key=lambda index: (
                float(relative_advantage[position[index]]),
                float(group_support[position[index]]),
            ),
            reverse=True,
        )
    else:
        ranked = legacy_ranked

    dominated: set[int] = set()
    if prune_dominated and relative_active:
        dominated = _dominated_candidates(utility, candidates, baseline_index)
    absolute_leader = legacy_ranked[0]
    dominated.discard(absolute_leader)
    eligible = [
        candidate for candidate in ranked if candidate not in dominated or candidate == absolute_leader
    ]
    promoted = [absolute_leader]
    promoted.extend(candidate for candidate in eligible if candidate != absolute_leader)
    promoted = promoted[:keep]
    if baseline_index not in promoted:
        promoted.append(baseline_index)

    block_wins = _block_winner_counts(utility) if relative_active else np.zeros(len(candidates), dtype=int)
    block_winner_candidates = int(np.count_nonzero(block_wins))

    primary_utility = np.asarray(
        [float(score_values(scores[index])[0]) * (1.0 if maximize else -1.0) for index in candidates]
    )
    primary_dispersion = float(np.ptp(primary_utility))
    numerical_floor = 32.0 * float(np.finfo(float).eps) * max(1.0, float(np.max(np.abs(primary_utility))))
    report = {
        "stage": "candidate_allocation",
        "method": ("paired_group_relative_median" if relative_active else "absolute_small_evidence_fallback"),
        "absolute_anchor": "certified_baseline",
        "evaluated_candidates": int(len(candidates)),
        "promoted_candidates": int(len(promoted)),
        "paired_evidence_units": int(paired.shape[1]),
        "minimum_relative_evidence_units": _MIN_GROUP_RELATIVE_EVIDENCE_UNITS,
        "group_relative_active": bool(relative_active),
        "consistently_dominated_candidates": int(len(dominated)),
        "block_winner_candidates": block_winner_candidates,
        "block_disagreement_detected": bool(block_winner_candidates > 1),
        "absolute_leader_retained": True,
        "baseline_retained": True,
        "primary_metric_dispersion": primary_dispersion,
        "zero_dispersion": bool(primary_dispersion <= numerical_floor),
        "variance_normalization": False,
    }
    return AllocationDecision(tuple(promoted), report)
