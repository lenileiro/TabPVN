"""Focused tests for verifier-relative candidate budget allocation."""

import numpy as np

import tabpvn.base as base
from tabpvn import TabPVN
from tabpvn.candidate_allocation import (
    VerifierScore,
    allocate_candidates,
    verification_blocks,
)


def test_verification_blocks_are_complete_disjoint_and_stratified():
    labels = np.repeat(np.arange(3), 128)

    blocks = verification_blocks(len(labels), labels)

    assert len(blocks) == 4
    np.testing.assert_array_equal(np.sort(np.concatenate(blocks)), np.arange(len(labels)))
    for block in blocks:
        np.testing.assert_array_equal(np.unique(labels[block]), np.arange(3))


def test_group_relative_allocation_retains_absolute_leader_and_stable_challenger():
    scores = {
        0: VerifierScore((0.81, 0.80), (0.95, 0.60, 0.95, 0.60)),
        1: VerifierScore((0.75, 0.76), (0.75, 0.75, 0.75, 0.75)),
        2: VerifierScore((0.80, 0.80), (0.80, 0.80, 0.80, 0.80)),
        3: VerifierScore((0.79, 0.80), (0.79, 0.79, 0.79, 0.79)),
    }

    decision = allocate_candidates(
        scores,
        [0, 1, 2, 3],
        1,
        keep=2,
        maximize=True,
        prune_dominated=False,
    )

    assert decision.promoted == (0, 2, 1)
    assert decision.report["absolute_leader_retained"] is True
    assert decision.report["baseline_retained"] is True
    assert decision.report["variance_normalization"] is False


def test_consistent_dominance_reduces_only_later_rung_allocation():
    scores = {
        0: VerifierScore((0.90,), (0.90, 0.90, 0.90, 0.90)),
        1: VerifierScore((0.80,), (0.80, 0.80, 0.80, 0.80)),
        2: VerifierScore((0.79,), (0.70, 0.75, 0.70, 0.75)),
        3: VerifierScore((0.85,), (0.95, 0.75, 0.95, 0.75)),
        4: VerifierScore((0.78,), (0.76, 0.76, 0.76, 0.76)),
    }

    broad = allocate_candidates(
        scores,
        list(scores),
        1,
        keep=4,
        maximize=True,
        prune_dominated=False,
    )
    reduced = allocate_candidates(
        scores,
        list(scores),
        1,
        keep=4,
        maximize=True,
        prune_dominated=True,
    )

    assert len(broad.promoted) == 4
    assert reduced.promoted == (0, 3, 1)
    assert reduced.report["consistently_dominated_candidates"] == 2


def test_block_disagreement_is_reported_without_expanding_budget():
    scores = {
        0: VerifierScore((0.80,), (0.80, 0.80, 0.80, 0.80)),
        1: VerifierScore((0.79,), (0.79, 0.79, 0.79, 0.79)),
        2: VerifierScore((0.78,), (0.78, 0.78, 0.78, 0.78)),
        3: VerifierScore((0.50,), (1.00, 1.00, 0.00, 0.00)),
        4: VerifierScore((0.70,), (0.70, 0.70, 0.70, 0.70)),
    }

    decision = allocate_candidates(
        scores,
        list(scores),
        0,
        keep=3,
        maximize=True,
        prune_dominated=False,
    )

    assert decision.promoted == (0, 1, 2)
    assert decision.report["block_winner_candidates"] == 2
    assert decision.report["block_disagreement_detected"] is True


def test_zero_dispersion_falls_back_deterministically_without_normalization():
    scores = {index: VerifierScore((0.5,), (0.5, 0.5)) for index in range(4)}

    decision = allocate_candidates(
        scores,
        list(scores),
        3,
        keep=2,
        maximize=True,
        prune_dominated=True,
    )

    assert decision.promoted == (0, 1, 3)
    assert decision.report["zero_dispersion"] is True
    assert decision.report["consistently_dominated_candidates"] == 0
    assert decision.report["method"] == "absolute_small_evidence_fallback"
    assert decision.report["group_relative_active"] is False


def test_successive_halving_records_aggregate_allocation_evidence(monkeypatch):
    monkeypatch.setattr(base, "_pmap", lambda thunks: {key: thunk() for key, thunk in thunks.items()})
    model = TabPVN(seed=0)
    candidates = [
        {
            "scores": (
                VerifierScore((0.80,), (0.80, 0.81, 0.80, 0.81)),
                VerifierScore((0.82,), (0.82, 0.82, 0.82, 0.82)),
            )
        },
        {
            "scores": (
                VerifierScore((0.75,), (0.75, 0.75, 0.75, 0.75)),
                VerifierScore((0.76,), (0.76, 0.76, 0.76, 0.76)),
            )
        },
        {
            "scores": (
                VerifierScore((0.79,), (0.79, 0.79, 0.79, 0.79)),
                VerifierScore((0.78,), (0.78, 0.78, 0.78, 0.78)),
            )
        },
    ]
    rungs = [{"index": 0, "rounds": 10}, {"index": 1, "rounds": 20}]

    def score_fn(candidate, rung):
        return lambda: candidate["scores"][rung["index"]]

    best, _scores, finalists = model._successive_halving(
        candidates,
        1,
        score_fn,
        rungs,
        maximize=True,
    )

    assert best == 0
    assert 1 in finalists
    assert len(model.search_allocation_report_) == 1
    report = model.search_allocation_report_[0]
    assert report["method"] == "paired_group_relative_median"
    assert report["absolute_anchor"] == "certified_baseline"
    assert report["evaluated_candidates"] == 3
    assert report["promoted_candidates"] == 3
    assert report["paired_evidence_units"] == 4
    assert report["group_relative_active"] is True
    assert report["baseline_retained"] is True
    assert report["variance_normalization"] is False
    assert report["budget"] == {"index": 0, "rounds": 10}
