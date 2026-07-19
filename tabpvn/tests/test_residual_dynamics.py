"""Residual-stream diagnostics and phase-controller coverage."""

import numpy as np
import pytest

from tabpvn.certified_boost import AdditiveCertifiedClassifier
from tabpvn.residual_dynamics import ResidualDynamicsTracker, hardest_class_pair, summarize_dynamics


def _confused_pair_state(rows_per_class=40):
    target = np.repeat(np.arange(3), rows_per_class)
    scores = np.zeros((len(target), 3), dtype=float)
    scores[target == 0, 0] = 3.0
    return scores, target


def test_tracker_activates_after_an_initial_consensus_pair_stalls():
    scores, target = _confused_pair_state()
    tracker = ResidualDynamicsTracker(scores, target)
    repeated_update = np.tile(np.array([0.08, -0.04, -0.04]), (len(target), 1))

    scores = scores + repeated_update
    assert tracker.observe(scores, repeated_update, ()) == (1, 2)
    assert tracker.records[-1]["activation_reason"] == "initial_pair_stall"


def test_tracker_activates_when_stable_updates_have_low_innovation():
    scores, target = _confused_pair_state()
    tracker = ResidualDynamicsTracker(scores, target)
    repeated_update = np.zeros_like(scores)
    repeated_update[target == 1, 1] = 0.08
    repeated_update[target == 1, 2] = -0.08
    repeated_update[target == 2, 1] = -0.08
    repeated_update[target == 2, 2] = 0.08

    scores = scores + repeated_update
    assert tracker.observe(scores, repeated_update, ()) == ()
    scores = scores + repeated_update
    assert tracker.observe(scores, repeated_update, ()) == (1, 2)

    first, second = tracker.records
    assert first["next_phase"] == "depth_wise"
    assert "pair_loss_improving" in first["blocked_by"]
    assert second["next_phase"] == "hard_pair"
    assert second["activation_reason"] == "low_innovation"
    assert second["pair_stable"] is True
    assert second["stalled_halves"] == (False, False)
    assert second["persistent_halves"] == (True, True)
    assert second["effective_rank"] == pytest.approx(1.0)
    assert second["innovation_ratio"] == pytest.approx(0.0, abs=1e-7)


def test_tracker_keeps_an_admitted_pair_until_consensus_changes():
    scores, target = _confused_pair_state()
    tracker = ResidualDynamicsTracker(scores, target)
    repeated_update = np.tile(np.array([0.08, -0.04, -0.04]), (len(target), 1))
    for _round in range(2):
        scores = scores + repeated_update
        tracker.observe(scores, repeated_update, ())
    assert tracker.next_pair == (1, 2)

    corrective_update = np.zeros_like(scores)
    corrective_update[target == 1, 1] = 0.8
    corrective_update[target == 1, 2] = -0.8
    corrective_update[target == 2, 1] = -0.8
    corrective_update[target == 2, 2] = 0.8
    scores = scores + corrective_update

    assert tracker.observe(scores, corrective_update, (1, 2)) == ()
    assert tracker.records[-1]["hard_pair"] != (1, 2)


def test_hardest_pair_and_summary_are_deterministic():
    scores, target = _confused_pair_state(12)
    assert hardest_class_pair(scores, target) == (1, 2)

    tracker = ResidualDynamicsTracker(scores, target)
    update = np.tile(np.array([0.05, -0.025, -0.025]), (len(target), 1))
    for _round in range(3):
        scores = scores + update
        tracker.observe(scores, update, tracker.next_pair)
    summary = summarize_dynamics(tracker.records)

    assert summary["rounds"] == 3
    assert summary["hard_pair_rounds"] == 2
    assert summary["activation_rate"] == pytest.approx(2.0 / 3.0)
    assert np.isfinite(summary["final_validation_logloss"])


def test_lightweight_controller_matches_detailed_decisions_without_records():
    scores, target = _confused_pair_state()
    detailed = ResidualDynamicsTracker(scores, target, detailed=True)
    lightweight = ResidualDynamicsTracker(scores, target, detailed=False)
    update = np.tile(np.array([0.05, -0.025, -0.025]), (len(target), 1))

    for _round in range(4):
        scores = scores + update
        expected = detailed.observe(scores, update, detailed.next_pair)
        assert lightweight.observe(scores, update, lightweight.next_pair) == expected

    assert detailed.records
    assert lightweight.records == []


def test_adaptive_booster_exposes_audit_without_changing_proof_trees():
    rng = np.random.default_rng(941)
    X = rng.normal(size=(540, 7))
    latent = np.column_stack(
        [
            X[:, 0] - 0.4 * X[:, 1],
            X[:, 1] + X[:, 2] * X[:, 3],
            -X[:, 0] - X[:, 2] * X[:, 3],
        ]
    )
    y = latent.argmax(axis=1)
    model = AdditiveCertifiedClassifier(
        rounds=16,
        lr=0.05,
        depth=3,
        leaf=10,
        patience=6,
        refit=True,
        max_leaves=8,
        best_first_pair=True,
        adaptive_best_first_pair=True,
        track_residual_dynamics=True,
        seed=19,
    ).fit(X, y)

    assert model.residual_dynamics_
    assert len(model.pair_growth_schedule_) == len(model.residual_dynamics_)
    assert len(model.trees_) % 3 == 0
    assert model.kernel_certify(X, n_trees=18, sample=30)["scores_reproduced"] == 1.0


def test_adaptive_growth_requires_the_hard_pair_architecture():
    X = np.zeros((30, 2))
    y = np.repeat(np.arange(3), 10)

    with pytest.raises(ValueError, match="requires best_first_pair"):
        AdditiveCertifiedClassifier(
            rounds=2,
            refit=False,
            max_leaves=8,
            adaptive_best_first_pair=True,
        ).fit(X, y)
