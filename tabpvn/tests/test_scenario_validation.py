"""Focused tests for deterministic correlated scenario verification."""

import json

import numpy as np

from tabpvn.scenario_validation import evaluate_candidate_scenarios, latin_hypercube


def test_latin_hypercube_covers_each_marginal_stratum_deterministically():
    first = latin_hypercube(32, 4, seed=7)
    second = latin_hypercube(32, 4, seed=7)

    np.testing.assert_array_equal(first, second)
    for column in range(first.shape[1]):
        strata = np.floor(first[:, column] * len(first)).astype(int)
        np.testing.assert_array_equal(np.sort(strata), np.arange(len(first)))


def test_scenario_verifier_prefers_stable_lower_tail_and_reports_success():
    utility = np.asarray(
        [
            [0.99, 0.51, 0.99, 0.51],
            [0.75, 0.75, 0.75, 0.75],
            [0.80, 0.80, 0.80, 0.80],
        ]
    )

    verification = evaluate_candidate_scenarios(utility, baseline_position=1, seed=11)

    np.testing.assert_allclose(verification.weights.sum(axis=1), 1.0)
    assert np.all(verification.weights >= 0.0)
    assert np.linalg.eigvalsh(verification.correlation).min() > 0.0
    assert verification.lower_relative_utility[2] > verification.lower_relative_utility[0]
    assert verification.summaries[2].baseline_success_rate == 1.0
    assert verification.summaries[0].baseline_success_rate < 1.0

    report = verification.report([10, 11, 12], baseline_position=1)
    assert report["method"] == "correlated_latin_hypercube"
    assert report["baseline_candidate"] == 11
    assert report["scenario_count"] == 69
    assert report["candidate_summaries"][2]["baseline_success_rate"] == 1.0
    json.dumps(report)


def test_single_evidence_block_falls_back_to_its_exact_anchor():
    verification = evaluate_candidate_scenarios([[0.7], [0.8]], baseline_position=0)

    np.testing.assert_array_equal(verification.weights, np.ones((1, 1)))
    assert verification.summaries[1].baseline_success_rate == 1.0
    report = verification.report([0, 1], baseline_position=0)
    assert report["stratified_scenarios"] == 0
    assert report["anchor_scenarios"] == 1
