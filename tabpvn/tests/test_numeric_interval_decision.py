"""Focused contracts for the accuracy-only numeric interval decision head."""

import copy

import numpy as np

from tabpvn import TabPVN
from tabpvn.proposers import NumericIntervalPosteriorChallenger


def _interval_data(rows=900):
    x0 = np.linspace(-3.0, 3.0, rows)
    x1 = np.sin(np.arange(rows) * 0.37)
    x2 = np.cos(np.arange(rows) * 0.19)
    X = np.column_stack([x0, x1, x2])
    y = ((np.abs(x0) < 1.45) ^ (x1 > 0.35)).astype(int)
    return X, y


def _three_splits(n):
    fold = np.arange(n) % 3
    return [(np.flatnonzero(fold != index), np.flatnonzero(fold == index)) for index in range(3)]


def _parity_data(repeats=40):
    bits = np.asarray(
        [[(value >> shift) & 1 for shift in range(3)] for value in range(8)],
        dtype=float,
    )
    X = np.repeat(2.0 * bits - 1.0, repeats, axis=0)
    y = np.repeat((bits.sum(1) % 2).astype(int), repeats)
    return X, y


def _numeric_model(width):
    model = TabPVN(seed=0)
    model.mode = "classification"
    model._prep = None
    model.n_input_features_ = width
    model.feature_names_ = None
    model._pred = type("Pred", (), {"classes_": [0, 1]})()
    model._temp = 1.0
    model._smooth_oof_proba = None
    model._category_memory_oof_proba = None
    model._proof_path_oof_proba = None
    model._category_posterior_oof_proba = None
    return model


def test_numeric_interval_posterior_recovers_finite_rules_and_verifies_membership():
    X, y = _interval_data()
    challenger = NumericIntervalPosteriorChallenger(
        X,
        y,
        [0, 1],
        names=("position", "wave", "phase"),
        aggregation="strongest",
        smoothing="hierarchical",
    )
    base = np.tile([0.55, 0.45], (len(y), 1))
    combined = challenger.combine(base, X, weight=0.25)
    changed = np.flatnonzero(combined.argmax(1) != base.argmax(1))

    report = challenger.report()
    assert report["quantile_bins"] == 16
    assert all(len(feature["cut_points"]) == 15 for feature in report["selected_features"])
    assert len(changed) > 0
    evidence = challenger.evidence(X[changed[:1]], 0, base[changed[:1]], weight=0.25)
    assert evidence["kind"] == "numeric_interval_dirichlet_posterior"
    assert evidence["override"] is True
    assert evidence["verified"] is True
    assert len(evidence["family"]) == 2
    assert len(evidence["parent_factors"]) == 2
    assert TabPVN.verify_posterior_evidence(evidence)
    assert TabPVN.check_proof(evidence)
    assert all(condition["kind"] == "numeric_interval" for condition in evidence["conditions"])

    tampered = copy.deepcopy(evidence)
    tampered["conditions"][0]["column"] += 1
    assert not TabPVN.verify_posterior_evidence(tampered)

    tampered_parent = copy.deepcopy(evidence)
    tampered_parent["parent_factors"][0]["class_counts"][0] += 1
    assert not TabPVN.verify_posterior_evidence(tampered_parent)


def test_numeric_interval_triple_fallback_adds_only_an_untouched_decision():
    X, y = _parity_data()
    base = np.tile([0.55, 0.45], (len(y), 1))
    incumbent = NumericIntervalPosteriorChallenger(
        X,
        y,
        [0, 1],
        aggregation="strongest",
        smoothing="hierarchical",
    )
    fallback = NumericIntervalPosteriorChallenger(
        X,
        y,
        [0, 1],
        aggregation=NumericIntervalPosteriorChallenger.TRIPLE_FALLBACK,
        smoothing="hierarchical",
    )

    incumbent_probability = incumbent.combine(base, X, weight=1.0)
    fallback_probability = fallback.combine(base, X, weight=1.0)
    changed = np.flatnonzero(fallback_probability.argmax(1) != incumbent_probability.argmax(1))

    assert np.array_equal(incumbent_probability.argmax(1), base.argmax(1))
    assert len(changed) == int(np.sum(y == 1))
    assert (fallback_probability.argmax(1) == y).mean() == 1.0
    assert fallback.report()["triple_families"] == 1
    evidence = fallback.evidence(X[changed[:1]], 0, base[changed[:1]], weight=1.0)
    assert evidence["decision_mode"] == fallback.TRIPLE_FALLBACK
    assert len(evidence["family"]) == 3
    assert len(evidence["parent_factors"]) == 3
    assert evidence["fallback_incumbent"]["override"] is False
    assert evidence["verified"] is True
    assert TabPVN.check_proof(evidence)

    tampered = copy.deepcopy(evidence)
    tampered["fallback_incumbent"]["combined_probability"][0] += 0.1
    assert not TabPVN.verify_posterior_evidence(tampered)

    tampered_parent = copy.deepcopy(evidence)
    tampered_parent["parent_factors"][0]["class_counts"][0] += 1
    assert not TabPVN.verify_posterior_evidence(tampered_parent)


def test_numeric_interval_batched_modes_match_independent_paths_exactly():
    X, y = _interval_data(rows=600)
    challenger = NumericIntervalPosteriorChallenger(
        X,
        y,
        [0, 1],
        smoothing="hierarchical",
    )
    codes = challenger._interval_codes(X)
    expected_modes = challenger._delegate._posterior_modes_from_codes_scalar(codes)
    expected_triple = challenger._triple_posterior_codes_scalar(codes)
    actual_modes, actual_triple = challenger.posterior_modes_with_triple(X)

    assert actual_modes.keys() == expected_modes.keys()
    for aggregation in actual_modes:
        np.testing.assert_array_equal(actual_modes[aggregation], expected_modes[aggregation])
    np.testing.assert_array_equal(actual_triple, expected_triple)


def test_numeric_interval_gate_requires_transferable_decision_gain():
    X, y = _interval_data()
    model = _numeric_model(X.shape[1])
    base = np.tile([0.55, 0.45], (len(y), 1))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}

    weight = model._numeric_interval_gate(X, y, precomputed)

    report = model.numeric_interval_report_[-1]
    assert weight > 0
    assert report["selected"] is True
    assert report["permission"] == "decision_and_rank"
    assert report["smoothing"] == "hierarchical"
    assert report["probability_surface"] == "numeric_interval_rank"
    assert report["rank_auc_delta"] >= report["minimum_rank_gain"]
    assert min(report["fold_rank_auc_delta"]) >= report["minimum_fold_rank_gain"]
    assert model._numeric_interval_oof_proba is not None
    assert report["mean_score"] > model.numeric_interval_report_[0]["mean_score"]
    selected = next(
        candidate
        for candidate in report["candidates"]
        if candidate["accepted"]
        and candidate["weight"] == weight
        and candidate["aggregation"] == report["aggregation"]
    )
    assert all(net >= 0 for net in selected["fold_net_wins"])
    assert selected["paired_z"] >= report["minimum_paired_z"]


def test_numeric_interval_rank_failure_retains_decision_only_authority(monkeypatch):
    X, y = _interval_data()
    model = _numeric_model(X.shape[1])
    base = np.tile([0.55, 0.45], (len(y), 1))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}
    rank_scores = iter((0.50, 0.51, 0.50, 0.51, 0.50, 0.49, 0.50, 0.51))
    monkeypatch.setattr("tabpvn.base._classification_rank_score", lambda *_args: next(rank_scores))

    weight = model._numeric_interval_gate(X, y, precomputed)

    report = model.numeric_interval_report_[-1]
    assert weight > 0
    assert report["selected"] is True
    assert report["permission"] == "decision_only"
    assert report["probability_surface"] == "unchanged"
    assert min(report["fold_rank_auc_delta"]) < 0.0
    assert model._numeric_interval_oof_labels is not None
    assert model._numeric_interval_oof_proba is None


def test_numeric_interval_gate_can_select_the_bounded_triple_fallback():
    X, y = _parity_data()
    model = _numeric_model(X.shape[1])
    base = np.tile([0.55, 0.45], (len(y), 1))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}

    weight = model._numeric_interval_gate(X, y, precomputed)

    report = model.numeric_interval_report_[-1]
    assert weight > 0
    assert report["selected"] is True
    assert report["aggregation"] == NumericIntervalPosteriorChallenger.TRIPLE_FALLBACK
    assert np.array_equal(model._numeric_interval_oof_labels, y)


def test_numeric_interval_gate_rejects_when_probability_argmax_is_already_correct():
    X, y = _interval_data()
    model = _numeric_model(X.shape[1])
    base = np.where(y[:, None] == 0, np.array([0.8, 0.2]), np.array([0.2, 0.8]))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}

    weight = model._numeric_interval_gate(X, y, precomputed)

    assert weight == 0.0
    assert model._numeric_interval_oof_labels is None
    assert model.numeric_interval_report_[-1]["selected"] is False
    assert model.numeric_interval_report_[-1]["reason"] == "no_transferable_decision_gain"


def test_decision_only_interval_changes_predict_but_not_predict_proba_or_proof_contract():
    X, y = _interval_data()
    model = _numeric_model(X.shape[1])

    class Baseline:
        classes_ = [0, 1]

        @staticmethod
        def _scores(rows):
            return np.tile(np.log([0.55, 0.45]), (len(rows), 1))

        @staticmethod
        def predict(rows):
            return np.zeros(len(rows), dtype=int)

    model._pred = Baseline()
    model._smooth = None
    model._category_memory = None
    model._proof_path_memory = None
    model._category_posterior = None
    model._numeric_interval = NumericIntervalPosteriorChallenger(
        X,
        y,
        [0, 1],
        names=("position", "wave", "phase"),
        smoothing="hierarchical",
    )
    model._numeric_interval_w = 0.25
    model._numeric_interval_permission = "decision_only"
    model._numeric_interval_aggregation = "strongest"
    model._numeric_interval_smoothing = "hierarchical"

    expected_probability = model._blended_proba(X)
    probability = model.predict_proba(X)
    prediction = model.predict(X)

    assert np.array_equal(probability, expected_probability)
    assert (prediction == y).mean() > (probability.argmax(1) == y).mean() + 0.15
    row = int(np.flatnonzero(prediction != probability.argmax(1))[0])
    evidence = model.posterior_evidence(X, row)
    assert evidence["kind"] == "numeric_interval_dirichlet_posterior"
    assert evidence["override"] is True
    response = model.proof(X, row)
    artifact = model.proof_artifact(X, row)
    assert response["verification"]["status"] == "verified"
    assert all(
        set(condition) == {"feature", "operator", "value", "observed"}
        and condition["operator"] in {"lt", "gte", "between"}
        for condition in response["reasons"][0]["conditions"]
    )
    assert TabPVN.check_proof(response, artifact=artifact)


def test_rank_admitted_interval_is_public_but_bypassed_by_calibrated_probability():
    X, y = _interval_data()
    model = _numeric_model(X.shape[1])

    class Baseline:
        classes_ = [0, 1]

        @staticmethod
        def _scores(rows):
            return np.tile(np.log([0.55, 0.45]), (len(rows), 1))

        @staticmethod
        def predict(rows):
            return np.zeros(len(rows), dtype=int)

    model._pred = Baseline()
    model._smooth = None
    model._category_memory = None
    model._proof_path_memory = None
    model._category_posterior = None
    model._numeric_interval = NumericIntervalPosteriorChallenger(
        X,
        y,
        [0, 1],
        names=("position", "wave", "phase"),
        smoothing="hierarchical",
    )
    model._numeric_interval_w = 0.25
    model._numeric_interval_permission = "decision_and_rank"
    model._numeric_interval_aggregation = "strongest"
    model._numeric_interval_smoothing = "hierarchical"

    calibrated = np.tile([0.55, 0.45], (len(X), 1))
    expected_rank = model._numeric_interval.combine(calibrated, X, 0.25)

    np.testing.assert_allclose(model.predict_proba(X), expected_rank)
    np.testing.assert_allclose(model.predict_calibrated_proba(X), calibrated)
    assert np.array_equal(model.predict(X), np.array([0, 1])[expected_rank.argmax(1)])
    row = int(np.flatnonzero(expected_rank.argmax(1) != calibrated.argmax(1))[0])
    assert model.posterior_evidence(X, row)["verified"] is True
    response = model.proof(X, row)
    artifact = model.proof_artifact(X, row)
    assert response["verification"]["status"] == "verified"
    assert TabPVN.check_proof(response, artifact=artifact)
