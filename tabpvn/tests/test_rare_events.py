"""Focused coverage for deterministic rare-event sampling and calibration."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import tabpvn.certified_boost as boost_module
from tabpvn import TabPVN, base
from tabpvn.certified_boost import AdditiveCertifiedClassifier
from tabpvn.certified_confidence import CertifiedClassConfidence
from tabpvn.predicate_compiler import Predicate, SymbolicPredicateMap
from tabpvn.trees import (
    _binary_rank_metric,
    _fit_verifier_split,
    _multiclass_ovo_auc,
    _prior_preserving_subset_weight,
)


def test_binary_rank_metrics_match_sklearn_with_weights_and_ties():
    probability = np.array([0.1, 0.4, 0.4, 0.7, 0.7, 0.9])
    target = np.array([0, 0, 1, 0, 1, 1], dtype=bool)
    weight = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.75])
    margin = np.log((1.0 - probability) / probability)

    assert _binary_rank_metric(margin, target, weight, metric="auc") == pytest.approx(
        roc_auc_score(target, probability, sample_weight=weight)
    )
    assert _binary_rank_metric(margin, target, weight, metric="average_precision") == pytest.approx(
        average_precision_score(target, probability, sample_weight=weight)
    )


def test_binary_booster_exposes_multiple_checkpoints_from_one_tree_trace():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(600, 5))
    y = ((X[:, 0] > 1.1) & (X[:, 1] < 0.5)).astype(int)

    model = AdditiveCertifiedClassifier(
        rounds=50,
        lr=0.08,
        depth=3,
        leaf=6,
        patience=8,
        refit=False,
        rare_event=True,
        min_verifier_events=8,
        validation_metric="logloss",
        track_validation_metrics=("average_precision",),
        seed=5,
    ).fit(X, y)
    plain = AdditiveCertifiedClassifier(
        rounds=50,
        lr=0.08,
        depth=3,
        leaf=6,
        patience=8,
        refit=False,
        rare_event=True,
        min_verifier_events=8,
        validation_metric="logloss",
        seed=5,
    ).fit(X, y)

    trace = model._checkpoint_trace
    assert trace is not None
    assert set(trace["tree_counts"]) == {"logloss", "average_precision"}
    assert all(count % 2 == 0 for count in trace["tree_counts"].values())
    assert max(trace["tree_counts"].values()) == len(trace["trees"])
    np.testing.assert_allclose(
        model._scores_at_checkpoint(X[:20], "logloss"),
        model._scores(X[:20]),
    )
    np.testing.assert_allclose(model._scores(X), plain._scores(X))
    assert len(model.trees_) == len(plain.trees_)
    assert model._scores_at_checkpoint(X[:20], "average_precision").shape == (20, 2)


def test_multiclass_ovo_metric_matches_sklearn_without_weights():
    rng = np.random.default_rng(79)
    target = np.repeat(np.arange(4), 30)
    scores = rng.normal(size=(len(target), 4))
    probability = np.exp(scores - scores.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)

    assert _multiclass_ovo_auc(scores, target) == pytest.approx(
        roc_auc_score(target, probability, multi_class="ovo", average="macro")
    )


def test_multiclass_booster_exposes_rank_checkpoint_from_one_tree_trace():
    rng = np.random.default_rng(83)
    X = rng.normal(size=(720, 6))
    latent = np.column_stack(
        [
            1.5 * X[:, 0] - X[:, 1],
            1.2 * X[:, 2] + 0.8 * X[:, 3],
            -X[:, 0] - X[:, 2] + 0.7 * X[:, 4],
        ]
    )
    y = latent.argmax(1)

    model = AdditiveCertifiedClassifier(
        rounds=40,
        lr=0.08,
        depth=3,
        leaf=8,
        patience=7,
        refit=False,
        validation_metric="logloss",
        track_validation_metrics=("macro_ovo_auc",),
        seed=11,
    ).fit(X, y)

    trace = model._checkpoint_trace
    assert trace is not None
    assert set(trace["tree_counts"]) == {"logloss", "macro_ovo_auc"}
    assert all(count % 3 == 0 for count in trace["tree_counts"].values())
    assert max(trace["tree_counts"].values()) == len(trace["trees"])
    np.testing.assert_allclose(
        model._scores_at_checkpoint(X[:20], "logloss"),
        model._scores(X[:20]),
    )
    assert model._scores_at_checkpoint(X[:20], "macro_ovo_auc").shape == (20, 3)


def test_case_control_reservoir_recovers_the_source_prevalence_with_weights():
    y = np.array([0] * 99_000 + [1] * 1_000)

    rows, weights = boost_module._fit_sample(y, cap=10_000, seed=19, stratified=True, min_class_rows=1_000)
    rows_again, weights_again = boost_module._fit_sample(
        y, cap=10_000, seed=19, stratified=True, min_class_rows=1_000
    )

    np.testing.assert_array_equal(rows, rows_again)
    np.testing.assert_array_equal(weights, weights_again)
    np.testing.assert_array_equal(np.bincount(y[rows]), np.array([9_000, 1_000]))
    assert np.isclose(np.average(y[rows] == 1, weights=weights), 0.01)
    assert np.isclose(weights.mean(), 1.0)


def test_stratified_verifier_floor_keeps_source_prior_in_fit_and_verifier():
    y = np.array([0] * 9_000 + [1] * 1_000)
    source_weight = np.where(y == 0, 1.1, 0.1)

    fit, verifier = _fit_verifier_split(y, holdout=0.05, seed=7, stratified=True, min_verifier_events=500)
    fit_weight = _prior_preserving_subset_weight(y, source_weight, fit)
    verifier_weight = _prior_preserving_subset_weight(y, source_weight, verifier)
    string_weight = _prior_preserving_subset_weight(
        np.where(y == 1, "fraud", "legitimate"), source_weight, verifier
    )

    assert int((y[verifier] == 1).sum()) == 333
    assert np.isclose(np.average(y[fit] == 1, weights=fit_weight), 0.01)
    assert np.isclose(np.average(y[verifier] == 1, weights=verifier_weight), 0.01)
    np.testing.assert_array_equal(string_weight, verifier_weight)
    assert np.isclose(fit_weight.mean(), 1.0)
    assert np.isclose(verifier_weight.mean(), 1.0)


def test_rare_verifier_keeps_a_single_event_available_for_training():
    y = np.array([0] * 999 + [1])

    fit, verifier = _fit_verifier_split(y, holdout=0.1, seed=2, stratified=True, min_verifier_events=500)

    assert int(y[fit].sum()) == 1
    assert int(y[verifier].sum()) == 0


def test_rare_auto_tune_does_not_cross_validate_too_few_events():
    model = TabPVN(seed=2)
    model.rare_event_ = True

    config = model._auto_tune_clf(np.zeros((1_000, 3)), np.array([0] * 999 + [1]))

    assert config == {"rounds": 600, "lr": 0.05, "depth": 4, "leaf": 5, "patience": 20}


def test_rare_architecture_is_a_rank_gated_candidate_above_five_percent(monkeypatch):
    class RankStub:
        def __init__(self, rare_event=False):
            self.rare_event = rare_event

        def fit(self, _X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def _scores(self, X):
            sign = 1.0 if self.rare_event else -1.0
            return np.column_stack([np.zeros(len(X)), sign * X[:, 0]])

    model = TabPVN(seed=3)
    model.rare_class_ = 1
    model.rare_event_report_ = {"class": 1, "source_rate": 0.08}
    monkeypatch.setattr(
        model,
        "_classifier",
        lambda **kwargs: RankStub(rare_event=kwargs.get("rare_event", False)),
    )
    y = np.array([0] * 920 + [1] * 80)
    X = y[:, None].astype(float)

    assert model._auto_rare_architecture(X, y) is True
    assert model.rare_architecture_report_[-1]["selected"] is True
    assert model.rare_architecture_report_[-1]["auc_delta"] > 0.0


def test_rare_architecture_fails_closed_when_rank_evidence_ties(monkeypatch):
    class RankStub:
        def fit(self, _X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def _scores(self, X):
            return np.column_stack([np.zeros(len(X)), X[:, 0]])

    model = TabPVN(seed=3)
    model.rare_class_ = 1
    model.rare_event_report_ = {"class": 1, "source_rate": 0.03}
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: RankStub())
    y = np.array([0] * 970 + [1] * 30)
    X = y[:, None].astype(float)

    assert model._auto_rare_architecture(X, y) is False
    assert model.rare_architecture_report_[0]["selected"] is True
    assert model.rare_architecture_report_[-1]["selected"] is False


def test_rare_tuning_keeps_average_precision_as_an_auc_safety_guard(monkeypatch):
    model = TabPVN(seed=2)
    model.rare_event_ = True
    model.rare_class_ = 1
    X = np.zeros((1_000, 3))
    y = np.array([0] * 950 + [1] * 50)

    monkeypatch.setattr(
        model,
        "_successive_halving",
        lambda _c, base_idx, _score, _rungs, maximize: (
            0,
            {0: (0.76, 0.197, 0.95), base_idx: (0.75, 0.20, 0.95)},
            [0, base_idx],
        ),
    )

    assert model._auto_tune_clf(X, y)["depth"] == 6


def test_classifier_exposes_prior_corrected_rare_verifier_weights(monkeypatch):
    y = np.array([0] * 99_000 + [1] * 1_000)
    X = np.arange(len(y), dtype=float)[:, None]
    captured = {}

    def stub_softmax(X_fit, y_fit, **kwargs):
        captured["y"] = y_fit.copy()
        captured["weight"] = kwargs["sample_weight"].copy()
        _fit, verifier = _fit_verifier_split(
            y_fit,
            kwargs["holdout"],
            kwargs["seed"],
            stratified=kwargs["stratified_holdout"],
            min_verifier_events=kwargs["min_verifier_events"],
        )
        return np.zeros(2), 0.1, [], [0, 1], [], verifier, False

    monkeypatch.setattr(boost_module, "reason_boost_softmax", stub_softmax)
    model = AdditiveCertifiedClassifier(
        seed=3,
        fit_cap=10_000,
        holdout=0.05,
        refit=False,
        rare_event=True,
        rare_min_events=1_000,
        min_verifier_events=500,
    ).fit(X, y)

    assert np.isclose(np.average(captured["y"] == 1, weights=captured["weight"]), 0.01)
    assert np.isclose(np.average(y[model.ver_] == 1, weights=model.ver_weight_), 0.01)


def test_rare_classifier_accepts_string_target_labels():
    y = np.array(["legitimate"] * 990 + ["fraud"] * 10)
    X = np.arange(len(y), dtype=float)[:, None]

    model = AdditiveCertifiedClassifier(
        rounds=0,
        seed=6,
        fit_cap=500,
        holdout=0.2,
        refit=False,
        rare_event=True,
        rare_min_events=50,
        min_verifier_events=20,
    ).fit(X, y)

    assert np.isclose(
        np.average(y[model.ver_] == "fraud", weights=model.ver_weight_),
        0.01,
    )


def test_weighted_rare_threshold_improves_f1_at_source_prevalence():
    y = np.array([0] * 9_000 + [1] * 1_000)
    rare_probability = np.r_[
        np.full(900, 0.6),
        np.full(8_100, 0.05),
        np.full(500, 0.8),
        np.full(500, 0.4),
    ]
    proba = np.column_stack([1.0 - rare_probability, rare_probability])
    weights = np.where(y == 0, 1.1, 0.1)

    _balanced, threshold, report = base._fit_binary_thresholds(
        proba, y, classes=[0, 1], rare_class=1, sample_weight=weights
    )

    assert threshold >= 0.8
    assert report["weighted_f1"] > report["default_f1"]
    assert report["weighted_precision"] == 1.0
    assert report["weighted_recall"] == 0.5


def test_temporal_rare_threshold_is_selected_on_expanding_future_windows():
    groups = np.repeat(np.arange(90, dtype=np.int64), 8)
    y = np.zeros(len(groups), dtype=np.int64)
    y[np.arange(0, len(y), 8)] = 1
    rare_probability = np.where(y == 1, 0.4, 0.2)
    proba = np.column_stack([1.0 - rare_probability, rare_probability])

    balanced, threshold, report = base._fit_binary_thresholds(
        proba,
        y,
        classes=[0, 1],
        rare_class=1,
        validation_groups=groups,
    )

    assert np.isclose(balanced, 0.4)
    assert np.isclose(threshold, 0.4)
    assert report["validation_mode"] == "prequential_future"
    assert report["validation_folds"] == 3
    assert 0 < report["evaluation_rows"] < len(y)
    assert report["cross_fitted"] is False


def test_temporal_rare_threshold_fails_closed_without_a_time_boundary():
    y = np.array([0] * 180 + [1] * 20)
    rare_probability = np.where(y == 1, 0.4, 0.2)
    proba = np.column_stack([1.0 - rare_probability, rare_probability])

    balanced, threshold, report = base._fit_binary_thresholds(
        proba,
        y,
        classes=[0, 1],
        rare_class=1,
        validation_groups=np.zeros(len(y), dtype=np.int64),
    )

    assert balanced is None
    assert threshold is None
    assert report["validation_mode"] == "insufficient_prequential_evidence"
    assert report["evaluation_rows"] == 0


def test_prequential_threshold_rejects_a_stale_operating_point_under_drift():
    groups = np.repeat(np.arange(100, dtype=np.int64), 10)
    y = np.zeros(len(groups), dtype=np.int64)
    y[::10] = 1
    rare_probability = np.full(len(y), 0.2)
    for group in range(100):
        start = group * 10
        rare_probability[start] = 0.4 if group < 40 else 0.6
        if group >= 40 and group % 3 == 0:
            rare_probability[start + 1] = 0.45
    proba = np.column_stack([1.0 - rare_probability, rare_probability])

    _balanced, exchangeable_threshold, _report = base._fit_binary_thresholds(
        proba,
        y,
        classes=[0, 1],
        rare_class=1,
    )
    _balanced, temporal_threshold, report = base._fit_binary_thresholds(
        proba,
        y,
        classes=[0, 1],
        rare_class=1,
        validation_groups=groups,
    )

    future_y = np.tile(np.array([1] + [0] * 9), 60)
    future_probability = np.full(len(future_y), 0.2)
    future_probability[::10] = 0.6
    future_probability[np.arange(1, len(future_y), 30)] = 0.45
    exchangeable_f1 = f1_score(future_y, future_probability >= exchangeable_threshold)
    temporal_f1 = f1_score(future_y, future_probability >= 0.5)

    assert np.isclose(exchangeable_threshold, 0.4)
    assert temporal_threshold is None
    assert report["validation_mode"] == "prequential_future"
    assert temporal_f1 == 1.0
    assert temporal_f1 > exchangeable_f1 + 0.1


def test_importance_weighted_confidence_uses_effective_sample_size():
    correct = np.r_[np.ones(90), np.zeros(10)]
    uniform_bound, uniform_n = CertifiedClassConfidence._weighted_lb(correct, np.ones(100), z=1.64)
    uneven_bound, uneven_n = CertifiedClassConfidence._weighted_lb(
        correct, np.r_[np.full(99, 0.1), 90.1], z=1.64
    )

    assert np.isclose(uniform_n, 100.0)
    assert uneven_n < 2.0
    assert uneven_bound < uniform_bound


def test_zero_knob_fit_detects_rare_events_and_builds_an_operating_point(monkeypatch):
    rng = np.random.default_rng(81)
    X = rng.normal(size=(1_200, 8))
    risk = 1.8 * X[:, 0] - 1.2 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3]
    y = np.zeros(len(X), dtype=int)
    y[np.argsort(risk)[-24:]] = 1

    monkeypatch.setattr(
        TabPVN,
        "_auto_tune",
        lambda self, _X, _y: {
            "rounds": 30,
            "lr": 0.1,
            "depth": 3,
            "leaf": 5,
            "patience": 5,
            "holdout": 0.2,
        },
    )
    monkeypatch.setattr(base, "_smooth_weight", lambda _n: 0.0)
    monkeypatch.setattr(TabPVN, "_auto_rare_architecture", lambda _self, _X, _y: True)

    model = TabPVN(seed=4).fit(X, y)
    prediction = model.predict_rare(X[:20])

    assert model.rare_event_ is True
    assert model.rare_class_ == 1
    assert model.boost_["rare_event"] is True
    assert model._conf is not None
    assert model.rare_event_report_["reservoir_events"] == 24
    assert model.rare_event_report_["weighted_reservoir_rate"] == 0.02
    assert model.rare_event_report_["calibration_source"] == "out_of_fold"
    assert model.rare_event_report_["weighted_calibration_rate"] == 0.02
    assert prediction.shape == (20,)


def test_rare_architecture_gate_selects_replayable_tail_rules(monkeypatch):
    rng = np.random.default_rng(93)
    X = rng.integers(0, 100, size=(10_000, 5)).astype(float)
    y = ((X[:, 0] >= 95) & (X[:, 1] <= 9)).astype(int)
    fits = []

    class RuleAwareClassifier:
        def fit(self, features, labels, sample_weight=None):
            fits.append(features.shape[1])
            self.n_features_ = features.shape[1]
            self.classes_ = np.array([0, 1])
            return self

        def _scores(self, features):
            signal = np.zeros(len(features)) if self.n_features_ == X.shape[1] else features[:, X.shape[1]]
            return np.column_stack([np.zeros(len(features)), 8.0 * signal])

    model = TabPVN(seed=7)
    model.rare_event_ = True
    model.rare_class_ = 1
    model.rare_event_report_ = {"active": True, "class": 1, "source_rate": float(y.mean())}
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: RuleAwareClassifier())
    boost = {"rounds": 20, "lr": 0.1, "depth": 3, "leaf": 5}

    mapper = model._auto_rare_interactions(X, y, boost)

    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert mapper is not None
    assert any(predicate.kind.startswith("threshold_") for predicate in mapper.predicates)
    assert boost["validation_metric"] == "average_precision"
    assert report["rare_rank_checkpoint"]["selected"] is False
    assert report["rare_rank_checkpoint"]["deployed"] is True
    assert report["rare_symbolic_predicate_boost"]["selected"] is True
    assert min(report["rare_symbolic_predicate_boost"]["fold_ap_delta"]) >= 0.002
    assert len(fits) == 4


def test_rare_architecture_gate_fails_closed_on_equal_ap(monkeypatch):
    rng = np.random.default_rng(96)
    X = rng.integers(0, 100, size=(6_000, 5)).astype(float)
    y = ((X[:, 0] >= 95) & (X[:, 1] <= 9)).astype(int)

    class FeatureBlindClassifier:
        def fit(self, features, labels, sample_weight=None):
            self.classes_ = np.array([0, 1])
            return self

        def _scores(self, features):
            signal = features[:, 2] / 100.0
            return np.column_stack([np.zeros(len(features)), signal])

    model = TabPVN(seed=8)
    model.rare_event_ = True
    model.rare_class_ = 1
    model.rare_event_report_ = {"active": True, "class": 1, "source_rate": float(y.mean())}
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: FeatureBlindClassifier())
    boost = {"rounds": 20, "lr": 0.1, "depth": 3, "leaf": 5}

    mapper = model._auto_rare_interactions(X, y, boost)

    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert mapper is None
    assert "validation_metric" not in boost
    assert report["certified_boost"]["selected"] is True
    assert report["rare_symbolic_predicate_boost"]["selected"] is False


def test_rare_rule_failure_preserves_a_selected_rank_checkpoint(monkeypatch):
    rng = np.random.default_rng(97)
    X = rng.integers(0, 100, size=(6_000, 5)).astype(float)
    y = ((X[:, 0] >= 95) & (X[:, 1] <= 9)).astype(int)

    class RankCheckpointClassifier:
        def fit(self, features, labels, sample_weight=None):
            if features.shape[1] > X.shape[1]:
                raise RuntimeError("rule candidate failed")
            self.classes_ = np.array([0, 1])
            return self

        def _scores(self, features):
            signal = features[:, 2] / 100.0
            return np.column_stack([np.zeros(len(features)), signal])

        def _scores_at_checkpoint(self, features, metric):
            if metric != "average_precision":
                return self._scores(features)
            signal = ((features[:, 0] >= 95) & (features[:, 1] <= 9)).astype(float)
            return np.column_stack([np.zeros(len(features)), 8.0 * signal])

    model = TabPVN(seed=9)
    model.rare_event_ = True
    model.rare_class_ = 1
    model.rare_event_report_ = {"active": True, "class": 1, "source_rate": float(y.mean())}
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: RankCheckpointClassifier())
    boost = {"rounds": 20, "lr": 0.1, "depth": 3, "leaf": 5}

    mapper = model._auto_rare_interactions(X, y, boost)

    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert mapper is None
    assert boost["validation_metric"] == "average_precision"
    assert report["rare_rank_checkpoint"]["selected"] is True
    assert report["rare_rank_checkpoint"]["deployed"] is True
    assert report["rare_symbolic_predicate_boost"]["selected"] is False


def test_selected_rare_interval_replays_through_prediction_and_certificate(monkeypatch):
    X = np.tile(np.arange(20, dtype=float), 30)[:, None]
    y = (X[:, 0] == 9).astype(int)
    mapper = SymbolicPredicateMap(seed=0, rare_rules=True, rare_class=1)
    mapper.predicates = [
        Predicate(
            "threshold_interval",
            (0, 0),
            1,
            thresholds=(8.5, 9.5),
            directions=(False, True),
        )
    ]
    monkeypatch.setattr(
        TabPVN,
        "_auto_tune",
        lambda _self, _X, _y: {
            "rounds": 30,
            "lr": 0.1,
            "depth": 2,
            "leaf": 4,
            "patience": 8,
        },
    )
    monkeypatch.setattr(
        TabPVN,
        "_auto_rare_interactions",
        lambda _self, _X, _y, _boost: mapper,
    )
    monkeypatch.setattr(TabPVN, "_auto_rare_architecture", lambda _self, _X, _y: True)

    model = TabPVN(seed=0).fit(X, y)

    assert model.rare_event_ is True
    assert model.interaction_features_ == ["feature[0] > 8.5 AND feature[0] <= 9.5"]
    assert model._X(X[:3]).shape[1] == 2
    assert np.array_equal(model.predict(X), y)
    assert model.certify(X[:20]) == 1.0


def test_independent_deploy_holdout_calibration_consumes_prior_corrected_weights():
    rng = np.random.default_rng(53)
    X = rng.normal(size=(1_000, 4))
    y = np.array([0] * 800 + [1] * 200)

    class VerifierStub:
        classes_ = [0, 1]
        ver_ = np.arange(len(y))
        ver_weight_ = np.where(y == 0, 0.99, 0.04)
        verifier_evidence_role_ = "independent_calibration"

        def _scores(self, rows):
            return np.column_stack([-rows[:, 0], rows[:, 0]])

        def predict(self, rows):
            return np.asarray(self.classes_)[self._scores(rows).argmax(1)]

    model = TabPVN(seed=8)
    model.mode = "classification"
    model._pred = VerifierStub()
    model.rare_event_ = True
    model.rare_class_ = 1
    model.rare_event_report_ = {"active": True, "class": 1, "source_rate": 0.01}
    model._build_confidence(X, y)

    assert model._conf is not None
    assert model._conf.weighted_ is True
    assert model.rare_event_report_["calibration_source"] == "independent_deploy_holdout"
    assert np.isclose(model.rare_event_report_["weighted_calibration_rate"], 0.01)
