"""Focused regression tests for zero-knob proposer gates."""

import numpy as np

import tabpvn.base as base
from tabpvn import TabPVN


def test_categorical_evidence_is_a_gated_default_member():
    model = TabPVN(seed=0)

    assert model._categorical_evidence is True
    assert model._threshold_predicates is True
    assert model._native_categorical is False


def test_large_fit_budget_bounds_rows_verifier_and_saturated_classifier_rounds():
    classifier = {"rounds": 800, "lr": 0.05}
    resolved = base._large_fit_budget(
        classifier,
        base._BOOST_FIT_MAX_ROWS + 1,
        "classification",
        n_classes=2,
    )

    assert resolved is classifier
    assert resolved["fit_cap"] == 2_000_000
    assert resolved["holdout"] == 0.025
    assert resolved["rounds"] == 1_000

    regression = {"rounds": 2_000}
    base._large_fit_budget(regression, base._BOOST_FIT_MAX_ROWS + 1, "regression")
    assert regression["rounds"] == 2_000
    assert regression["fit_cap"] == 2_000_000
    assert regression["holdout"] == 0.025

    multiclass = {"rounds": 800}
    base._large_fit_budget(
        multiclass,
        base._BOOST_FIT_MAX_ROWS + 1,
        "classification",
        n_classes=4,
    )
    assert multiclass["rounds"] == 800

    small = {"rounds": 800}
    assert base._large_fit_budget(
        small,
        base._BOOST_FIT_MAX_ROWS,
        "classification",
        n_classes=2,
    ) == {"rounds": 800}


def test_class_weight_gate_selects_only_an_oof_auc_gain(monkeypatch):
    class RankStub:
        def __init__(self, class_weight=None, **_kwargs):
            self.class_weight = class_weight

        def fit(self, _X, _y):
            return self

        def _scores(self, X):
            # The balanced candidate ranks the positive class correctly; the
            # unweighted candidate reverses it. This isolates the gate metric.
            sign = 1.0 if self.class_weight == "balanced" else -1.0
            return np.column_stack([np.zeros(len(X)), sign * X[:, 0]])

    monkeypatch.setattr(base, "AdditiveCertifiedClassifier", RankStub)
    y = np.array([0] * 90 + [1] * 15)
    X = y[:, None].astype(float)

    assert TabPVN(seed=0)._auto_class_weight(X, y, {"rounds": 100}) == "balanced"


def test_class_weight_gate_rejects_threshold_only_changes(monkeypatch):
    class RankStub:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y):
            return self

        def _scores(self, X):
            return np.column_stack([np.zeros(len(X)), X[:, 0]])

    monkeypatch.setattr(base, "AdditiveCertifiedClassifier", RankStub)
    y = np.array([0] * 90 + [1] * 15)
    X = y[:, None].astype(float)

    assert TabPVN(seed=0)._auto_class_weight(X, y, {"rounds": 100}) is None


def test_classifier_tuning_reuses_stratified_rung_evidence(monkeypatch):
    validation_rows = []
    rank_calls = []

    class RankStub:
        def fit(self, _X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def _scores(self, X):
            validation_rows.append(X[:, 0].astype(int))
            return np.column_stack([np.zeros(len(X)), X[:, 1]])

    model = TabPVN(seed=7)
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: RankStub())
    monkeypatch.setattr(
        base,
        "_classification_rank_score",
        lambda yidx, probability: rank_calls.append((yidx, probability)) or 0.75,
    )

    def evaluate_two_candidates(candidates, base_idx, score_fn, rungs, maximize):
        assert maximize is True
        first = score_fn(candidates[0], rungs[0])()
        baseline = score_fn(candidates[base_idx], rungs[0])()
        return base_idx, {base_idx: baseline, 0: first}, [base_idx]

    monkeypatch.setattr(model, "_successive_halving", evaluate_two_candidates)
    y = np.array([0] * 480 + [1] * 120)
    X = np.column_stack([np.arange(len(y)), y]).astype(float)

    config = model._auto_tune_clf(X, y)

    assert config["depth"] == 6
    # Each candidate receives one aggregate score plus two paired verifier
    # blocks, all computed from the same validation predictions.
    assert len(rank_calls) == 6
    np.testing.assert_array_equal(rank_calls[0][0], rank_calls[3][0])
    np.testing.assert_array_equal(validation_rows[0], validation_rows[1])
    validation_y = y[validation_rows[0]]
    np.testing.assert_array_equal(np.unique(validation_y, return_counts=True)[1], [120, 30])


def test_classifier_tuning_reuses_final_rung_for_linear_leaf_gate(monkeypatch):
    fits = []

    class LinearGateStub:
        def __init__(self, linear_leaf=False):
            self.linear_leaf = linear_leaf

        def fit(self, _X, y, sample_weight=None):
            del sample_weight
            fits.append(self.linear_leaf)
            self.classes_ = np.unique(y)
            return self

        def _scores(self, X):
            scale = 2.0 if self.linear_leaf else 0.2
            return np.column_stack([np.zeros(len(X)), scale * (2.0 * X[:, 0] - 1.0)])

    model = TabPVN(seed=3)
    monkeypatch.setattr(
        model,
        "_classifier",
        lambda **kwargs: LinearGateStub(linear_leaf=kwargs.get("linear_leaf", False)),
    )

    def evaluate_baseline(candidates, base_idx, score_fn, rungs, maximize):
        assert maximize is True
        baseline = score_fn(candidates[base_idx], rungs[-1])()
        return base_idx, {base_idx: baseline}, [base_idx]

    monkeypatch.setattr(model, "_successive_halving", evaluate_baseline)
    y = np.tile([0, 1], 300)
    X = y[:, None].astype(float)

    config = model._auto_tune_clf(X, y)

    assert model._linear_leaf_tune_config == config
    assert model._linear_leaf_tune_decision is True
    assert fits.count(False) == 2
    assert fits.count(True) == 2
    assert model.linear_leaf_report_[-1]["selected"] is True
    assert model.linear_leaf_report_[-1]["loss_scenario_robust"] is True


def test_classifier_tuning_requires_rank_gain_without_accuracy_regression(monkeypatch):
    model = TabPVN(seed=0)
    X = np.zeros((600, 3))
    y = np.array([0] * 300 + [1] * 300)

    monkeypatch.setattr(
        model,
        "_successive_halving",
        lambda _c, base_idx, _score, _rungs, maximize: (
            0,
            {0: (0.71, 0.799), base_idx: (0.70, 0.80)},
            [0, base_idx],
        ),
    )
    assert model._auto_tune_clf(X, y)["depth"] == 4

    monkeypatch.setattr(
        model,
        "_successive_halving",
        lambda _c, base_idx, _score, _rungs, maximize: (
            0,
            {0: (0.71, 0.797), base_idx: (0.70, 0.80)},
            [0, base_idx],
        ),
    )
    assert model._auto_tune_clf(X, y)["depth"] == 6


def test_regression_tuning_gates_squared_loss_on_the_shared_final_rung(monkeypatch):
    class LossStub:
        def __init__(self, huber=0.95, **_kwargs):
            self.huber = huber

        def fit(self, _X, _y):
            return self

        def predict(self, X):
            return X[:, 0] + (0.0 if self.huber is None else 1.0)

    model = TabPVN(seed=5)
    model.mode = "regression"
    monkeypatch.setattr(base, "AdditiveCertifiedRegressor", LossStub)

    def evaluate_baseline(candidates, base_idx, score_fn, rungs, maximize):
        assert maximize is False
        baseline = score_fn(candidates[base_idx], rungs[-1])()
        return base_idx, {base_idx: baseline}, [base_idx]

    monkeypatch.setattr(model, "_successive_halving", evaluate_baseline)
    y = np.linspace(-2.0, 2.0, 600)
    X = np.column_stack([y, np.ones(len(y))])

    config = model._auto_tune(X, y)

    assert config["huber"] is None
    assert model.regression_loss_report_[-1]["selected"] is True
    assert model.regression_loss_report_[-1]["relative_rmse_reduction"] == 1.0


def test_wide_classifier_pool_is_selected_only_from_untouched_fold_gains(monkeypatch):
    class RankStub:
        def __init__(self, allowed=None):
            self.screened = allowed is not None

        def fit(self, _X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def _scores(self, X):
            sign = 1.0 if self.screened else -1.0
            return np.column_stack([np.zeros(len(X)), sign * X[:, 0]])

    model = TabPVN(seed=5)
    monkeypatch.setattr(
        model,
        "_classifier",
        lambda **kwargs: RankStub(allowed=kwargs.get("allowed")),
    )
    monkeypatch.setattr(
        model,
        "_wide_feature_pool",
        lambda _X, _y, limit=256: np.array([0], dtype=np.int64),
    )

    def evaluate_baseline(candidates, base_idx, score_fn, rungs, maximize):
        assert maximize is True
        baseline = score_fn(candidates[base_idx], rungs[-1])()
        return base_idx, {base_idx: baseline}, [base_idx]

    monkeypatch.setattr(model, "_successive_halving", evaluate_baseline)
    y = np.array([0] * 300 + [1] * 300)
    X = np.zeros((len(y), base._WIDE_SCREEN_MIN_FEATURES))
    X[:, 0] = y

    config = model._auto_tune_clf(X, y)

    assert config["allowed"] == (0,)
    assert model.feature_screen_report_[-1]["selected"] is True
    assert model.feature_screen_report_[-1]["auc_delta"] > 0.0


def test_rank_checkpoint_gate_reuses_one_tree_trace_per_fold(monkeypatch):
    fits = []

    class TraceStub:
        def fit(self, _X, y):
            fits.append(1)
            self.classes_ = np.unique(y)
            return self

        def _scores(self, X):
            return np.column_stack([np.zeros(len(X)), -X[:, 0]])

        def _scores_at_checkpoint(self, X, metric):
            assert metric == "auc"
            return np.column_stack([np.zeros(len(X)), X[:, 0]])

    model = TabPVN(seed=3)
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: TraceStub())
    y = np.array([0] * 300 + [1] * 100)
    X = np.zeros((len(y), base._WIDE_SCREEN_MIN_FEATURES))
    X[:, 0] = y

    selected = model._auto_rank_checkpoint_clf(
        X,
        y,
        {"rounds": 800, "lr": 0.05, "depth": 6, "leaf": 20},
    )

    assert selected is True
    assert len(fits) == 2
    assert model.rank_checkpoint_report_[-1]["shared_tree_trace"] is True


def test_shallow_boost_requires_twofold_auc_win(monkeypatch):
    class RankStub:
        def __init__(self, depth=None, **_kwargs):
            self.depth = depth
            self.classes_ = [0, 1]

        def fit(self, _X, _y):
            return self

        def _scores(self, X):
            sign = 1.0 if self.depth == 3 else -1.0
            return np.column_stack([np.zeros(len(X)), sign * X[:, 0]])

    monkeypatch.setattr(base, "AdditiveCertifiedClassifier", RankStub)
    y = np.array([0] * 300 + [1] * 100)
    X = y[:, None].astype(float)

    selected = TabPVN(seed=0)._auto_shallow_boost(
        X, y, {"rounds": 100, "lr": 0.1, "depth": 6, "leaf": 20, "patience": 10}
    )

    assert selected is not None
    assert selected["depth"] == 3
    assert selected["lr"] == 0.05


def test_joint_rank_regions_require_a_twofold_combination_win(monkeypatch):
    class RankStub:
        def __init__(self, linear_leaf=False, validation_metric="logloss", **_kwargs):
            self.challenger = linear_leaf and validation_metric == "auc"
            self.classes_ = [0, 1]

        def fit(self, _X, _y):
            return self

        def _scores(self, X):
            sign = 1.0 if self.challenger else -1.0
            return np.column_stack([np.zeros(len(X)), sign * X[:, 0]])

    monkeypatch.setattr(base, "AdditiveCertifiedClassifier", RankStub)
    model = TabPVN(seed=0)
    model._prep = type(
        "Prep",
        (),
        {
            "num_cols": [],
            "na_cols": [],
            "cat_cols": ["a", "b", "c", "d"],
            "onehot": {name: [0, 1] for name in ("a", "b", "c", "d")},
            "target_encoding": {},
        },
    )()
    y = np.array([0] * 360 + [1] * 40)
    X = np.zeros((len(y), 8))
    X[:, 0] = y

    selected = model._auto_joint_rank_regions(X, y, {"rounds": 100, "lr": 0.05, "depth": 6, "leaf": 20})

    assert selected is not None
    assert selected["depth"] == 4
    assert selected["linear_leaf"] is True
    assert selected["validation_metric"] == "auc"
    assert model.booster_selection_report_[-1]["selected"] is True


def test_smooth_gate_requires_material_binary_auc_gain(monkeypatch):
    class SmoothStub:
        def __init__(self, _X, _y, _classes, fw=None, k=15):
            pass

        def proba(self, X):
            return np.column_stack([1.0 - X[:, 0], X[:, 0]])

    monkeypatch.setattr(base, "_SmoothKNN", SmoothStub)
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": [0, 1]})()
    model._temp = 1.0
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    X = y[:, None].astype(float)
    splits = [(np.arange(4, 8), np.arange(0, 4)), (np.arange(0, 4), np.arange(4, 8))]
    neutral_scores = np.zeros((len(y), 2))
    no_gain_scores = np.column_stack([np.zeros(len(y)), X[:, 0]])

    neutral = {"scores": neutral_scores, "splits": splits}
    no_gain = {"scores": no_gain_scores, "splits": splits}

    assert model._smooth_gate(X, y, 0.4, np.ones(1), precomp=neutral) == 0.4
    assert model._smooth_gate(X, y, 0.4, np.ones(1), precomp=no_gain) == 0.0


def test_binary_smooth_gate_skips_fold_local_diffuse_metrics(monkeypatch):
    class UnexpectedSmooth:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("diffuse metrics must be rejected before neighbour construction")

    monkeypatch.setattr(base, "_SmoothKNN", UnexpectedSmooth)
    monkeypatch.setattr(
        base,
        "_booster_importance",
        lambda _model, width: np.ones(width, dtype=float),
    )
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": [0, 1]})()
    model._temp = 1.0
    y = np.tile(np.array([0, 1]), 6)
    X = np.zeros((len(y), 64), dtype=float)
    rows = np.arange(len(y))
    splits = [
        (np.arange(6, 12), np.arange(0, 6)),
        (np.arange(0, 6), np.arange(6, 12)),
    ]
    precomp = {
        "scores": np.zeros((len(y), 2)),
        "splits": splits,
        "models": [object(), object()],
        "evidence_rows": rows,
    }

    assert model._smooth_gate(X, y, 0.4, np.ones(X.shape[1]), precomp=precomp) == 0.0
    report = model.smooth_memory_report_[-1]
    assert report["selected"] is False
    assert report["reason"] == "diffuse_distance_metric"
    assert report["fold_effective_dimensions"] == [64.0, 64.0]


def test_multiclass_smooth_gate_requires_rank_gain_on_every_fold(monkeypatch):
    class SmoothStub:
        def __init__(self, _X, _y, _classes, fw=None, k=15):
            pass

        def proba(self, X):
            return X[:, :3]

    monkeypatch.setattr(base, "_SmoothKNN", SmoothStub)
    y = np.tile(np.array([0, 1, 2]), 12)
    X = np.eye(3)[y]
    rows = np.arange(len(y))
    splits = []
    for fold in range(3):
        valid = np.concatenate(
            [np.arange(block + 3 * fold, block + 3 * (fold + 1)) for block in (0, 9, 18, 27)]
        )
        splits.append((np.setdiff1d(rows, valid), valid))
    precomp = {
        "scores": np.zeros((len(y), 3)),
        "splits": splits,
        "evidence_rows": rows,
    }
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": [0, 1, 2]})()
    model._temp = 1.0

    assert model._smooth_gate(X, y, 0.4, np.ones(3), precomp=precomp) == 0.4
    assert model.smooth_memory_report_[-1]["selected"] is True
    assert min(model.smooth_memory_report_[-1]["fold_auc_delta"]) > 0.3

    losing = X.copy()
    losing[splits[1][1], :3] = np.roll(losing[splits[1][1], :3], 1, axis=1)

    assert model._smooth_gate(losing, y, 0.4, np.ones(3), precomp=precomp) == 0.0
    assert model.smooth_memory_report_[-1]["selected"] is False
    assert min(model.smooth_memory_report_[-1]["fold_auc_delta"]) < 0.0


def test_multiclass_smooth_gate_can_select_adaptive_sqrt_geometry(monkeypatch):
    class SmoothStub:
        def __init__(self, _X, _y, _classes, fw=None, k=15):
            self.k = k

        def proba(self, X):
            return X[:, :3] if self.k < 15 else np.full((len(X), 3), 1.0 / 3.0)

    monkeypatch.setattr(base, "_SmoothKNN", SmoothStub)
    y = np.tile(np.array([0, 1, 2]), 12)
    X = np.eye(3)[y]
    rows = np.arange(len(y))
    splits = []
    for fold in range(3):
        valid = np.concatenate(
            [np.arange(block + 3 * fold, block + 3 * (fold + 1)) for block in (0, 9, 18, 27)]
        )
        splits.append((np.setdiff1d(rows, valid), valid))
    precomp = {
        "scores": np.zeros((len(y), 3)),
        "splits": splits,
        "evidence_rows": rows,
    }
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": [0, 1, 2]})()
    model._temp = 1.0

    weight = model._smooth_gate(X, y, 0.4, np.ones(3), precomp=precomp)

    report = model.smooth_memory_report_[-1]
    assert weight == 0.6
    assert report["selected"] is True
    assert report["geometry"] == "adaptive_sqrt"
    assert report["neighbors"] == base._adaptive_smooth_neighbors(len(y))
    assert model._smooth_k == report["neighbors"]


def test_smooth_gate_uses_fold_local_booster_feature_weights(monkeypatch):
    observed_weights = []

    class SmoothStub:
        def __init__(self, _X, _y, _classes, fw=None, k=15):
            observed_weights.append(float(fw[0]))

        def proba(self, X):
            return np.full((len(X), 3), 1.0 / 3.0)

    monkeypatch.setattr(base, "_SmoothKNN", SmoothStub)
    monkeypatch.setattr(
        base,
        "_booster_importance",
        lambda model, width: np.full(width, model.marker, dtype=float),
    )
    y = np.tile(np.array([0, 1, 2]), 6)
    X = np.eye(3)[y]
    rows = np.arange(len(y))
    splits = []
    for fold in range(3):
        valid = np.concatenate([np.arange(block + 3 * fold, block + 3 * (fold + 1)) for block in (0, 9)])
        splits.append((np.setdiff1d(rows, valid), valid))
    fold_models = [type("Fold", (), {"marker": marker})() for marker in (1.0, 2.0, 3.0)]
    precomp = {
        "scores": np.zeros((len(y), 3)),
        "splits": splits,
        "models": fold_models,
        "evidence_rows": rows,
    }
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": [0, 1, 2]})()
    model._temp = 1.0

    model._smooth_gate(X, y, 0.4, np.full(3, 99.0), precomp=precomp)

    assert observed_weights == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_smooth_index_matches_dense_source_row_ties():
    X = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    member = base._SmoothKNN(X, np.array([0, 1, 1, 0]), [0, 1], k=2)
    query = (np.array([[0.0]]) - member.mu) / member.sd * member.fw

    indexed = member._neighbours(query)
    member._search_index = None
    dense = member._neighbours(query)

    np.testing.assert_array_equal(indexed, dense)


def test_auxiliary_probabilities_cannot_cross_the_certified_class_boundary():
    booster = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7]])
    auxiliary = np.array([[0.4, 0.6], [0.7, 0.3], [0.55, 0.45], [0.2, 0.8]])

    projected = base._preserve_certified_class(booster, auxiliary)

    assert np.array_equal(projected.argmax(1), booster.argmax(1))
    assert np.allclose(projected.sum(1), 1.0)

    multi_booster = np.array([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]])
    multi_auxiliary = np.array([[0.4, 0.3, 0.3], [0.2, 0.6, 0.2]])
    multi_projected = base._preserve_certified_class(multi_booster, multi_auxiliary)

    assert np.array_equal(multi_projected.argmax(1), multi_booster.argmax(1))
    assert np.allclose(multi_projected.sum(1), 1.0)


def test_dominant_multiclass_no_signal_gate_selects_class_preserving_prior():
    y = np.array([0] * 4 + [1] * 92 + [2] * 4)
    prior = np.bincount(y, minlength=3) / len(y)
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": [0, 1, 2]})()
    model._temp = 1.0
    model._fit_validation = None
    precomp = {
        "scores": np.tile(np.log(prior), (len(y), 1)),
        "evidence_rows": np.arange(len(y)),
    }

    probability = model._multiclass_no_signal_gate(y, precomp)

    assert probability is not None
    np.testing.assert_allclose(model._no_signal_prior, prior)
    np.testing.assert_allclose(probability, np.tile(prior, (len(y), 1)))
    assert model.multiclass_signal_report_[-1]["selected"] is True
    assert model.multiclass_signal_report_[-1]["reason"] == "oof_indistinguishable_from_prior"


def test_dominant_multiclass_no_signal_gate_rejects_predictive_ranking():
    y = np.array([0] * 4 + [1] * 92 + [2] * 4)
    probability = np.full((len(y), 3), 0.01)
    probability[np.arange(len(y)), y] = 0.98
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": [0, 1, 2]})()
    model._temp = 1.0
    model._fit_validation = None
    precomp = {
        "scores": np.log(probability),
        "evidence_rows": np.arange(len(y)),
    }

    selected = model._multiclass_no_signal_gate(y, precomp)

    assert selected is None
    assert model._no_signal_prior is None
    assert model.multiclass_signal_report_[-1]["selected"] is False
    assert model.multiclass_signal_report_[-1]["reason"] == "oof_predictive_signal_supported"


def test_dominant_multiclass_no_signal_prior_is_used_at_inference():
    scores = np.array([[0.0, 2.0, -1.0], [2.0, 0.0, -1.0]])
    model = TabPVN(seed=0)
    model._pred = type(
        "Pred",
        (),
        {"classes_": [0, 1, 2], "_scores": lambda _self, _X: scores},
    )()
    model._temp = 1.0
    model._no_signal_prior = np.array([0.05, 0.90, 0.05])

    probability = model._blended_proba(np.zeros((2, 1)))

    np.testing.assert_allclose(probability[0], model._no_signal_prior)
    np.testing.assert_array_equal(probability.argmax(1), scores.argmax(1))
    np.testing.assert_allclose(probability.sum(1), 1.0)


def test_shared_classifier_oof_replays_linear_leaves_and_class_weight(monkeypatch):
    calls = []

    class OOFStub:
        def __init__(self, linear_leaf=False, class_weight=None, validation_metric="logloss", **_kwargs):
            calls.append((linear_leaf, class_weight, validation_metric))
            self.classes_ = [0, 1]

        def fit(self, _X, _y):
            return self

        def _scores(self, X):
            return np.column_stack([np.zeros(len(X)), X[:, 0]])

    monkeypatch.setattr(base, "AdditiveCertifiedClassifier", OOFStub)
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": [0, 1]})()
    model._cfg = {
        "rounds": 100,
        "linear_leaf": True,
        "class_weight": "balanced",
        "validation_metric": "auc",
    }
    y = np.array([0, 0, 0, 1, 1, 1])

    model._clf_oof(y[:, None].astype(float), y)

    assert calls == [(True, "balanced", "auc")] * 3


def test_large_linear_leaf_gate_uses_a_deterministic_stratified_audit(monkeypatch):
    calls = []

    class GateStub:
        def __init__(self, linear_leaf=False, **_kwargs):
            self.linear_leaf = linear_leaf

        def fit(self, X, y):
            calls.append((self.linear_leaf, len(X), int(X[:, 0].sum())))
            self.classes_ = [0, 1]
            return self

        def _scores(self, X):
            return np.zeros((len(X), 2))

    monkeypatch.setattr(base, "AdditiveCertifiedClassifier", GateStub)
    n = 100_000
    X = np.arange(n, dtype=float)[:, None]
    y = np.arange(n) % 2
    boost = {"rounds": 100, "lr": 0.05, "depth": 6, "leaf": 20, "patience": 10}

    def run_gate():
        calls.clear()
        selected = TabPVN(seed=9)._auto_linear_leaf_clf(X, y, boost)
        return selected, list(calls)

    selected_a, first = run_gate()
    selected_b, second = run_gate()

    assert selected_a is False
    assert selected_b is False
    assert first == second
    assert len(first) == 2  # first fold rejects the affine candidate, then short-circuits
    assert {rows for _linear, rows, _sum in first} == {33_333}


def test_memory_blend_search_covers_every_low_authority_stratum_and_legacy_anchor():
    weights = np.asarray(base._STRATIFIED_MEMORY_BLEND_WEIGHTS)
    anchors = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5))
    midpoints = weights[~np.isin(weights, anchors)]

    np.testing.assert_allclose(np.sort(midpoints), np.arange(0.025, 0.5, 0.05))
    assert set(anchors).issubset(weights)


def test_target_encoding_accepts_weak_rank_gain_only_with_blockwise_proper_score_support():
    y = np.resize(np.array([0, 1]), 256)
    baseline = np.column_stack(
        (np.where(y == 0, 0.55, 0.45), np.where(y == 1, 0.55, 0.45))
    )
    candidate = np.column_stack(
        (np.where(y == 0, 0.65, 0.35), np.where(y == 1, 0.65, 0.35))
    )

    supported = base._target_encoding_classification_decision(
        y,
        np.array([0, 1]),
        baseline,
        candidate,
        0.75,
        0.752,
    )
    harmed = candidate.copy()
    first_block = base.verification_blocks(len(y), y)[0]
    harmed[first_block] = baseline[first_block, ::-1]
    rejected = base._target_encoding_classification_decision(
        y,
        np.array([0, 1]),
        baseline,
        harmed,
        0.75,
        0.752,
    )

    assert supported["selected"] is True
    assert supported["selection_path"] == "robust_proper_score_support"
    assert min(supported["block_log_loss_gain"]) > 0.0
    assert rejected["selected"] is False
    assert min(rejected["block_log_loss_gain"]) < 0.0
