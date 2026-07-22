"""Focused coverage for the coupled multiclass zero-knob architecture."""

import numpy as np

from tabpvn import TabPVN
from tabpvn.certified_boost import AdditiveCertifiedClassifier


def _three_class_table(seed=131, rows=1_200):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 100, size=(rows, 6)).astype(float)
    first = (X[:, 0] >= 70) & (X[:, 1] <= 35)
    second = ~first & (X[:, 2] >= 65) & (X[:, 3] <= 40)
    return X, np.where(first, 0, np.where(second, 1, 2))


def test_automatic_multiclass_fit_stratifies_the_deployed_verifier(monkeypatch):
    X, y = _three_class_table(rows=450)
    model = TabPVN(seed=3)
    monkeypatch.setattr(
        model,
        "_auto_tune",
        lambda _X, _y: {
            "rounds": 4,
            "lr": 0.05,
            "depth": 2,
            "leaf": 5,
            "patience": 2,
        },
    )

    def no_interactions(_X, _y, _boost):
        model.multiclass_architecture_report_ = {}
        return None

    monkeypatch.setattr(model, "_auto_multiclass_interactions", no_interactions)

    model.fit(X, y)

    assert model.boost_["stratified_holdout"] is True
    assert model._pred.stratified_holdout is True
    assert model.multiclass_architecture_report_["stratified_verifier"] is True


def test_encoded_categorical_multiclass_fit_enables_numeric_pair_growth(monkeypatch):
    import pandas as pd

    X, y = _three_class_table(rows=450)
    frame = pd.DataFrame(X, columns=[f"feature_{index}" for index in range(X.shape[1])])
    frame["category"] = pd.Categorical(np.where(X[:, 0] > 50, "high", "low"))
    model = TabPVN(seed=3)

    def selected_config(features, labels):
        return model._with_adaptive_multiclass_pair_growth(
            features,
            labels,
            {
                "rounds": 4,
                "lr": 0.05,
                "depth": 2,
                "leaf": 5,
                "patience": 2,
            },
        )

    monkeypatch.setattr(
        model,
        "_auto_tune",
        selected_config,
    )
    monkeypatch.setattr(model, "_auto_target_encoding", lambda _X, _y: False)

    def no_interactions(_X, _y, _boost):
        model.multiclass_architecture_report_ = {}
        return None

    monkeypatch.setattr(model, "_auto_multiclass_interactions", no_interactions)

    model.fit(frame, y)

    assert "stratified_holdout" not in model.boost_
    assert model._pred.stratified_holdout is False
    assert model.multiclass_architecture_report_["stratified_verifier"] is False
    pair_report = next(
        row for row in model.booster_selection_report_ if row["name"] == "adaptive_multiclass_pair_growth"
    )
    assert model.boost_["adaptive_best_first_pair"] is True
    assert model.boost_["verifier_gated_pair_growth"] is True
    assert "coupled_pair_growth" not in model.boost_
    assert pair_report["controller_enabled"] is True
    assert pair_report["verifier_gate_enabled"] is True
    assert pair_report["reason"] in {
        "verified_pair_expert",
        "pair_expert_rejected",
        "no_verified_residual_pair",
    }


def test_adaptive_pair_growth_is_bounded_to_supported_multiclass_schemas():
    X, y = _three_class_table(rows=450)
    model = TabPVN(seed=3)
    base = {"rounds": 20, "depth": 4}

    selected = model._with_adaptive_multiclass_pair_growth(X, y, base)

    assert selected["max_leaves"] == 24
    assert selected["best_first_pair"] is True
    assert selected["adaptive_best_first_pair"] is True
    assert selected["verifier_gated_pair_growth"] is True
    assert "coupled_pair_growth" not in selected
    high_cardinality_y = np.resize(np.arange(8), len(y))
    high_cardinality = model._with_adaptive_multiclass_pair_growth(X, high_cardinality_y, base)
    assert high_cardinality["max_leaves"] == 32
    assert model._with_adaptive_multiclass_pair_growth(X[:399], y[:399], base) == base
    encoded_wide = model._with_adaptive_multiclass_pair_growth(np.zeros((450, 360)), y, base)
    assert encoded_wide["adaptive_best_first_pair"] is True
    assert model._with_adaptive_multiclass_pair_growth(np.zeros((450, 513)), y, base) == base
    large_y = np.resize(y, 3_000)
    assert model._with_adaptive_multiclass_pair_growth(np.zeros((3_000, 360)), large_y, base) == base
    model._cat_groups = ((0, 1),)
    assert model._with_adaptive_multiclass_pair_growth(X, y, base) == base


def test_multiclass_gate_deploys_rank_checkpoint_without_extra_fold_fits(monkeypatch):
    import tabpvn.predicate_compiler as compiler

    X, y = _three_class_table(rows=900)
    fits = []

    class RankClassifier:
        def fit(self, features, labels, sample_weight=None):
            fits.append(features.shape[1])
            self.classes_ = np.array([0, 1, 2])
            self.ver_ = np.arange(0, len(features), 2)
            return self

        @staticmethod
        def _scores(features):
            return np.zeros((len(features), 3))

        def _scores_at_checkpoint(self, features, metric):
            if metric != "macro_ovo_auc":
                return self._scores(features)
            first = (features[:, 0] >= 70) & (features[:, 1] <= 35)
            second = ~first & (features[:, 2] >= 65) & (features[:, 3] <= 40)
            label = np.where(first, 0, np.where(second, 1, 2))
            scores = np.zeros((len(features), 3))
            scores[np.arange(len(features)), label] = 8.0
            return scores

    class EmptyResidualMap:
        def __init__(self, *args, **kwargs):
            self.predicates = []
            self.predicate_classes_ = []
            self.proposal_objective_ = "multiclass_booster_residual_newton_gain"

        def fit(self, *args, **kwargs):
            return self

    monkeypatch.setattr(compiler, "MulticlassResidualPredicateMap", EmptyResidualMap)
    model = TabPVN(seed=4)
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: RankClassifier())
    boost = {"rounds": 30, "lr": 0.1, "depth": 3, "leaf": 5}

    mapper = model._auto_multiclass_interactions(X, y, boost)

    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert mapper is None
    assert boost["validation_metric"] == "macro_ovo_auc"
    assert report["multiclass_rank_checkpoint"]["selected"] is True
    assert report["multiclass_residual_predicate_boost"]["selected"] is False
    assert len(fits) == 2
    assert model.multiclass_architecture_report_["fold_booster_fits"] == 2


def test_multiclass_gate_selects_residual_rules_with_four_fold_fits(monkeypatch):
    X, y = _three_class_table()
    fits = []

    class RuleAwareClassifier:
        def fit(self, features, labels, sample_weight=None):
            fits.append(features.shape[1])
            self.n_features_ = features.shape[1]
            self.classes_ = np.array([0, 1, 2])
            self.ver_ = np.arange(0, len(features), 2)
            return self

        def _scores(self, features):
            scores = np.zeros((len(features), 3))
            if self.n_features_ > X.shape[1]:
                first = (features[:, 0] >= 70) & (features[:, 1] <= 35)
                second = ~first & (features[:, 2] >= 65) & (features[:, 3] <= 40)
                label = np.where(first, 0, np.where(second, 1, 2))
                scores[np.arange(len(features)), label] = 8.0
            return scores

        def _scores_at_checkpoint(self, features, metric):
            return self._scores(features)

    model = TabPVN(seed=5)
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: RuleAwareClassifier())
    boost = {"rounds": 30, "lr": 0.1, "depth": 3, "leaf": 5}

    mapper = model._auto_multiclass_interactions(X, y, boost)

    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert mapper is not None
    assert mapper.predicates
    assert boost["validation_metric"] == "macro_ovo_auc"
    assert report["multiclass_rank_checkpoint"]["selected"] is False
    assert report["multiclass_residual_predicate_boost"]["selected"] is True
    assert min(report["multiclass_residual_predicate_boost"]["fold_auc_delta"]) > 0.4
    assert len(fits) == 4
    assert model.multiclass_architecture_report_["fold_booster_fits"] == 4


def test_multiclass_head_deploys_crossfit_map_without_an_extra_booster_fit(
    monkeypatch,
):
    import tabpvn.predicate_compiler as compiler

    X, y = _three_class_table(rows=900)
    fits = []

    class FlatClassifier:
        def fit(self, features, labels, sample_weight=None):
            fits.append(features.shape[1])
            self.classes_ = np.array([0, 1, 2])
            self.ver_ = np.arange(0, len(features), 2)
            return self

        @staticmethod
        def _scores(features):
            return np.zeros((len(features), 3))

        def _scores_at_checkpoint(self, features, metric):
            return self._scores(features)

    class ResidualMap(compiler.SymbolicPredicateMap):
        MIN_FIT_ROWS = 96

        def __init__(self, seed=0, exclusive_groups=(), **_kwargs):
            super().__init__(seed=seed, exclusive_groups=exclusive_groups)
            self.predicates = [
                compiler.Predicate(
                    "threshold_and",
                    (0, 1),
                    1,
                    (69.5, 35.5),
                    (False, True),
                ),
                compiler.Predicate(
                    "threshold_and",
                    (2, 3),
                    1,
                    (64.5, 40.5),
                    (False, True),
                ),
            ]
            self.predicate_classes_ = [0, 1]
            self.predicate_updates_ = [(-1.0, 4.0), (-1.0, 4.0)]
            self.proposal_objective_ = "multiclass_booster_residual_newton_gain"

        def fit(self, *args, **kwargs):
            return self

        def residual_score_update(
            self,
            scores,
            features,
            classes,
            learning_rate=0.05,
        ):
            out = np.asarray(scores, dtype=float).copy()
            class_index = {label: index for index, label in enumerate(classes)}
            derived = self.transform(features)[:, -len(self.predicates) :] > 0.5
            for column, (owner, update) in enumerate(
                zip(self.predicate_classes_, self.predicate_updates_, strict=False)
            ):
                out[:, class_index[owner]] += learning_rate * np.where(
                    derived[:, column], update[1], update[0]
                )
            return out

    monkeypatch.setattr(compiler, "MulticlassResidualPredicateMap", ResidualMap)
    model = TabPVN(seed=6)
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: FlatClassifier())
    boost = {"rounds": 30, "lr": 0.1, "depth": 3, "leaf": 5}

    mapper = model._auto_multiclass_interactions(X, y, boost)

    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert isinstance(mapper, compiler.MulticlassCrossfitPredicateMap)
    assert report["multiclass_residual_stump_head"]["selected"] is True
    assert report["multiclass_residual_stump_head"]["deployment_objective"] == (
        "multiclass_crossfit_fold_predicate_merge"
    )
    assert len(fits) == 4
    assert boost["base_feature_count"] == X.shape[1]
    assert len(boost["residual_stumps"]) == len(mapper.predicates)


def test_multiclass_residual_stumps_are_part_of_scores_and_kernel_proof():
    rng = np.random.default_rng(139)
    raw = rng.normal(size=(600, 2))
    class_zero = raw[:, 0] > 0.7
    class_one = ~class_zero & (raw[:, 1] < -0.5)
    y = np.where(class_zero, 0, np.where(class_one, 1, 2))
    derived = np.column_stack([class_zero, class_one]).astype(float)
    X = np.column_stack([raw, derived])

    model = AdditiveCertifiedClassifier(
        rounds=0,
        refit=False,
        base_feature_count=2,
        residual_stumps=(
            (2, 0, -0.5, 4.0),
            (3, 1, -0.5, 4.0),
        ),
    ).fit(X, y)

    assert len(model.trees_) == 2
    assert [tree[1] for _class_index, tree in model.trees_] == [2, 3]
    scores = model._scores(X)
    assert scores[class_zero, 0].mean() > scores[~class_zero, 0].mean()
    assert scores[class_one, 1].mean() > scores[~class_one, 1].mean()
    assert model.kernel_certify(X[:32])["scores_reproduced"] == 1.0


def test_empty_residual_head_preserves_the_ordinary_classifier_path():
    rng = np.random.default_rng(149)
    X = rng.normal(size=(360, 4))
    y = np.argmax(
        np.column_stack([X[:, 0], X[:, 1] - 0.2 * X[:, 2], -X[:, 0]]),
        axis=1,
    )
    common = dict(rounds=12, lr=0.08, depth=3, leaf=8, refit=False, seed=7)

    ordinary = AdditiveCertifiedClassifier(**common).fit(X, y)
    bounded = AdditiveCertifiedClassifier(
        **common,
        base_feature_count=X.shape[1],
        residual_stumps=(),
    ).fit(X, y)

    np.testing.assert_allclose(bounded._scores(X), ordinary._scores(X))
