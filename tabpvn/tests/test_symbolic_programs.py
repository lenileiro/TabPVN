"""Deterministic predicate compiler integration tests."""

import numpy as np
import pandas as pd

from tabpvn import TabPVN
from tabpvn.predicate_compiler import (
    MulticlassCrossfitPredicateMap,
    MulticlassResidualPredicateMap,
    Predicate,
    SymbolicPredicateMap,
)
from tabpvn.trees import _multiclass_ovo_auc


def _and_map():
    mapper = SymbolicPredicateMap(seed=0)
    mapper.predicates = [Predicate("state", (0, 1), 3)]
    return mapper


def test_symbolic_map_replays_pair_state_predicate():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    augmented = _and_map().transform(X)

    assert augmented.shape == (4, 3)
    assert np.array_equal(augmented[:, -1], [0.0, 0.0, 0.0, 1.0])


def test_symbolic_compiler_discovers_xor_pair_state():
    rng = np.random.default_rng(11)
    X = rng.integers(0, 2, size=(600, 6)).astype(float)
    y = (X[:, 0] != X[:, 1]).astype(int)

    mapper = SymbolicPredicateMap(seed=0).fit(X, y)

    assert any(predicate.kind == "state" and predicate.columns == (0, 1) for predicate in mapper.predicates)
    parity = next(
        predicate
        for predicate in mapper.predicates
        if predicate.kind == "xor" and predicate.columns == (0, 1)
    )
    assert np.array_equal(mapper.transform(X)[:, X.shape[1] + mapper.predicates.index(parity)], y)
    assert mapper.transform(X).shape[1] > X.shape[1]


def test_symbolic_compiler_screens_wide_binary_facts_before_pair_search(monkeypatch):
    rng = np.random.default_rng(71)
    X = rng.integers(0, 2, size=(400, 80)).astype(float)
    y = (X[:, 0] * X[:, 1]).astype(int)
    monkeypatch.setattr(SymbolicPredicateMap, "MAX_PAIR_OPERATIONS", 400 * 16 * 16)

    mapper = SymbolicPredicateMap(seed=4).fit(X, y)

    assert mapper.source_binary_columns_ == 80
    assert mapper.screened_binary_columns_ == 16
    assert mapper.predicates


def test_symbolic_compiler_replays_pair_or_projection():
    rng = np.random.default_rng(12)
    X = rng.integers(0, 2, size=(600, 6)).astype(float)
    y = np.logical_or(X[:, 0], X[:, 1]).astype(int)

    mapper = SymbolicPredicateMap(seed=0).fit(X, y)

    projection = next(
        predicate for predicate in mapper.predicates if predicate.kind == "or" and predicate.columns == (0, 1)
    )
    assert np.array_equal(mapper.transform(X)[:, X.shape[1] + mapper.predicates.index(projection)], y)


def test_symbolic_compiler_composes_replayable_numeric_threshold_clause():
    rng = np.random.default_rng(91)
    X = rng.integers(0, 10, size=(1_200, 6)).astype(float)
    y = ((X[:, 0] >= 7) & (X[:, 1] <= 2)).astype(int)

    mapper = SymbolicPredicateMap(seed=0, numeric_rules=True).fit(X, y)
    clauses = [
        predicate
        for predicate in mapper.predicates
        if predicate.kind == "threshold_and" and set(predicate.columns) == {0, 1}
    ]

    assert clauses
    clause = clauses[0]
    derived = mapper.transform(X)[:, X.shape[1] + mapper.predicates.index(clause)]
    assert np.array_equal(derived, y)
    assert " AND " in mapper.names()[mapper.predicates.index(clause)]


def test_symbolic_compiler_replays_three_literal_threshold_clause():
    rng = np.random.default_rng(92)
    X = rng.integers(0, 3, size=(2_000, 6)).astype(float)
    y = ((X[:, 0] >= 1) & (X[:, 1] <= 1) & (X[:, 2] >= 1)).astype(int)

    mapper = SymbolicPredicateMap(seed=0, numeric_rules=True).fit(X, y)
    clause = next(
        predicate
        for predicate in mapper.predicates
        if predicate.kind == "threshold_and"
        and len(predicate.columns) == 3
        and set(predicate.columns) == {0, 1, 2}
    )
    derived = mapper.transform(X)[:, X.shape[1] + mapper.predicates.index(clause)]

    assert np.array_equal(derived, y)
    assert mapper.names()[mapper.predicates.index(clause)].count(" AND ") == 2


def test_residual_compiler_prioritizes_the_boosters_missing_cardinality_concept():
    rng = np.random.default_rng(121)
    facts = rng.integers(0, 2, size=(8_000, 6)).astype(float)
    numeric = rng.random((8_000, 4))
    X = np.column_stack([facts, numeric])
    y = ((facts.sum(axis=1) == 3) & (numeric[:, 0] > 0.95)).astype(int)
    # The baseline already models the tail but not the exact-cardinality rule.
    probability = np.where(numeric[:, 0] > 0.95, 0.22, 0.002)
    residual = y - probability
    hessian = probability * (1.0 - probability)

    mapper = SymbolicPredicateMap(seed=0, rare_rules=True, rare_class=1).fit(
        X,
        y,
        residual=residual,
        hessian=hessian,
    )

    first = mapper.predicates[0]
    assert first.kind == "count"
    assert set(first.columns) == set(range(6))
    assert first.value == 3
    derived = mapper.transform(X)[:, X.shape[1]]
    np.testing.assert_array_equal(derived, facts.sum(axis=1) == 3)


def test_multiclass_residual_compiler_balances_replayable_rules_across_classes():
    rng = np.random.default_rng(122)
    X = rng.integers(0, 100, size=(4_000, 6)).astype(float)
    first = (X[:, 0] >= 75) & (X[:, 1] <= 30)
    second = ~first & (X[:, 2] >= 70) & (X[:, 3] <= 35)
    y = np.where(first, 0, np.where(second, 1, 2))
    probability = np.full((len(y), 3), 1.0 / 3.0)

    mapper = MulticlassResidualPredicateMap(seed=3, max_predicates=9).fit(
        X,
        y,
        probability,
        classes=np.array([0, 1, 2]),
    )

    assert mapper.predicates
    assert len(mapper.predicates) <= 9
    assert set(mapper.predicate_classes_) == {0, 1, 2}
    assert mapper.transform(X).shape[1] == X.shape[1] + len(mapper.predicates)
    assert len(mapper.names()) == len(mapper.predicates)
    updated = mapper.residual_score_update(
        np.log(probability),
        X,
        classes=np.array([0, 1, 2]),
        learning_rate=0.1,
    )
    assert _multiclass_ovo_auc(updated, y) > 0.7


def test_multiclass_crossfit_map_merges_fold_predicates_on_unseen_rows():
    rng = np.random.default_rng(125)
    X = rng.integers(0, 2, size=(600, 6)).astype(float)
    y = np.where(X[:, 0] > 0.5, 0, np.where(X[:, 1] > 0.5, 1, 2))
    probability = np.full((len(y), 3), 1.0 / 3.0)
    folds = (np.arange(0, 300), np.arange(300, 600))
    first = MulticlassResidualPredicateMap(seed=1)
    first.predicates = [Predicate("state", (0, 2), 2)]
    first.predicate_classes_ = [0]
    second = MulticlassResidualPredicateMap(seed=2)
    second.predicates = [Predicate("state", (1, 3), 2)]
    second.predicate_classes_ = [1]

    mapper = MulticlassCrossfitPredicateMap(seed=3).fit_from_folds(
        X,
        y,
        probability,
        classes=np.array([0, 1, 2]),
        fold_maps=(first, second),
        valid_rows=folds,
    )

    assert mapper.predicates == [first.predicates[0], second.predicates[0]]
    assert mapper.predicate_classes_ == [0, 1]
    assert mapper.predicate_evidence_rows_ == [300, 300]
    updated = mapper.residual_score_update(
        np.log(probability),
        X,
        classes=np.array([0, 1, 2]),
        learning_rate=0.1,
    )
    assert _multiclass_ovo_auc(updated, y) > _multiclass_ovo_auc(np.log(probability), y)


def test_rare_compiler_discovers_extreme_tail_conjunction():
    rng = np.random.default_rng(93)
    X = rng.integers(0, 100, size=(10_000, 5)).astype(float)
    y = ((X[:, 0] >= 95) & (X[:, 1] <= 9)).astype(int)

    mapper = SymbolicPredicateMap(seed=0, rare_rules=True, rare_class=1).fit(X, y)
    clauses = [
        predicate
        for predicate in mapper.predicates
        if predicate.kind == "threshold_and" and set(predicate.columns) == {0, 1}
    ]

    assert clauses
    derived = mapper.transform(X)[:, X.shape[1] + mapper.predicates.index(clauses[0])]
    assert (derived[y == 1] == 1.0).mean() >= 0.9
    assert derived.mean() < 0.02


def test_rare_compiler_materializes_same_feature_interval():
    rng = np.random.default_rng(94)
    X = rng.integers(0, 10, size=(4_000, 4)).astype(float)
    interval = (X[:, 0] >= 4) & (X[:, 0] <= 5)
    y = (interval & (X[:, 1] >= 9)).astype(int)

    mapper = SymbolicPredicateMap(seed=0, rare_rules=True, rare_class=1).fit(X, y)
    predicate = next(
        predicate
        for predicate in mapper.predicates
        if predicate.kind == "threshold_interval" and predicate.columns == (0, 0)
    )
    derived = mapper.transform(X)[:, X.shape[1] + mapper.predicates.index(predicate)]

    assert np.array_equal(derived.astype(bool), interval)
    assert " AND " in mapper.names()[mapper.predicates.index(predicate)]


def test_threshold_union_replays_exact_disjoint_intervals_and_names_branches():
    X = np.arange(-1.0, 7.0)[:, None]
    mapper = SymbolicPredicateMap(seed=0)
    mapper.predicates = [
        Predicate(
            "threshold_union",
            (0, 0, 0, 0),
            1,
            (0.0, 2.0, 3.0, 5.0),
            (False, True, False, True),
        )
    ]

    derived = mapper.transform(X)[:, 1]

    assert np.array_equal(
        derived.astype(bool),
        ((X[:, 0] > 0.0) & (X[:, 0] <= 2.0)) | ((X[:, 0] > 3.0) & (X[:, 0] <= 5.0)),
    )
    assert mapper.names(["amount"]) == ["(amount > 0 AND amount <= 2) OR (amount > 3 AND amount <= 5)"]


def test_residual_compiler_keeps_disjoint_same_feature_risk_bands():
    rng = np.random.default_rng(403)
    X = rng.integers(0, 100, size=(10_000, 5)).astype(float)
    first_band = (X[:, 0] >= 10) & (X[:, 0] <= 15)
    second_band = (X[:, 0] >= 80) & (X[:, 0] <= 85)
    event = (first_band | second_band) & (X[:, 1] >= 80)
    y = event.astype(int)
    probability = np.where(X[:, 1] >= 80, 0.12, 0.001)

    mapper = SymbolicPredicateMap(seed=0, rare_rules=True, rare_class=1).fit(
        X,
        y,
        residual=y - probability,
        hessian=probability * (1.0 - probability),
    )
    intervals = [
        (index, predicate)
        for index, predicate in enumerate(mapper.predicates)
        if predicate.kind == "threshold_interval" and predicate.columns == (0, 0)
    ]
    mapped = mapper.transform(X)
    covered = np.logical_or.reduce(
        [mapped[:, X.shape[1] + index].astype(bool) for index, _predicate in intervals]
    )
    interval_unions = [
        (index, predicate)
        for index, predicate in enumerate(mapper.predicates)
        if predicate.kind == "threshold_union" and predicate.columns == (0, 0, 0, 0)
    ]

    assert len(intervals) == 2
    assert len(interval_unions) == 1
    union_index, _union = interval_unions[0]
    assert np.array_equal(mapped[:, X.shape[1] + union_index].astype(bool), covered)
    assert covered[first_band | second_band].mean() > 0.99
    assert mapper.residual_interval_predicates_ >= 1
    assert mapper.multi_interval_columns_ >= 1
    assert mapper.interval_predicates_selected_ <= mapper.MAX_RARE_INTERVALS
    assert 1 <= mapper.interval_union_candidates_
    assert mapper.interval_union_predicates_selected_ <= mapper.MAX_RARE_INTERVAL_UNIONS


def test_residual_allocator_preserves_the_incumbent_prefix_without_interval_evidence():
    mapper = SymbolicPredicateMap(seed=0, rare_rules=True, rare_class=1)
    boolean = [Predicate("count", (index,), index % 2) for index in range(16)]
    numeric = [
        Predicate("threshold_interval", (index, index), 1, (0.0, 1.0), (False, True)) for index in range(4)
    ]
    numeric += [
        Predicate("threshold_and", (index, index + 1), 1, (0.0, 0.0), (False, False)) for index in range(8)
    ]
    candidates = boolean + numeric

    selected = mapper._select_residual_candidates(candidates)

    assert selected == candidates[: mapper.MAX_RESIDUAL_PREDICATES]
    assert mapper.residual_allocator_ == "ordered_prefix"
    assert mapper.residual_selected_family_counts_ == {
        "boolean": 16,
        "interval": 4,
        "numeric_composition": 0,
    }


def test_residual_allocator_reserves_capacity_for_multimodal_numeric_prefix():
    mapper = SymbolicPredicateMap(seed=0, rare_rules=True, rare_class=1)
    boolean = [Predicate("count", (index,), index % 2) for index in range(16)]
    intervals = [
        Predicate("threshold_interval", (index, index), 1, (0.0, 1.0), (False, True)) for index in range(5)
    ]
    composition = Predicate("threshold_and", (5, 6), 1, (0.0, 0.0), (False, False))
    candidates = boolean + intervals + [composition]

    first = mapper._select_residual_candidates(candidates)
    second = mapper._select_residual_candidates(candidates)

    assert first == second == boolean[:14] + intervals + [composition]
    assert len(first) == mapper.MAX_RESIDUAL_PREDICATES
    assert mapper.residual_allocator_ == "multimodal_numeric_reserve"
    assert mapper.residual_selected_family_counts_ == {
        "boolean": 14,
        "interval": 5,
        "numeric_composition": 1,
    }


def test_residual_allocator_counts_explicit_union_branches_as_interval_evidence():
    mapper = SymbolicPredicateMap(seed=0, rare_rules=True, rare_class=1)
    boolean = [Predicate("count", (index,), index % 2) for index in range(16)]
    union = Predicate(
        "threshold_union",
        (0, 0, 0, 0),
        1,
        (0.0, 1.0, 2.0, 3.0),
        (False, True, False, True),
    )
    intervals = [
        Predicate("threshold_interval", (index, index), 1, (0.0, 1.0), (False, True)) for index in range(1, 4)
    ]
    compositions = [
        Predicate("threshold_and", (index, index + 1), 1, (0.0, 0.0), (False, False)) for index in range(2)
    ]
    numeric = [union, *intervals, *compositions]

    selected = mapper._select_residual_candidates([*boolean, *numeric])

    assert selected == [*boolean[:14], *numeric]
    assert mapper.residual_allocator_ == "multimodal_numeric_reserve"
    assert mapper.residual_selected_family_counts_ == {
        "boolean": 14,
        "interval": 4,
        "numeric_composition": 2,
    }


def test_auto_numeric_gate_requires_a_wide_bounded_ordinal_schema():
    bounded = np.tile(np.arange(8, dtype=float)[:, None], (5, 24))

    assert SymbolicPredicateMap.auto_numeric_applicable(bounded)
    assert not SymbolicPredicateMap.auto_numeric_applicable(bounded[:, :23])
    assert not SymbolicPredicateMap.auto_numeric_applicable(
        np.tile(np.arange(32, dtype=float)[:, None], (1, 24))
    )


def test_threshold_layer_cannot_displace_a_verified_binary_program(monkeypatch):
    import tabpvn.predicate_compiler as compiler

    class FakeMap:
        def __init__(self, seed=0, exclusive_groups=(), numeric_rules=False):
            self.numeric_rules = numeric_rules
            self.predicates = []

        @staticmethod
        def applicable(X, y, numeric_rules=False):
            return True

        @staticmethod
        def auto_numeric_applicable(X):
            return True

        def fit(self, X, y):
            self.predicates = [Predicate("state", (0, 1), 3)]
            if self.numeric_rules:
                self.predicates.append(Predicate("threshold_and", (0, 1), 1, (0.5, 0.5), (False, False)))
            return self

        def transform(self, X):
            X = np.asarray(X, float)
            exact = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(float)
            out = np.column_stack([X, exact])
            if self.numeric_rules:
                out = np.column_stack([out, X[:, 0] != X[:, 1]])
            return out

    class FakeClassifier:
        def fit(self, X, y):
            self.n_features_ = X.shape[1]
            self.classes_ = np.array([0, 1])
            return self

        def _scores(self, X):
            if self.n_features_ == 2:
                signal = X[:, 0]
            elif self.n_features_ == 3:
                signal = X[:, 2]
            else:
                signal = 1.0 - X[:, 0]
            return np.column_stack([-signal, signal])

    monkeypatch.setattr(compiler, "SymbolicPredicateMap", FakeMap)
    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), (200, 1))
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(int)
    model = TabPVN(seed=0)
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: FakeClassifier())

    mapper = model._auto_interactions(
        X,
        y,
        {"rounds": 100, "depth": 3, "leaf": 10, "lr": 0.05},
    )

    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert mapper is not None and mapper.numeric_rules is False
    assert report["symbolic_predicate_boost"]["selected"] is True
    assert report["threshold_predicate_boost"]["selected"] is False


def test_large_symbolic_gate_uses_bounded_stratified_evidence(monkeypatch):
    import tabpvn.predicate_compiler as compiler

    fit_rows = []

    class SampledMap:
        MAX_ROWS = 1_000

        def __init__(self, seed=0, exclusive_groups=(), numeric_rules=False):
            self.predicates = []
            self.source_binary_columns_ = 2
            self.screened_binary_columns_ = 2

        @staticmethod
        def applicable(X, y, numeric_rules=False):
            return True

        @staticmethod
        def auto_numeric_applicable(X):
            return False

        def fit(self, X, y):
            fit_rows.append(len(X))
            self.predicates = [Predicate("state", (0, 1), 3)]
            return self

        def transform(self, X):
            X = np.asarray(X, float)
            return np.column_stack([X, X[:, 0] * X[:, 1]])

    class RankClassifier:
        def fit(self, X, y):
            self.width = X.shape[1]
            self.classes_ = np.array([0, 1])
            return self

        def _scores(self, X):
            signal = X[:, 2] if self.width == 3 else X[:, 0]
            return np.column_stack([-signal, signal])

    monkeypatch.setattr(compiler, "SymbolicPredicateMap", SampledMap)
    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), (3_000, 1))
    y = (X[:, 0] * X[:, 1]).astype(int)
    model = TabPVN(seed=2)
    monkeypatch.setattr(model, "_classifier", lambda **_kwargs: RankClassifier())

    mapper = model._auto_interactions(X, y, {"rounds": 100, "depth": 3, "leaf": 10, "lr": 0.05})

    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert mapper is not None
    assert max(fit_rows) == 1_000
    assert report["symbolic_predicate_boost"]["source_rows"] == 12_000
    assert report["symbolic_predicate_boost"]["evidence_rows"] == 1_000


def test_symbolic_compiler_rejects_redundant_mutually_exclusive_states():
    rng = np.random.default_rng(23)
    state = rng.integers(0, 3, size=600)
    X = np.column_stack(
        [state == 0, state == 1, rng.integers(0, 2, size=600), rng.integers(0, 2, size=600)]
    ).astype(float)
    y = (X[:, 0] * X[:, 2]).astype(int)

    mapper = SymbolicPredicateMap(seed=0, exclusive_groups=((0, 1),)).fit(X, y)

    assert all(
        not (predicate.kind == "state" and set(predicate.columns) == {0, 1} and predicate.value != 0)
        for predicate in mapper.predicates
    )


def test_cross_fitted_gate_selects_exact_six_fact_cardinality_program():
    rng = np.random.default_rng(52)
    X = rng.integers(0, 2, size=(800, 12)).astype(float)
    y = (X[:, :6].sum(1) == 3).astype(int)

    model = TabPVN(seed=0)
    mapper = model._auto_interactions(X, y, {"rounds": 300, "depth": 6, "leaf": 20, "lr": 0.05})

    assert mapper is not None
    assert any(predicate.kind == "count" and len(predicate.columns) == 6 for predicate in mapper.predicates)
    report = {entry["name"]: entry for entry in model.candidate_report_}
    assert report["symbolic_predicate_boost"]["selected"] is True
    assert min(report["symbolic_predicate_boost"]["fold_auc_delta"]) > 0.003


def test_selected_symbolic_program_is_replayed_by_prediction_and_certificate(monkeypatch):
    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, size=(160, 2)).astype(float)
    y = (X[:, 0] * X[:, 1]).astype(int)
    mapper = _and_map()
    monkeypatch.setattr(TabPVN, "_auto_interactions", lambda _self, _X, _y, _boost: mapper)

    model = TabPVN(seed=0).fit(X, y)

    assert model.interaction_features_ == ["feature[0]=1 AND feature[1]=1"]
    assert model._X(X).shape[1] == 3
    assert np.array_equal(model.predict(X), y)
    assert model.certify(X[:20]) == 1.0


def test_symbolic_program_names_follow_dataframe_columns(monkeypatch):
    X = pd.DataFrame({"left": [0, 0, 1, 1] * 40, "right": [0, 1, 0, 1] * 40})
    y = (X["left"] & X["right"]).to_numpy()
    monkeypatch.setattr(TabPVN, "_auto_interactions", lambda *_args: _and_map())

    model = TabPVN(seed=0).fit(X, y)

    assert model.interaction_features_ == ["left=1 AND right=1"]
    assert model._X(X.iloc[:3]).shape[1] == 3
