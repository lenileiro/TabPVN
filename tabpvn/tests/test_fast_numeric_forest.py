"""Parity coverage for the packed numeric serving representation."""

import numpy as np
import pytest

from tabpvn import trees as tree_module
from tabpvn.certified_boost import AdditiveCertifiedClassifier, AdditiveCertifiedRegressor, _flat_pred
from tabpvn.trees import _affine_flat_pred, _affine_tree_pred, _flatten_affine_tree


def _classification_data(seed=41):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(180, 8))
    signal = 1.4 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + rng.normal(size=len(X)) * 0.35
    return X, (signal > np.median(signal)).astype(int)


def _regression_data(seed=45):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(220, 7))
    y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2] * X[:, 3] + rng.normal(size=len(X)) * 0.15
    return X, y


def _flat_leaf(value):
    tree = ("leaf", float(value))
    flat = (
        np.array([-1], dtype=np.int64),
        np.zeros(1),
        np.array([-1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        np.array([float(value)]),
    )
    return tree, flat


def _flat_split(feature, left_value, right_value=None):
    right_value = left_value if right_value is None else right_value
    tree = (
        "node",
        int(feature),
        0.0,
        ("leaf", float(left_value)),
        ("leaf", float(right_value)),
    )
    flat = (
        np.array([feature, -1, -1], dtype=np.int64),
        np.zeros(3),
        np.array([1, -1, -1], dtype=np.int64),
        np.array([2, -1, -1], dtype=np.int64),
        np.array([0.0, float(left_value), float(right_value)]),
    )
    return tree, flat


def _manual_classifier(base, rounds):
    model = AdditiveCertifiedClassifier(seed=0)
    model.base_ = np.asarray(base, dtype=float)
    model.lr_ = 0.1
    model.classes_ = list(range(len(base)))
    model.trees_ = []
    model._flat_cache = []
    model.linear_ = False
    model._serving_forest_cache = None
    model._serving_forest_ready = False
    model._adaptive_depth_cache = None
    model._adaptive_depth_ready = False
    model._adaptive_depth_reason = None
    model._adaptive_depth_selected = True
    model.rounds = int(rounds)
    return model


def test_packed_constant_forest_matches_legacy_stage_sum():
    X, y = _classification_data()
    model = AdditiveCertifiedClassifier(
        rounds=70, lr=0.05, depth=4, leaf=12, patience=10, refit=False, seed=7
    ).fit(X, y)
    query = X[15:55]
    expected = np.tile(np.asarray(model.base_, float), (len(query), 1))
    for (cls, _tree), flat in zip(model.trees_, model._flats(), strict=False):
        expected[:, cls] += model.lr_ * _flat_pred(flat, query)

    actual = model._scores(query)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert np.array_equal(actual.argmax(1), expected.argmax(1))


def test_classifier_split_pool_preserves_original_feature_ids():
    X, y = _classification_data(seed=42)
    allowed = {0, 1, 4}
    model = AdditiveCertifiedClassifier(
        rounds=50,
        lr=0.05,
        depth=4,
        leaf=12,
        patience=10,
        refit=False,
        allowed=sorted(allowed),
        seed=8,
    ).fit(X, y)

    used = {int(feature) for flat in model._flats() for feature in flat[0] if feature >= 0}
    assert used
    assert used <= allowed


def test_packed_affine_forest_preserves_scores_and_classes():
    X, y = _classification_data(seed=43)
    model = AdditiveCertifiedClassifier(
        rounds=70, lr=0.05, depth=4, leaf=12, patience=10, refit=False, linear_leaf=True, seed=9
    ).fit(X, y)
    query = X[20:65]
    expected = np.tile(np.asarray(model.base_, float), (len(query), 1))
    for cls, tree in model.trees_:
        expected[:, cls] += model.lr_ * _affine_tree_pred(tree, query)

    actual = model._scores(query)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.array_equal(actual.argmax(1), expected.argmax(1))
    assert model.adaptive_depth_report(query) == {
        "active": False,
        "reason": "affine_leaves_require_full_scores",
        "rows": len(query),
        "probability_path": "full_forest_required",
    }


@pytest.mark.skipif(not tree_module._HAS_NUMBA, reason="adaptive packed forest requires numba")
def test_adaptive_binary_forest_exits_only_after_suffix_dominance():
    model = _manual_classifier([2.0, -2.0], rounds=40)
    positive_tree, positive_flat = _flat_leaf(1.0)
    negative_tree, negative_flat = _flat_leaf(-1.0)
    for _round in range(model.rounds):
        model.trees_.extend(((0, positive_tree), (1, negative_tree)))
        model._flat_cache.extend((positive_flat, negative_flat))
    X = np.zeros((17, 2))

    model._adaptive_depth_selected = False
    model._select_adaptive_depth(X, held_out=True)
    assert model.adaptive_depth_selection_["selected"] is True
    assert model.adaptive_depth_selection_["reason"] == "verified_stage_reduction"
    assert model.adaptive_depth_selection_["held_out"] is True

    expected = np.asarray(model.classes_)[model._scores(X).argmax(1)]
    np.testing.assert_array_equal(model.predict(X), expected)
    report = model.adaptive_depth_report(X)

    assert report["forest_kind"] == "binary_mirrored"
    assert report["predictions_match_full_forest"] is True
    assert report["rows_exited_early"] == len(X)
    assert report["mean_routing_stages"] == 8.0
    assert report["routing_stage_reduction"] == pytest.approx(0.8)


@pytest.mark.skipif(not tree_module._HAS_NUMBA, reason="adaptive packed forest requires numba")
def test_adaptive_binary_forest_runs_to_completion_when_final_scores_tie():
    model = _manual_classifier([0.0, 0.0], rounds=40)
    for round_index in range(model.rounds):
        value = 1.0 if round_index % 2 == 0 else -1.0
        first_tree, first_flat = _flat_leaf(value)
        second_tree, second_flat = _flat_leaf(-value)
        model.trees_.extend(((0, first_tree), (1, second_tree)))
        model._flat_cache.extend((first_flat, second_flat))
    X = np.zeros((9, 1))

    model._select_adaptive_depth(X, held_out=False)
    assert model.adaptive_depth_selection_["selected"] is False
    assert model.adaptive_depth_selection_["reason"] == "stage_reduction_below_gate"

    report = model.adaptive_depth_report(X)

    assert report["predictions_match_full_forest"] is True
    assert report["rows_exited_early"] == 0
    assert report["mean_routing_stages"] == model.rounds
    np.testing.assert_array_equal(model.predict(X), np.zeros(len(X), dtype=int))


@pytest.mark.skipif(not tree_module._HAS_NUMBA, reason="adaptive packed forest requires numba")
def test_adaptive_generic_and_shared_multiclass_layouts_match_full_scores():
    X = np.column_stack([np.linspace(-1.0, 1.0, 21), np.linspace(1.0, -1.0, 21)])
    generic = _manual_classifier([3.0, -1.0, -2.0], rounds=16)
    for _round in range(generic.rounds):
        for cls, (tree, flat) in enumerate(
            (
                _flat_leaf(0.8),
                _flat_split(0, -0.4, 0.4),
                _flat_split(1, -0.4, 0.2),
            )
        ):
            generic.trees_.append((cls, tree))
            generic._flat_cache.append(flat)
    generic_report = generic.adaptive_depth_report(X)

    assert generic_report["forest_kind"] == "flat"
    assert generic_report["predictions_match_full_forest"] is True
    assert generic_report["rows_exited_early"] == len(X)
    np.testing.assert_array_equal(generic.predict(X), np.zeros(len(X), dtype=int))

    shared = _manual_classifier([3.0, -1.0, -2.0], rounds=40)
    for _round in range(shared.rounds):
        for cls, (left_value, right_value) in enumerate(((0.8, -0.1), (-0.4, 0.5), (-0.4, -0.3))):
            tree, flat = _flat_split(0, left_value, right_value)
            shared.trees_.append((cls, tree))
            shared._flat_cache.append(flat)
    shared_report = shared.adaptive_depth_report(X)

    assert shared_report["forest_kind"] == "multiclass_shared"
    assert shared_report["predictions_match_full_forest"] is True
    assert shared_report["rows_exited_early"] == len(X)
    np.testing.assert_array_equal(shared.predict(X), np.zeros(len(X), dtype=int))


def test_compiled_affine_leaf_tree_matches_tuple_evaluator():
    X, y = _regression_data(seed=47)
    model = AdditiveCertifiedRegressor(
        rounds=80, lr=0.05, depth=4, leaf=14, patience=10, refit=False, linear_leaf=True, seed=11
    ).fit(X, y)
    query = X[25:70]
    tree = model.trees_[0]
    flat = _flatten_affine_tree(tree)
    assert flat is not None
    np.testing.assert_allclose(
        _affine_flat_pred(flat, query), _affine_tree_pred(tree, query), rtol=1e-12, atol=1e-12
    )


def test_packed_constant_regression_forest_matches_legacy_stage_sum():
    X, y = _regression_data()
    model = AdditiveCertifiedRegressor(
        rounds=80, lr=0.05, depth=4, leaf=14, patience=10, refit=False, seed=13
    ).fit(X, y)
    query = X[30:90]
    expected = np.full(len(query), model.base_, dtype=float)
    for flat in model._flat_cache:
        expected += model.lr_ * _flat_pred(flat, query)

    actual = model.predict(query)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert model._serving_forest()[0] == "flat"


def test_packed_affine_regression_forest_matches_legacy_stage_sum():
    X, y = _regression_data(seed=49)
    model = AdditiveCertifiedRegressor(
        rounds=80, lr=0.05, depth=4, leaf=14, patience=10, refit=False, linear_leaf=True, seed=17
    ).fit(X, y)
    query = X[35:95]
    expected = np.full(len(query), model.base_, dtype=float)
    for tree in model.trees_:
        expected += model.lr_ * _affine_tree_pred(tree, query)

    actual = model.predict(query)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert model._serving_forest()[0] == "affine"
