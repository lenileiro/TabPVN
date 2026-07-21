"""Parity and scheduling coverage for concurrent final softmax fits."""

import numpy as np
import pytest

import tabpvn.certified_boost as boost_module
import tabpvn.trees as trees
from tabpvn.certified_boost import AdditiveCertifiedClassifier
from tabpvn.trees import (
    _default_multiclass_feature_fraction,
    _fit_tree_2nd_binned,
    _fit_tree_softmax_pair_binned,
    _fit_tree_softmax_shared_binned,
    _flat_leaf_ids,
    _hardest_class_pair,
    _honest_pair_partition,
    _reestimate_numeric_newton_leaves,
    _softmax_pair_newton_terms,
    _tree_pred,
    _tree_pred_rows,
    boost_softmax_predict,
    reason_boost_softmax,
)


def test_hardest_class_pair_balances_each_class_and_breaks_ties_stably():
    scores = np.array(
        [
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 2.9],
            [0.0, 2.9, 3.0],
        ]
    )
    target = np.array([0, 0, 1, 2])

    assert _hardest_class_pair(scores, target) == (1, 2)
    assert _hardest_class_pair(np.zeros_like(scores), target) == (0, 1)


def test_high_cardinality_feature_search_is_full_only_when_work_is_bounded():
    assert _default_multiclass_feature_fraction(1_132, 360, 8, True) == 1.0
    assert _default_multiclass_feature_fraction(10_000, 128, 8, True) == 0.7
    assert _default_multiclass_feature_fraction(1_132, 360, 7, True) == 0.7
    assert _default_multiclass_feature_fraction(1_132, 360, 8, False) == 0.7
    assert _default_multiclass_feature_fraction(100_000, 20, 8, True) == 1.0


def test_hard_pair_growth_spends_leaf_budget_only_on_selected_logits(monkeypatch):
    rng = np.random.default_rng(904)
    X = rng.normal(size=(300, 6))
    y = np.repeat(np.arange(3), 100)
    calls = []
    original = trees._fit_tree_2nd_binned

    def record_growth(*args, **kwargs):
        calls.append((args[4], kwargs.get("max_leaves")))
        return original(*args, **kwargs)

    monkeypatch.setattr(trees, "_fit_tree_2nd_binned", record_growth)
    reason_boost_softmax(
        X,
        y,
        rounds=2,
        depth=3,
        min_leaf=8,
        holdout=0.2,
        refit=False,
        stratified_holdout=True,
        sub=1.0,
        mf=1.0,
        max_leaves=8,
        best_first_pair=True,
    )

    assert calls[:3] == [(3, None)] * 3
    assert sorted(calls[3:]) == [(3, None), (7, 8), (7, 8)]


def test_hard_pair_growth_replays_through_certified_scalar_trees():
    rng = np.random.default_rng(906)
    X = rng.normal(size=(360, 6))
    y = np.argmax(
        np.column_stack(
            [
                X[:, 0] - 0.5 * X[:, 1],
                X[:, 1] + X[:, 2] * X[:, 3],
                -X[:, 0] - X[:, 2] * X[:, 3],
            ]
        ),
        axis=1,
    )
    model = AdditiveCertifiedClassifier(
        rounds=12,
        lr=0.05,
        depth=3,
        leaf=10,
        patience=5,
        refit=False,
        max_leaves=8,
        best_first_pair=True,
        seed=17,
    ).fit(X, y)

    assert model.kernel_certify(X, n_trees=18, sample=40)["scores_reproduced"] == 1.0


@pytest.mark.parametrize("activate", [False, True])
def test_adaptive_pair_monitor_has_bounded_inactive_probe(monkeypatch, activate):
    rng = np.random.default_rng(909)
    X = rng.normal(size=(360, 6))
    y = np.repeat(np.arange(3), 120)
    calls = []

    class Tracker:
        records = []

        def __init__(self, *_args, **_kwargs):
            pass

        def observe(self, *_args, **_kwargs):
            calls.append(len(calls))
            return (0, 1) if activate and len(calls) >= 2 else ()

    monkeypatch.setattr(trees, "ResidualDynamicsTracker", Tracker)
    result = reason_boost_softmax(
        X,
        y,
        rounds=24,
        depth=3,
        min_leaf=8,
        holdout=0.2,
        patience=30,
        refit=False,
        stratified_holdout=True,
        sub=1.0,
        mf=1.0,
        max_leaves=8,
        best_first_pair=True,
        adaptive_best_first_pair=True,
    )
    trace = result[-1]

    assert len(trace["pair_growth_schedule"]) == 24
    assert len(calls) == (24 if activate else trees._ADAPTIVE_PAIR_PROBE_ROUNDS)
    assert trace["dynamics_monitored_rounds"] == len(calls)


def test_shared_softmax_tree_uses_one_partition_with_centered_vector_leaves():
    rng = np.random.default_rng(901)
    X = rng.normal(size=(800, 10))
    Xb, edges, bins = trees._prebin(X, 64)
    raw = rng.normal(size=(len(X), 4))
    probability = np.exp(raw - raw.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    target = rng.integers(0, 4, size=len(X))
    gradient = probability - np.eye(4)[target]
    hessian = probability * (1.0 - probability)

    shared_trees, selected_prediction, flats, _binned = _fit_tree_softmax_shared_binned(
        Xb,
        edges,
        gradient,
        hessian,
        depth=4,
        min_leaf=20,
        feats=np.arange(X.shape[1]),
        lam=1.0,
        NB=bins,
    )

    assert len(shared_trees) == 4
    for flat in flats[1:]:
        for reference, actual in zip(flats[0][:4], flat[:4], strict=False):
            np.testing.assert_array_equal(actual, reference)
    np.testing.assert_allclose(np.column_stack([flat[4] for flat in flats]).sum(1), 0.0, atol=1e-14)
    for cls, flat in enumerate(flats):
        np.testing.assert_array_equal(_tree_pred(flat, X), selected_prediction[:, cls])


def test_softmax_pair_tree_uses_off_diagonal_curvature_and_opposite_leaves():
    rng = np.random.default_rng(907)
    X = rng.normal(size=(500, 7))
    Xb, edges, bins = trees._prebin(X, 64)
    raw = rng.normal(size=(len(X), 3))
    probability = np.exp(raw - raw.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    target = np.eye(3)[rng.integers(0, 3, size=len(X))]
    weight = 0.5 + rng.random(len(X))
    gradient, hessian = _softmax_pair_newton_terms(probability, target, weight, (0, 2))

    direction = np.array([1.0, 0.0, -1.0])
    expected_hessian = np.empty(len(X))
    for row, p in enumerate(probability):
        full_hessian = np.diag(p) - np.outer(p, p)
        expected_hessian[row] = weight[row] * direction @ full_hessian @ direction
    np.testing.assert_allclose(hessian, expected_hessian)

    pair_trees, selected_prediction, pair_flats = _fit_tree_softmax_pair_binned(
        Xb,
        edges,
        gradient,
        hessian,
        depth=7,
        min_leaf=12,
        feats=np.arange(X.shape[1]),
        lam=1.0,
        NB=bins,
        max_leaves=12,
    )

    for reference, opposite in zip(pair_flats[0][:4], pair_flats[1][:4], strict=False):
        np.testing.assert_array_equal(reference, opposite)
    np.testing.assert_array_equal(pair_flats[0][4], -pair_flats[1][4])
    np.testing.assert_array_equal(selected_prediction[:, 0], -selected_prediction[:, 1])
    for column, flat in enumerate(pair_flats):
        np.testing.assert_allclose(_tree_pred(flat, X), selected_prediction[:, column])
        assert pair_trees[column][0] == "node"


def test_honest_newton_refit_preserves_topology_and_uses_only_estimation_rows():
    rng = np.random.default_rng(905)
    X = rng.normal(size=(500, 7))
    Xb, edges, bins = trees._prebin(X[:320], 64)
    structure_gradient = rng.normal(size=320)
    structure_hessian = 0.2 + rng.random(320)
    _tree, _prediction, flat = _fit_tree_2nd_binned(
        Xb,
        edges,
        structure_gradient,
        structure_hessian,
        depth=5,
        min_leaf=12,
        feats=np.arange(X.shape[1]),
        lam=1.0,
        NB=bins,
    )
    estimation_X = X[320:]
    estimation_gradient = rng.normal(size=len(estimation_X))
    estimation_hessian = 0.2 + rng.random(len(estimation_X))

    honest_tree, honest_flat = _reestimate_numeric_newton_leaves(
        flat,
        estimation_X,
        estimation_gradient,
        estimation_hessian,
        lam=1.0,
    )

    for original, honest in zip(flat[:4], honest_flat[:4], strict=False):
        np.testing.assert_array_equal(honest, original)
    leaf_id = _flat_leaf_ids(flat, estimation_X)
    for leaf in np.flatnonzero(flat[0] < 0):
        rows = leaf_id == leaf
        expected = -estimation_gradient[rows].sum() / (estimation_hessian[rows].sum() + 1.0)
        assert honest_flat[4][leaf] == pytest.approx(expected)
    np.testing.assert_array_equal(_tree_pred(honest_tree, X), _tree_pred(honest_flat, X))

    selected = np.arange(120) % 3 != 0
    first = _honest_pair_partition(selected, np.random.default_rng(11), min_leaf=8)
    second = _honest_pair_partition(selected, np.random.default_rng(11), min_leaf=8)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.any(first[0] & first[1])
    assert np.all(first[1] == ~first[0])


@pytest.mark.parametrize("refit", [False, True])
def test_adaptive_honest_pair_growth_keeps_scalar_proofs(monkeypatch, refit):
    rng = np.random.default_rng(910)
    X = rng.normal(size=(480, 7))
    y = np.argmax(
        np.column_stack(
            [
                X[:, 0] - 0.4 * X[:, 1],
                X[:, 1] + X[:, 2] * X[:, 3],
                -X[:, 0] - X[:, 2] * X[:, 3],
            ]
        ),
        axis=1,
    )
    reestimated_rows = []
    original_reestimate = trees._reestimate_numeric_newton_leaves

    class Tracker:
        def __init__(self, *_args, **_kwargs):
            self.records = []

        def observe(self, *_args, **_kwargs):
            return (0, 1)

    def record_reestimate(flat, features, gradient, hessian, lam):
        reestimated_rows.append(len(features))
        return original_reestimate(flat, features, gradient, hessian, lam)

    monkeypatch.setattr(trees, "ResidualDynamicsTracker", Tracker)
    monkeypatch.setattr(trees, "_reestimate_numeric_newton_leaves", record_reestimate)
    model = AdditiveCertifiedClassifier(
        rounds=8,
        lr=0.05,
        depth=3,
        leaf=10,
        patience=12,
        refit=refit,
        max_leaves=8,
        best_first_pair=True,
        adaptive_best_first_pair=True,
        honest_pair_growth=True,
        seed=20,
    ).fit(X, y)

    assert reestimated_rows
    assert all(10 <= rows < len(X) for rows in reestimated_rows)
    assert any(model.pair_growth_schedule_)
    assert model.kernel_certify(X, n_trees=18, sample=30)["scores_reproduced"] == 1.0


@pytest.mark.parametrize(
    ("linear_leaf", "refit"),
    [(False, False), (True, False), (False, True)],
)
def test_adaptive_coupled_pair_growth_keeps_scalar_proofs(monkeypatch, linear_leaf, refit):
    rng = np.random.default_rng(908)
    X = rng.normal(size=(420, 6))
    y = np.argmax(
        np.column_stack(
            [
                X[:, 0] - 0.5 * X[:, 1],
                X[:, 1] + X[:, 2] * X[:, 3],
                -X[:, 0] - X[:, 2] * X[:, 3],
            ]
        ),
        axis=1,
    )

    class Tracker:
        def __init__(self, *_args, **_kwargs):
            self.records = []

        def observe(self, *_args, **_kwargs):
            return (0, 1)

    monkeypatch.setattr(trees, "ResidualDynamicsTracker", Tracker)
    model = AdditiveCertifiedClassifier(
        rounds=8,
        lr=0.05,
        depth=3,
        leaf=10,
        patience=12,
        refit=refit,
        max_leaves=8,
        best_first_pair=True,
        adaptive_best_first_pair=True,
        coupled_pair_growth=True,
        linear_leaf=linear_leaf,
        seed=18,
    ).fit(X, y)

    active_rounds = [(index, pair) for index, pair in enumerate(model.pair_growth_schedule_) if pair]
    assert active_rounds
    for round_index, pair in active_rounds:
        round_flats = model._flats()[round_index * 3 : (round_index + 1) * 3]
        first, second = pair
        for reference, opposite in zip(round_flats[first][:4], round_flats[second][:4], strict=False):
            np.testing.assert_array_equal(reference, opposite)
        np.testing.assert_array_equal(round_flats[first][4], -round_flats[second][4])
    assert model.kernel_certify(X, n_trees=18, sample=30)["scores_reproduced"] == 1.0


def test_best_first_newton_tree_honors_leaf_budget_and_flat_routing():
    rng = np.random.default_rng(903)
    X = rng.normal(size=(500, 7))
    Xb, edges, bins = trees._prebin(X, 64)
    gradient = rng.normal(size=len(X))
    hessian = 0.2 + rng.random(len(X))

    tree, selected_prediction, flat = _fit_tree_2nd_binned(
        Xb,
        edges,
        gradient,
        hessian,
        depth=12,
        min_leaf=12,
        feats=np.arange(X.shape[1]),
        lam=1.0,
        NB=bins,
        max_leaves=16,
    )

    assert int(np.sum(flat[0] < 0)) == 16
    np.testing.assert_allclose(_tree_pred(flat, X), selected_prediction)
    assert tree[0] == "node"


def test_shared_multiclass_serving_matches_scalar_proof_trees_exactly():
    rng = np.random.default_rng(902)
    X = rng.normal(size=(700, 9))
    y = np.argmax(
        np.column_stack(
            [
                1.2 * X[:, 0] - X[:, 1],
                X[:, 1] + 0.8 * X[:, 2],
                -X[:, 0] - 0.7 * X[:, 2],
            ]
        ),
        axis=1,
    )
    model = AdditiveCertifiedClassifier(
        rounds=25,
        lr=0.05,
        depth=4,
        leaf=15,
        patience=8,
        refit=False,
        shared_structure=True,
        seed=12,
    ).fit(X, y)
    expected = np.tile(model.base_, (len(X), 1))
    for (cls, _tree), flat in zip(model.trees_, model._flats(), strict=False):
        expected[:, cls] += model.lr_ * _tree_pred(flat, X)

    assert model._serving_forest()[0] == "multiclass_shared"
    np.testing.assert_array_equal(model._scores(X), expected)
    assert model.kernel_certify(X, n_trees=12, sample=30)["scores_reproduced"] == 1.0


def test_parallel_softmax_growth_matches_serial_scores_and_classes():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(500, 14))
    y = np.mod(np.argsort(1.1 * X[:, 0] - 0.8 * X[:, 1] + X[:, 2] * X[:, 3]), 3)
    common = dict(rounds=80, lr=0.05, depth=4, min_leaf=14, patience=10, refit=False, seed=5)

    serial = reason_boost_softmax(X, y, parallel_k=False, **common)
    concurrent = reason_boost_softmax(X, y, parallel_k=True, **common)
    serial_class, serial_confidence = boost_softmax_predict(serial, X)
    concurrent_class, concurrent_confidence = boost_softmax_predict(concurrent, X)

    np.testing.assert_array_equal(concurrent_class, serial_class)
    np.testing.assert_allclose(concurrent_confidence, serial_confidence, rtol=0.0, atol=0.0)
    assert len(concurrent[2]) == len(serial[2])


@pytest.mark.skipif(not trees._HAS_NUMBA, reason="parallel histogram kernel requires numba")
def test_parallel_histogram_growth_matches_serial_tree_exactly():
    rng = np.random.default_rng(831)
    n, width, bins = 20_000, 16, 64
    Xb = rng.integers(0, bins, size=(n, width), dtype=np.uint8)
    edges = [np.arange(bins - 1, dtype=float) for _ in range(width)]
    gradient = rng.normal(size=n)
    hessian = 0.2 + rng.random(n)
    common = dict(depth=3, min_leaf=20, feats=np.arange(width), lam=1.0, NB=bins)

    serial = _fit_tree_2nd_binned(Xb, edges, gradient, hessian, parallel_hist=False, **common)
    parallel = _fit_tree_2nd_binned(Xb, edges, gradient, hessian, parallel_hist=True, **common)

    np.testing.assert_array_equal(parallel[1], serial[1])
    for parallel_array, serial_array in zip(parallel[2], serial[2], strict=False):
        np.testing.assert_array_equal(parallel_array, serial_array)


def test_parallel_prebin_matches_serial_bins_and_edges_exactly(monkeypatch):
    X = np.random.default_rng(912).normal(size=(12_000, 12))
    monkeypatch.setattr(trees, "_PARALLEL_PREBIN_MIN_WORK", X.size + 1)
    serial_bins, serial_edges, serial_count = trees._prebin(X, 64)
    monkeypatch.setattr(trees, "_PARALLEL_PREBIN_MIN_WORK", 0)
    parallel_bins, parallel_edges, parallel_count = trees._prebin(X, 64)

    assert parallel_count == serial_count
    np.testing.assert_array_equal(parallel_bins, serial_bins)
    for parallel_edge, serial_edge in zip(parallel_edges, serial_edges, strict=False):
        np.testing.assert_array_equal(parallel_edge, serial_edge)


@pytest.mark.skipif(not trees._HAS_NUMBA, reason="compact gather parity requires numba")
def test_compact_binary_tree_sample_matches_full_width_tree_exactly():
    rng = np.random.default_rng(219)
    X = rng.normal(size=(5_000, 12))
    Xb, edges, bins = trees._prebin(X, 64)
    probability = 0.1 + 0.8 * rng.random(len(X))
    target_zero = rng.random(len(X)) < 0.45
    weight = 0.5 + rng.random(len(X))
    selected = rng.random(len(X)) < 0.7
    rows = np.flatnonzero(selected)
    feats = np.array([8, 2, 11, 0, 5], dtype=np.int64)
    gradient = weight * (probability - target_zero)
    hessian = weight * probability * (1.0 - probability)

    full = _fit_tree_2nd_binned(
        Xb[selected],
        edges,
        gradient[selected],
        hessian[selected],
        depth=4,
        min_leaf=20,
        feats=feats,
        lam=1.0,
        NB=bins,
    )
    Xb_compact, gradient_compact, hessian_compact = trees._gather_binary_tree_fit_data(
        Xb, probability, target_zero, weight, rows, feats
    )
    compact = _fit_tree_2nd_binned(
        Xb_compact,
        [edges[int(j)] for j in feats],
        gradient_compact,
        hessian_compact,
        depth=4,
        min_leaf=20,
        feats=np.arange(len(feats)),
        lam=1.0,
        NB=bins,
        return_binned=True,
    )
    compact_tree = trees._remap_tree_features(compact[0], feats)
    compact_flat = trees._remap_flat_features(compact[2], feats)
    compact_binned = trees._remap_flat_features(compact[3], feats)

    np.testing.assert_array_equal(Xb_compact, Xb[np.ix_(rows, feats)])
    np.testing.assert_array_equal(gradient_compact, gradient[selected])
    np.testing.assert_array_equal(hessian_compact, hessian[selected])
    np.testing.assert_array_equal(compact[1], full[1])
    for compact_array, full_array in zip(compact_flat, full[2], strict=False):
        np.testing.assert_array_equal(compact_array, full_array)
    assert compact_tree._mat() == full[0]._mat()

    step = 0.1
    expected_margin = rng.normal(size=len(X))
    actual_margin = expected_margin.copy()
    expected_margin[rows] += step * full[1]
    trees._add_rows_in_place(actual_margin, rows, compact[1], step)
    np.testing.assert_array_equal(actual_margin, expected_margin)

    rest_rows = np.flatnonzero(~selected)
    expected_margin[rest_rows] += step * _tree_pred_rows(full[2], X, rest_rows)
    trees._add_flat_binned_rows_in_place(compact_binned, Xb, rest_rows, actual_margin, step)
    np.testing.assert_array_equal(actual_margin, expected_margin)


def test_binary_softmax_uses_sign_mirrored_numeric_trees_in_refit():
    rng = np.random.default_rng(44)
    X = rng.normal(size=(600, 12))
    y = (X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2] * X[:, 3] > 0).astype(int)

    model = reason_boost_softmax(
        X,
        y,
        rounds=20,
        lr=0.05,
        depth=5,
        min_leaf=20,
        patience=8,
        class_weight="balanced",
        refit=True,
        seed=6,
    )
    predicted, confidence = boost_softmax_predict(model, X[:100])

    assert len(model[2]) > 0
    assert len(model[2]) % 2 == 0
    assert predicted.shape == (100,)
    assert np.isfinite(confidence).all()
    for left, right in zip(model[4][::2], model[4][1::2], strict=False):
        for left_array, right_array in zip(left[:4], right[:4], strict=False):
            np.testing.assert_array_equal(left_array, right_array)
        np.testing.assert_array_equal(left[4], -right[4])

    stage_class = [stage for stage, _tree in model[2]]
    full = trees._pack_flat_forest(model[4], stage_class)
    mirrored = trees._pack_binary_mirrored_forest(model[4], stage_class)
    assert full is not None
    assert mirrored is not None
    Xc = np.ascontiguousarray(X[:100], dtype=np.float64)
    base = np.ascontiguousarray(model[0], dtype=np.float64)
    full_scores = trees._forest_scores_flat_nb(*full, base, model[1], Xc)
    mirrored_scores = trees._forest_scores_binary_mirrored_nb(*mirrored, base, model[1], Xc)
    np.testing.assert_array_equal(mirrored_scores, full_scores)


def test_binary_softmax_auc_verifier_is_deterministic_and_keeps_proof_trees():
    rng = np.random.default_rng(822)
    X = rng.normal(size=(700, 10))
    latent = X[:, 0] + 0.8 * X[:, 1] * X[:, 2] - 0.5 * X[:, 3]
    y = (latent + rng.normal(scale=0.7, size=len(X)) > 0).astype(int)
    kwargs = dict(
        rounds=35,
        lr=0.05,
        depth=4,
        min_leaf=15,
        patience=8,
        refit=False,
        validation_metric="auc",
        seed=9,
    )

    first = reason_boost_softmax(X, y, **kwargs)
    second = reason_boost_softmax(X, y, **kwargs)

    assert len(first[2]) > 0
    assert len(first[2]) % 2 == 0
    assert [stage for stage, _tree in first[2]] == [stage for stage, _tree in second[2]]
    np.testing.assert_array_equal(
        boost_softmax_predict(first, X)[0],
        boost_softmax_predict(second, X)[0],
    )
    for left, right in zip(first[4], second[4], strict=False):
        for left_array, right_array in zip(left, right, strict=False):
            np.testing.assert_array_equal(left_array, right_array)


def test_indexed_numeric_tree_routing_matches_a_dense_row_slice():
    flat = (
        np.array([0, -1, -1], dtype=np.int64),
        np.array([0.0, 0.0, 0.0]),
        np.array([1, -1, -1], dtype=np.int64),
        np.array([2, -1, -1], dtype=np.int64),
        np.array([0.0, -0.4, 0.6]),
    )
    X = np.array([[-2.0], [0.0], [1.0], [-0.5], [4.0]])
    rows = np.array([4, 1, 3], dtype=np.int64)

    np.testing.assert_array_equal(_tree_pred_rows(flat, X, rows), _tree_pred(flat, X[rows]))


def test_fit_reservoir_is_stratified_deterministic_and_maps_holdout_rows(monkeypatch):
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    X = np.arange(len(y), dtype=float)[:, None]
    first = boost_module._fit_rows(y, 100, seed=38, stratified=True)
    second = boost_module._fit_rows(y, 100, seed=38, stratified=True)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.bincount(y[first], minlength=3), np.array([90, 9, 1]))

    captured = {}

    def stub_softmax(X_fit, y_fit, **_kwargs):
        captured["X"] = X_fit.copy()
        captured["y"] = y_fit.copy()
        return np.zeros(3), 0.1, [], [0, 1, 2], [], np.array([0, 3]), False

    monkeypatch.setattr(boost_module, "reason_boost_softmax", stub_softmax)
    model = AdditiveCertifiedClassifier(seed=5, fit_cap=100).fit(X, y)

    np.testing.assert_array_equal(captured["X"], X[model.fit_rows_])
    np.testing.assert_array_equal(captured["y"], y[model.fit_rows_])
    np.testing.assert_array_equal(model.ver_, model.fit_rows_[[0, 3]])


def test_early_parallelism_is_final_constant_leaf_only(monkeypatch):
    calls = []

    def stub_softmax(X, y, **kwargs):
        calls.append(bool(kwargs["parallel_k"]))
        classes = list(np.unique(y))
        return np.zeros(len(classes)), 0.1, [], classes, [], None, False

    monkeypatch.setattr(boost_module, "reason_boost_softmax", stub_softmax)
    monkeypatch.setattr(boost_module, "_numba_threading_is_threadsafe", lambda: True)
    X = np.zeros((boost_module._PARALLEL_K_FINAL_MIN_ROWS, 3))
    y = np.resize(np.array([0, 1]), len(X))

    AdditiveCertifiedClassifier(refit=True, linear_leaf=False).fit(X, y)
    AdditiveCertifiedClassifier(refit=True, linear_leaf=True).fit(X, y)
    AdditiveCertifiedClassifier(refit=False, linear_leaf=False).fit(X, y)
    AdditiveCertifiedClassifier(refit=True, linear_leaf=False).fit(X[:-1], y[:-1])

    assert calls == [True, False, False, False]


def test_workqueue_disables_per_class_thread_pool(monkeypatch):
    calls = []

    def stub_softmax(X, y, **kwargs):
        calls.append(bool(kwargs["parallel_k"]))
        classes = list(np.unique(y))
        return np.zeros(len(classes)), 0.1, [], classes, [], None, False

    monkeypatch.setattr(boost_module, "reason_boost_softmax", stub_softmax)
    monkeypatch.setattr(boost_module, "_numba_threading_is_threadsafe", lambda: False)
    X = np.zeros((boost_module._PARALLEL_K_MIN_ROWS, 3))
    y = np.resize(np.array([0, 1, 2]), len(X))

    AdditiveCertifiedClassifier(refit=True, linear_leaf=False).fit(X, y)

    assert calls == [False]
