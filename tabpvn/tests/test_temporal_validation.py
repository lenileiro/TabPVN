"""Strict future-validation geometry for event fitting."""

import numpy as np
import pandas as pd

import tabpvn.trees as tree_module
from tabpvn.certified_boost import AdditiveCertifiedClassifier, _temporal_fit_sample
from tabpvn.preprocessing import _Preprocessor
from tabpvn.trees import _fit_verifier_split, reason_boost_2nd, reason_boost_softmax
from tabpvn.validation import FutureValidation


def test_future_split_keeps_timestamp_ties_atomic_and_classes_covered():
    groups = np.repeat(np.arange(12, dtype=np.int64), 8)
    labels = np.tile([0, 1], len(groups) // 2)
    shuffled = np.random.default_rng(4).permutation(len(groups))
    validation = FutureValidation(groups[shuffled])

    train, valid = validation.split(
        labels[shuffled],
        holdout=0.25,
        min_train=20,
        min_valid=12,
        require_class_coverage=True,
    )

    assert validation.groups[train].max() < validation.groups[valid].min()
    assert set(labels[shuffled][train]) == {0, 1}
    assert set(labels[shuffled][valid]) == {0, 1}


def test_expanding_splits_use_only_past_rows_and_keep_ties_atomic():
    groups = np.repeat(np.arange(20, dtype=np.int64), 6)
    labels = np.tile([0, 1, 0, 1, 0, 1], 20)
    shuffled = np.random.default_rng(14).permutation(len(groups))
    validation = FutureValidation(groups[shuffled])

    splits = validation.expanding_splits(
        labels[shuffled],
        folds=3,
        warmup=0.4,
        min_train=24,
        min_valid=12,
        require_train_class_coverage=True,
    )

    assert len(splits) == 3
    previous_valid = np.empty(0, dtype=np.int64)
    for train, valid in splits:
        assert validation.groups[train].max() < validation.groups[valid].min()
        assert not np.intersect1d(train, valid).size
        assert set(previous_valid).issubset(set(train))
        previous_valid = np.concatenate([previous_valid, valid])


def test_bounded_rows_respects_cap_without_cutting_the_first_retained_tie():
    groups = np.repeat(np.arange(10, dtype=np.int64), 10)
    rows = FutureValidation(groups).bounded_rows(25)

    assert len(rows) <= 25
    assert np.array_equal(rows, np.arange(80, 100))
    assert groups[rows[0] - 1] != groups[rows[0]]


def test_bounded_rows_samples_across_recent_groups_when_one_batch_exceeds_cap():
    groups = np.repeat(np.arange(6, dtype=np.int64), 1_000)
    validation = FutureValidation(groups)

    rows = validation.bounded_rows(120)
    train, valid = validation.take(rows).split(
        holdout=0.25,
        min_train=40,
        min_valid=20,
    )

    assert len(rows) <= 120
    assert np.unique(groups[rows]).size >= 2
    assert groups[rows[train]].max() < groups[rows[valid]].min()


def test_temporal_multiscale_reservoir_keeps_recent_and_logarithmic_history():
    groups = np.repeat(np.arange(100, dtype=np.int64), 10)
    target = np.linspace(-1.0, 1.0, len(groups))

    rows, weight, report = _temporal_fit_sample(
        target,
        groups,
        cap=200,
        seed=9,
        stratified=False,
    )
    repeated_rows, _weight, _report = _temporal_fit_sample(
        target,
        groups,
        cap=200,
        seed=9,
        stratified=False,
    )

    np.testing.assert_array_equal(rows, repeated_rows)
    assert weight is None
    assert len(rows) == 200
    assert set(np.arange(850, 1_000)).issubset(set(rows))
    assert np.count_nonzero(rows < 850) == 50
    assert np.all(np.diff(groups[rows]) >= 0)
    assert report["mode"] == "temporal_multiscale_reservoir"
    assert report["recent_share"] == 0.75
    assert len(report["history_band_rows"]) == 6
    assert sum(report["history_band_samples"]) == 50
    assert all(
        sample <= source
        for sample, source in zip(
            report["history_band_samples"],
            report["history_band_rows"],
            strict=True,
        )
    )


def test_temporal_multiscale_reservoir_preserves_recurring_age_bands():
    rows = 60_000
    cap = 12_000
    groups = np.arange(rows, dtype=np.int64)
    recurring_regime = np.zeros(rows, dtype=bool)
    recurring_regime[-5_000:] = True
    recurring_regime[43_500:51_000] = True

    selected, _weight, report = _temporal_fit_sample(
        recurring_regime.astype(float),
        groups,
        cap=cap,
        seed=9,
        stratified=False,
    )

    assert report["history_band_rows"] == (500, 1_000, 2_000, 4_000, 8_000, 35_500)
    assert report["history_band_samples"] == (500, 500, 500, 500, 500, 500)
    assert np.isclose(recurring_regime[selected].mean(), 7 / 12)


def test_temporal_multiscale_reservoir_enforces_rare_floor_and_temporal_prior():
    groups = np.repeat(np.arange(100, dtype=np.int64), 10)
    target = np.zeros(len(groups), dtype=np.int64)
    target[:30] = 1

    baseline_rows, _baseline_weight, _baseline_report = _temporal_fit_sample(
        target,
        groups,
        cap=200,
        seed=7,
        stratified=True,
    )
    rows, weight, report = _temporal_fit_sample(
        target,
        groups,
        cap=200,
        seed=7,
        stratified=True,
        min_class_rows=20,
    )

    assert np.count_nonzero(target[rows] == 1) >= 20
    assert weight is not None
    assert np.isclose(
        np.average(target[rows] == 1, weights=weight),
        np.mean(target[baseline_rows] == 1),
    )
    assert report["class_floor"] == 20
    assert report["prior_reference"] == "temporal_reservoir"


def test_temporal_classifier_cap_maps_hybrid_sample_and_future_verifier():
    groups = np.repeat(np.arange(100, dtype=np.int64), 10)
    target = np.tile([0, 1], len(groups) // 2)
    features = np.arange(len(groups), dtype=float)[:, None]

    model = AdditiveCertifiedClassifier(
        rounds=0,
        refit=False,
        fit_cap=200,
        holdout=0.2,
        seed=4,
    ).fit(features, target, validation_groups=groups)

    selected = np.asarray(model.fit_rows_, dtype=np.int64)
    verifier = np.asarray(model.ver_, dtype=np.int64)
    fitted = np.setdiff1d(selected, verifier)
    assert model.fit_sampling_["mode"] == "temporal_multiscale_reservoir"
    assert set(np.arange(850, 1_000)).issubset(set(selected))
    assert groups[fitted].max() < groups[verifier].min()


def test_temporal_rare_verifier_keeps_future_prevalence_weights():
    groups = np.repeat(np.arange(100, dtype=np.int64), 10)
    target = np.zeros(len(groups), dtype=np.int64)
    target[np.arange(0, 850, 100)] = 1
    for group in range(85, 100):
        target[group * 10 : group * 10 + 2] = 1
    features = np.arange(len(groups), dtype=float)[:, None]

    model = AdditiveCertifiedClassifier(
        rounds=0,
        refit=False,
        fit_cap=200,
        holdout=0.2,
        seed=5,
        rare_event=True,
        rare_min_events=20,
        min_verifier_events=5,
    ).fit(features, target, validation_groups=groups)

    assert model.fit_sample_weight_ is not None
    assert model.ver_weight_ is not None
    selected = np.asarray(model.fit_rows_, dtype=np.int64)
    verifier = np.asarray(model.ver_, dtype=np.int64)
    selected_positions = np.searchsorted(selected, verifier)
    np.testing.assert_array_equal(
        model.ver_weight_,
        model.fit_sample_weight_[selected_positions],
    )


def test_causal_target_encoding_does_not_read_same_timestamp_labels():
    frame = pd.DataFrame({"category": ["a", "b", "a", "a", "b", "b"]})
    target = np.array([1, 0, 0, 1, 1, 0])
    groups = np.repeat(np.arange(3, dtype=np.int64), 2)
    preprocessor = _Preprocessor(task="classification").fit(frame, target)
    keys = preprocessor._category_keys(frame["category"])

    encoded = preprocessor._causal_target_values(keys, target, groups)[:, 0]

    np.testing.assert_allclose(encoded[:2], 0.5)
    # Both 'a' rows at timestamp 1 see only timestamp 0, not each other.
    np.testing.assert_allclose(encoded[2:4], encoded[2])
    assert encoded[2] > 0.5


def test_certified_verifier_uses_strictly_future_timestamp_groups():
    groups = np.repeat(np.arange(20, dtype=np.int64), 6)
    labels = np.tile([0, 1, 0, 1, 0, 1], 20)

    fit, verifier = _fit_verifier_split(
        labels,
        holdout=0.2,
        seed=7,
        stratified=True,
        validation_groups=groups,
    )

    assert groups[fit].max() < groups[verifier].min()
    assert set(labels[fit]) == {0, 1}
    assert set(labels[verifier]) == {0, 1}


def test_temporal_rare_verifier_expands_until_event_floor_is_met():
    groups = np.repeat(np.arange(120, dtype=np.int64), 5)
    labels = np.zeros(len(groups), dtype=np.int64)
    labels[np.arange(10, 120, 10) * 5] = 1

    fit, verifier = _fit_verifier_split(
        labels,
        holdout=0.1,
        seed=7,
        stratified=True,
        min_verifier_events=3,
        validation_groups=groups,
    )

    assert groups[fit].max() < groups[verifier].min()
    assert np.count_nonzero(labels[verifier] == 1) >= 3
    assert set(labels[fit]) == {0, 1}


def test_temporal_regression_cap_keeps_weights_aligned(monkeypatch):
    rows = 240
    groups = np.repeat(np.arange(24, dtype=np.int64), 10)
    X = np.column_stack([np.linspace(-2.0, 2.0, rows), np.sin(np.arange(rows))])
    y = 4.0 * X[:, 0] + 0.1 * X[:, 1]
    weight = np.linspace(0.5, 2.0, rows)
    selected = FutureValidation(groups).bounded_rows(120)
    captured = {}

    def capture_refit(*args, **kwargs):
        captured["X"] = args[0].copy()
        captured["weight"] = np.asarray(args[15]).copy()
        return float(np.mean(args[1])), [], []

    monkeypatch.setattr(tree_module, "_boost_2nd_run", capture_refit)
    reason_boost_2nd(
        X,
        y,
        n_terms=8,
        depth=2,
        min_leaf=5,
        patience=4,
        fit_cap=120,
        w=weight,
        validation_groups=groups,
        seed=3,
    )

    np.testing.assert_array_equal(captured["X"], X[selected])
    np.testing.assert_array_equal(captured["weight"], weight[selected])


def test_temporal_classification_cap_keeps_weights_aligned(monkeypatch):
    rows = 240
    groups = np.repeat(np.arange(24, dtype=np.int64), 10)
    X = np.column_stack([np.linspace(-2.0, 2.0, rows), np.sin(np.arange(rows))])
    y = np.tile([0, 1], rows // 2)
    weight = np.linspace(0.5, 2.0, rows)
    selected = FutureValidation(groups).bounded_rows(120)
    captured = {}

    def capture_binary(X_fit, _yi, classes, *args):
        captured["X"] = X_fit.copy()
        captured["weight"] = np.asarray(args[13]).copy()
        return np.zeros(2), 0.1, [], classes, [], np.array([0]), False

    monkeypatch.setattr(tree_module, "_reason_boost_binary_symmetric", capture_binary)
    reason_boost_softmax(
        X,
        y,
        rounds=0,
        fit_cap=120,
        sample_weight=weight,
        validation_groups=groups,
        seed=3,
    )

    np.testing.assert_array_equal(captured["X"], X[selected])
    np.testing.assert_array_equal(captured["weight"], weight[selected])
