"""Leak-safe high-cardinality categorical encoding in the TabPVN default."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tabpvn import TabPVN
from tabpvn.base import _Preprocessor


def test_unique_category_ids_do_not_leak_their_training_label():
    X = pd.DataFrame({"customer": [f"id-{i}" for i in range(30)]})
    y = np.array([0, 1] * 15)
    prep = _Preprocessor()

    encoded = prep.fit_transform(X, y)

    assert "customer" not in prep.target_enabled
    assert "customer" not in prep.target_indices
    assert encoded.shape[1] == 1
    assert np.allclose(prep.transform(pd.DataFrame({"customer": ["new-id"]})), 0.0)


def test_structured_identifiers_remain_exact_categories_without_component_tokens():
    identifiers = [
        f"WC-{year}_{team}"
        for year in (2002, 2006, 2010, 2014, 2018)
        for team in ("ARG", "BRA", "FRA", "JPN")
    ]
    frame = pd.DataFrame({"ID": identifiers})
    target = np.tile([4.0, 5.0, 3.0, 1.0], 5)
    prep = _Preprocessor(task="regression")

    prep.fit_transform(frame, target)
    query = prep.transform(pd.DataFrame({"ID": ["WC-2026_ARG", "WC-2026_NEW"]}))

    assert prep.byte_cols == []
    assert prep.cat_cols == ["ID"]
    assert len(prep.names) == len(identifiers)
    assert all(name.startswith("ID=") for name in prep.names)
    assert not any("token" in name or "~" in name for name in prep.names)
    np.testing.assert_array_equal(query, np.zeros((2, len(identifiers))))


def test_repeated_categories_keep_only_out_of_fold_target_signal():
    groups = np.repeat([f"segment-{i}" for i in range(24)], 12)
    X = pd.DataFrame({"segment": groups})
    y = np.repeat([i % 2 for i in range(24)], 12)
    prep = _Preprocessor()

    encoded = prep.fit_transform(X, y)
    target_col = prep.target_indices["segment"]

    assert prep.target_enabled["segment"]
    assert np.corrcoef(encoded[:, target_col[0]], y)[0, 1] > 0.9


def test_tabpvn_uses_target_encoding_for_repeated_categorical_signal():
    groups = np.repeat([f"segment-{i}" for i in range(24)], 20)
    X = pd.DataFrame({"segment": groups, "noise": np.tile(np.arange(20), 24)})
    y = np.repeat([i % 2 for i in range(24)], 20)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    model = TabPVN(seed=7).fit(Xtr, ytr)

    assert model._prep.target_enabled["segment"]
    assert model.target_encoding_selected_
    assert (model.predict(Xte) == yte).mean() > 0.95
    assert model.certify(Xte.iloc[:20]) == 1.0


def test_gaussian_regression_statistics_match_conjugate_update():
    keys = np.array(["a", "a", "b"], dtype=object)
    target = np.array([1.0, 3.0, 10.0])

    stats = _Preprocessor._target_stats(
        keys,
        target,
        is_classification=False,
        classes=None,
        smooth=2.0,
        gaussian=True,
    )

    prior = target.mean()
    variance = target.var(ddof=1)
    within_a = np.sum((target[:2] - target[:2].mean()) ** 2)
    local_variance_a = (within_a + 2.0 * variance) / 3.0
    prior_precision = 2.0 / variance
    data_precision_a = 2.0 / local_variance_a
    expected_a = (prior_precision * prior + data_precision_a * target[:2].mean()) / (
        prior_precision + data_precision_a
    )
    np.testing.assert_allclose(stats["table"]["a"], [expected_a])
    np.testing.assert_allclose(stats["prior"], [prior])
    np.testing.assert_allclose(
        _Preprocessor._apply_target_stats(np.array(["unseen"], dtype=object), stats),
        [[prior]],
    )


def test_gaussian_regression_features_are_oof_and_have_stable_inference_width():
    categories = np.repeat([f"segment-{index}" for index in range(24)], 12)
    target = np.repeat(np.arange(24, dtype=float) * 5.0, 12) + np.tile(
        np.linspace(-0.5, 0.5, 12),
        24,
    )
    frame = pd.DataFrame({"segment": categories})
    prep = _Preprocessor(task="regression", gaussian_target_statistics=True)

    encoded = prep.fit_transform(frame, target)
    indices = prep.target_indices["segment"]

    assert encoded.shape == (len(frame), 2)
    assert len(indices) == 1
    assert [prep.names[index] for index in indices] == [
        "segment__target=gaussian_mean",
    ]
    assert np.corrcoef(encoded[:, indices[0]], target)[0, 1] > 0.9
    unseen = prep.transform(pd.DataFrame({"segment": ["new-segment"]}))
    np.testing.assert_allclose(unseen[0, indices], [target.mean()])


def test_causal_gaussian_statistics_keep_equal_timestamps_atomic():
    frame = pd.DataFrame({"segment": ["a", "b", "a", "a"]})
    target = np.array([2.0, 6.0, 100.0, -100.0])
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    prep = _Preprocessor(task="regression", gaussian_target_statistics=True).fit(frame, target)
    keys = prep._category_keys(frame["segment"])

    encoded = prep._causal_target_values(keys, target, groups)

    np.testing.assert_allclose(encoded[:2], 0.0)
    np.testing.assert_allclose(encoded[2], encoded[3])
    np.testing.assert_allclose(encoded[2, 0], (prep._target_smoothing * 4.0 + 2.0) / 6.0)
