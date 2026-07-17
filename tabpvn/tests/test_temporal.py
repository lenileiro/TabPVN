"""Causal temporal evidence and native datetime schema coverage."""

import numpy as np
import pandas as pd
import pytest

from tabpvn import TemporalLaplaceMap
from tabpvn.preprocessing import _onehot_groups, _Preprocessor


def _events(entity, hours):
    return pd.DataFrame(
        {
            "entity": entity,
            "time": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h"),
        }
    )


def test_laplace_history_is_strictly_causal_and_batches_equal_timestamps():
    events = _events(["a", "a", "a", "a", "b", "b"], [0, 1, 1, 3, 0, 8])
    temporal = TemporalLaplaceMap(entity="entity", timestamp="time")

    evidence = temporal.fit_transform(events)

    assert evidence.shape[1] <= 32
    np.testing.assert_array_equal(evidence[[0, 4]], 0.0)
    scale = temporal.scales_seconds_[0]
    expected_at_one_hour = np.exp(-3_600.0 / scale)
    np.testing.assert_allclose(evidence[1:3, 0], expected_at_one_hour)
    expected_at_three_hours = expected_at_one_hour * np.exp(-7_200.0 / scale) + 2.0 * np.exp(-7_200.0 / scale)
    np.testing.assert_allclose(evidence[3, 0], expected_at_three_hours)
    assert temporal.report_["same_timestamp_policy"] == "emit_then_update_batch"


def test_laplace_transform_starts_from_fitted_state_without_mutating_it():
    train = _events(["a", "a", "a", "b", "b"], [0, 2, 6, 0, 4])
    query = _events(["a", "a", "new", "new"], [8, 10, 1, 3])
    temporal = TemporalLaplaceMap(entity="entity", timestamp="time").fit(train)

    first = temporal.transform(query)
    second = temporal.transform(query)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[2], 0.0)
    states = slice(0, len(temporal.scales_seconds_))
    assert np.all(first[0, states] > 0.0)
    assert np.all(first[1, states] > first[0, states] * np.exp(-7_200.0 / temporal.scales_seconds_))


def test_laplace_training_facts_are_invariant_to_input_row_order():
    events = _events(["a", "a", "a", "b", "b", "b"], [0, 2, 7, 1, 4, 9]).assign(row=np.arange(6))
    shuffled = events.sample(frac=1.0, random_state=17).reset_index(drop=True)

    ordered_map = TemporalLaplaceMap("entity", "time")
    shuffled_map = TemporalLaplaceMap("entity", "time")
    ordered = ordered_map.fit_transform(events)
    permuted = shuffled_map.fit_transform(shuffled)
    restored = permuted[np.argsort(shuffled["row"].to_numpy())]

    np.testing.assert_allclose(ordered, restored)
    np.testing.assert_allclose(ordered_map.scales_seconds_, shuffled_map.scales_seconds_)


def test_laplace_transform_rejects_missing_semantics_and_history_overlap():
    train = _events(["a", "a", "a"], [0, 2, 4])
    temporal = TemporalLaplaceMap(entity="entity", timestamp="time").fit(train)

    with pytest.raises(ValueError, match="strictly later"):
        temporal.transform(_events(["a"], [4]))
    with pytest.raises(ValueError, match="missing semantic columns"):
        temporal.transform(pd.DataFrame({"entity": ["a"], "other": [1]}))
    with pytest.raises(TypeError, match="ambiguous units"):
        TemporalLaplaceMap("entity", "time").fit(pd.DataFrame({"entity": ["a", "a"], "time": [0, 1]}))


def test_laplace_frame_augmentation_is_named_and_non_mutating():
    train = _events(["a", "a", "a", "b", "b"], [0, 2, 6, 0, 4])
    original = train.copy(deep=True)
    temporal = TemporalLaplaceMap(entity="entity", timestamp="time")

    augmented = temporal.fit_augment(train)

    pd.testing.assert_frame_equal(train, original)
    assert list(augmented.columns[:2]) == ["entity", "time"]
    assert list(augmented.columns[2:]) == list(temporal.get_feature_names_out())
    assert all(item["causal_scope"] == "strictly_prior_timestamp" for item in temporal.feature_metadata_)


def test_marked_values_and_adjacent_scale_bands_replay_exactly():
    events = _events(["a", "a", "a", "a", "b", "b"], [0, 1, 3, 8, 0, 5]).assign(
        amount=[10.0, 20.0, 40.0, 5.0, 30.0, 60.0]
    )
    temporal = TemporalLaplaceMap(
        entity="entity",
        timestamp="time",
        value_columns=["amount"],
    )

    evidence = temporal.fit_transform(events)

    scale_count = len(temporal.scales_seconds_)
    channel_width = 2 * scale_count - 1
    amount_scale = temporal.value_scales_["amount"]
    expected_short = (10.0 / amount_scale) * np.exp(-3_600.0 / temporal.scales_seconds_[0])
    expected_long = (10.0 / amount_scale) * np.exp(-3_600.0 / temporal.scales_seconds_[1])
    np.testing.assert_allclose(evidence[1, channel_width], expected_short)
    np.testing.assert_allclose(
        evidence[1, channel_width + scale_count],
        expected_long - expected_short,
    )
    assert evidence.shape[1] <= 32
    assert temporal.report_["channels"] == ["event_count", "amount"]
    assert {item["kind"] for item in temporal.feature_metadata_} >= {
        "decayed_event_count",
        "event_count_age_band",
        "decayed_value_sum",
        "value_age_band",
    }


def test_marked_value_schema_is_bounded_and_numeric():
    with pytest.raises(TypeError, match="not a string"):
        TemporalLaplaceMap("entity", "time", value_columns="amount")
    with pytest.raises(ValueError, match="at most 2"):
        TemporalLaplaceMap("entity", "time", value_columns=["a", "b", "c"])
    events = _events(["a", "a", "a"], [0, 1, 2]).assign(mark=["low", "mid", "high"])
    with pytest.raises(TypeError, match="numeric dtype"):
        TemporalLaplaceMap("entity", "time", value_columns=["mark"]).fit(events)


def test_preprocessor_compiles_native_datetime_into_finite_numeric_facts():
    frame = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "when": pd.to_datetime(
                ["2025-01-01 00:00Z", "2025-01-02 06:00Z", None, "2025-02-03 18:00Z"],
                utc=True,
            ),
            "kind": ["a", "b", "a", "b"],
        }
    )
    before = frame.copy(deep=True)
    preprocessor = _Preprocessor(target_encoding=False).fit(frame, np.array([0, 1, 0, 1]))

    encoded = preprocessor.transform(frame)

    pd.testing.assert_frame_equal(frame, before)
    assert preprocessor.datetime_cols == ["when"]
    assert "when" not in preprocessor.cat_cols
    assert any(name == "when__datetime_elapsed_days" for name in preprocessor.names)
    assert any(name == "when__datetime_isna" for name in preprocessor.names)
    assert encoded.shape[1] == len(preprocessor.names)
    assert np.isfinite(encoded).all()
    category_start = len(preprocessor.num_cols) + sum(
        feature.n_features_out_ for feature in preprocessor.datetime_feat.values()
    )
    assert _onehot_groups(preprocessor) == ((category_start, category_start + 1),)


def test_preprocessor_does_not_guess_datetime_semantics_from_strings():
    frame = pd.DataFrame({"when": ["2025-01-01", "2025-01-02", "2025-01-03"]})
    preprocessor = _Preprocessor(target_encoding=False).fit(frame, np.array([0, 1, 0]))

    assert preprocessor.datetime_cols == []
    assert preprocessor.cat_cols == ["when"]
