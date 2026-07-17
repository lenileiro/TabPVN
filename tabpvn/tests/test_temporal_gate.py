"""Future-holdout selection and estimator lifecycle for event evidence."""

import pickle

import numpy as np
import pandas as pd
import pytest

from tabpvn import TabPVN
from tabpvn.event_schema import bounded_event_gate, discover_event_schemas
from tabpvn.proposers import TemporalEvidenceChallenger
from tabpvn.temporal import TemporalLaplaceMap


def _stream(seed=31, entities=24, events_per_entity=42):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []
    labels = []
    regression = []
    for entity in range(entities):
        clock = 0.0
        count_state = 0.0
        value_state = 0.0
        for step in range(events_per_entity):
            gap = float(rng.choice([0.25, 0.5, 1.0, 5.0, 12.0], p=[0.2] * 5))
            amount = float(rng.lognormal(3.0, 0.45))
            clock += gap
            count_state *= np.exp(-gap / 3.0)
            value_state *= np.exp(-gap / 6.0)
            rows.append((entity, base + pd.to_timedelta(clock, unit="h"), amount, step))
            labels.append(int(count_state > 1.35))
            regression.append(value_state)
            count_state += 1.0
            value_state += amount / 20.0
    frame = pd.DataFrame(rows, columns=["entity", "time", "amount", "step"])
    return frame, np.asarray(labels), np.asarray(regression)


def test_future_gate_selects_temporal_evidence_for_classification_and_regression():
    events, labels, regression = _stream()
    features = events.drop(columns="step")
    gate = TemporalEvidenceChallenger(seed=7)

    classification = gate.evaluate(
        features,
        labels,
        entity="entity",
        timestamp="time",
        value_columns=["amount"],
        task="classification",
    )
    continuous = gate.evaluate(
        features,
        regression,
        entity="entity",
        timestamp="time",
        value_columns=["amount"],
        task="regression",
    )

    assert classification["selected"] is True
    assert classification["candidate_score"] > classification["baseline_score"] + 0.1
    assert classification["power_aware"] is True
    assert classification["forward_windows"] >= 2
    assert classification["forward_consistency_passed"] is True
    assert classification["confidence_method"] == "delete_one_time_block_jackknife"
    assert classification["confidence_samples"] > 0
    assert continuous["selected"] is True
    assert continuous["relative_gain"] > 0.1
    assert continuous["confidence_lower_relative_gain"] > 0.0
    assert classification["features"] <= 32


def test_future_gate_rejects_history_when_current_row_already_determines_target():
    rng = np.random.default_rng(83)
    rows = 800
    events = pd.DataFrame(
        {
            "entity": np.repeat(np.arange(20), 40),
            "time": pd.Timestamp("2025-01-01", tz="UTC")
            + pd.to_timedelta(np.tile(np.arange(40), 20), unit="h"),
            "amount": rng.normal(size=rows),
        }
    )
    labels = (events["amount"].to_numpy() > 0.0).astype(int)

    report = TemporalEvidenceChallenger(seed=5).evaluate(
        events,
        labels,
        entity="entity",
        timestamp="time",
        value_columns=["amount"],
        task="classification",
    )

    assert report["selected"] is False
    assert report["reason"] == "future_holdout_gain_below_gate"
    assert report["candidate_score"] < report["baseline_score"] + report["minimum_gain"]


def test_gate_cap_samples_complete_histories_across_many_entities():
    entity_count = 300
    steps = 100
    events = pd.DataFrame(
        {
            "entity": np.repeat(np.arange(entity_count), steps),
            "time": pd.Timestamp("2025-01-01", tz="UTC")
            + pd.to_timedelta(np.tile(np.arange(steps), entity_count), unit="h"),
        }
    )
    probe = TemporalLaplaceMap("entity", "time")
    frame, entity_codes, _, timestamps_ns = probe._extract(events)

    split = TemporalEvidenceChallenger(seed=3)._window_split(
        frame,
        entity_codes,
        timestamps_ns,
    )

    assert split is not None
    train, valid = split
    selected = np.r_[train, valid]
    assert len(selected) == 20_000
    assert len(np.unique(entity_codes[selected])) == 200
    assert timestamps_ns[train].max() < timestamps_ns[valid].min()


def test_power_gate_uses_representative_rows_with_a_history_warmup():
    rows = 60_000
    events = pd.DataFrame(
        {
            "entity": np.arange(rows) % 1_000,
            "time": pd.date_range("2025-01-01", periods=rows, freq="s", tz="UTC"),
        }
    )
    probe = TemporalLaplaceMap("entity", "time")
    frame, entity_codes, _, timestamps_ns = probe._extract(events)

    split = TemporalEvidenceChallenger(seed=3)._power_window_split(
        frame,
        entity_codes,
        timestamps_ns,
    )

    assert split is not None
    warmup, train, windows = split
    valid = np.concatenate(windows)
    assert len(warmup) == 10_000
    assert len(train) == 35_000
    assert len(valid) == 15_000
    assert timestamps_ns[warmup].max() < timestamps_ns[train].min()
    assert timestamps_ns[train].max() < timestamps_ns[valid].min()


def test_automatic_gate_sampling_keeps_a_dense_atomic_recent_stream():
    rows = 100_005
    timestamps = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        np.arange(rows) // 10,
        unit="s",
    )
    events = pd.DataFrame({"entity": np.arange(rows) % 1_000, "time": timestamps})
    target = np.arange(rows)

    bounded, bounded_target = bounded_event_gate(events, target, timestamp="time")

    assert len(bounded) == 99_995
    assert bounded.index[0] == 10
    assert bounded_target[0] == 10
    assert bounded["time"].is_monotonic_increasing
    assert events.loc[:9, "time"].max() < bounded["time"].min()


def test_automatic_event_schema_discovers_named_text_timestamps_without_labels():
    events, _, _ = _stream(entities=12, events_per_entity=25)
    features = events.drop(columns="step").rename(columns={"time": "event_timestamp"})
    features["event_timestamp"] = features["event_timestamp"].astype(str)

    candidates = discover_event_schemas(features)

    assert 1 <= len(candidates) <= 3
    assert candidates[0].entity == "entity"
    assert candidates[0].timestamp == "event_timestamp"
    assert candidates[0].timestamp_source == "named_parseable_text"
    assert any("amount" in candidate.value_columns for candidate in candidates)


def test_automatic_event_schema_prefers_ip_and_does_not_accumulate_integer_ids():
    rows = 300
    features = pd.DataFrame(
        {
            "ip": np.repeat(np.arange(30), 10),
            "device": np.tile(np.arange(5), 60),
            "app": np.arange(rows) % 17,
            "click_time": pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC"),
        }
    )

    candidates = discover_event_schemas(features)

    assert candidates[0].entity == "ip"
    assert all(candidate.value_columns == () for candidate in candidates)


def test_event_aware_fit_deploys_selected_map_and_predicts_from_original_schema():
    events, labels, _ = _stream(entities=20, events_per_entity=40)
    train = events["step"] < 30
    train_events = events.loc[train].drop(columns="step").reset_index(drop=True)
    future_events = events.loc[~train].drop(columns="step").reset_index(drop=True)
    model = TabPVN(
        seed=7,
        boost={"rounds": 30, "depth": 3, "leaf": 6, "patience": 8, "refit": False},
    )

    model.fit(
        train_events,
        labels[train],
        entity="entity",
        timestamp="time",
        value_columns=["amount"],
    )
    probability = model.predict_proba(future_events)

    assert model.temporal_selected_ is True
    assert model.validation_report_["mode"] == "strict_future_holdout"
    assert model.validation_report_["same_timestamp_rows_are_atomic"] is True
    assert model.validation_report_["fit_sampling"]["mode"] == "temporal_full"
    assert model.validation_report_["threshold_validation"] == "prequential_future"
    assert model.validation_report_["threshold_validation_rows"] > 0
    assert model._fit_validation is None
    assert model.temporal_evidence_report_[0]["deployed_features"] <= 32
    assert "entity" not in model._prep.input_cols
    assert any("history_laplace" in str(column) for column in model._prep.input_cols)
    assert tuple(model.feature_names_in_) == tuple(train_events.columns)
    assert probability.shape == (len(future_events), 2)
    assert model.certify(future_events.iloc[:20]) == 1.0
    verifier = np.asarray(model._pred.ver_, dtype=int)
    sorted_times = train_events.sort_values("time", kind="stable")["time"].astype("int64").to_numpy()
    fit_rows = np.setdiff1d(np.arange(len(sorted_times)), verifier)
    assert sorted_times[fit_rows].max() < sorted_times[verifier].min()
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(restored.predict_proba(future_events), probability)


def test_fit_automatically_discovers_and_selects_event_semantics():
    events, labels, _ = _stream(entities=20, events_per_entity=40)
    train = events["step"] < 30
    train_events = events.loc[train].drop(columns="step").reset_index(drop=True)
    future_events = events.loc[~train].drop(columns="step").reset_index(drop=True)
    model = TabPVN(
        seed=7,
        boost={"rounds": 30, "depth": 3, "leaf": 6, "patience": 8, "refit": False},
    ).fit(train_events, labels[train])

    assert model.temporal_selected_ is True
    assert model.event_schema_["selection"] == "automatic"
    assert model.event_schema_["entity"] == "entity"
    assert model.event_schema_["timestamp"] == "time"
    assert model.event_schema_["drop_entity"] is False
    assert model.validation_report_["mode"] == "strict_future_holdout"
    assert sum(report["selected"] for report in model.event_discovery_report_) == 1
    assert "entity" in model._prep.input_cols
    assert model.predict_proba(future_events).shape == (len(future_events), 2)


def test_fit_rejects_unearned_automatic_event_schema():
    rows = 240
    features = pd.DataFrame(
        {
            "event_id": np.arange(rows),
            "event_time": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "measurement": np.linspace(0.1, 1.9, rows),
        }
    )
    labels = np.arange(rows) % 2
    model = TabPVN(boost={"rounds": 4, "depth": 2, "leaf": 4, "patience": 2, "refit": False}).fit(
        features, labels
    )

    assert model._event_mode is False
    assert model.event_schema_ is None
    assert model.validation_report_["mode"] == "exchangeable"
    assert model.event_discovery_report_[0]["selected"] is False
    assert model.event_discovery_report_[0]["reason"] == "no_structurally_eligible_event_schema"


def test_ordinary_fit_clears_event_replay_state(monkeypatch):
    events, labels, _ = _stream(entities=8, events_per_entity=30)
    features = events.drop(columns="step")

    def reject(*_args, **_kwargs):
        return {
            "name": "temporal_laplace_evidence",
            "selected": False,
            "stage": "schema",
            "reason": "test_rejection",
        }

    monkeypatch.setattr(TemporalEvidenceChallenger, "evaluate", reject)
    model = TabPVN(boost={"rounds": 8, "depth": 2, "leaf": 4, "patience": 3, "refit": False})
    model.fit(features, labels, entity="entity", timestamp="time")

    assert model.temporal_selected_ is False
    assert model._temporal_map is None
    assert "entity" not in model._prep.input_cols

    numeric = np.arange(40, dtype=float).reshape(20, 2)
    model.fit(numeric, np.array([0, 1] * 10))
    assert model._event_mode is False
    assert model.event_schema_ is None


def test_event_aware_fit_requires_complete_schema_semantics():
    events, labels, _ = _stream(entities=4, events_per_entity=10)
    features = events.drop(columns="step")
    model = TabPVN(boost={"rounds": 4, "depth": 2, "leaf": 4, "patience": 2, "refit": False})

    assert not hasattr(TabPVN, "fit_events")
    with pytest.raises(ValueError, match="both entity= and timestamp="):
        model.fit(features, labels, entity="entity")
    with pytest.raises(ValueError, match="both entity= and timestamp="):
        model.fit(features, labels, timestamp="time")
    with pytest.raises(ValueError, match="requires y"):
        model.fit(features, entity="entity", timestamp="time")
