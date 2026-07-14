"""Protocol invariants for the TabPVN-versus-TabPFN research harness."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from benchmark.datasets import Dataset, _encode_target
from benchmark.experiments import models, run
from tabpvn import TabPVN


def test_binary_benchmark_can_score_average_precision_for_rare_events():
    probability = np.array([0.9, 0.8, 0.7, 0.1])
    y = np.array([0, 0, 1, 1])

    class Estimator:
        def predict_proba(self, _X):
            return np.column_stack([1.0 - probability, probability])

    score, _seconds = run._predict_and_score(
        "classification",
        Estimator(),
        np.zeros((len(y), 1)),
        y,
        classification_metric="average_precision",
    )

    assert score == average_precision_score(y, probability)


def test_evaluate_reuses_declared_task_splits():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.array([0, 1] * 6)
    declared = (
        (np.arange(0, 8), np.arange(8, 12)),
        (np.r_[0:4, 8:12], np.arange(4, 8)),
    )
    ds = Dataset("declared", X, y, "classification", declared)

    results, _, observations = run.evaluate(["linear"], [ds], splits=0, seed=99)

    assert results["declared"]["linear"] is not None
    assert set(observations["declared"]["linear"]) == {0, 1}


def test_evaluate_selects_original_official_fold_identity():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.array([0, 1] * 6)
    declared = (
        (np.arange(0, 8), np.arange(8, 12)),
        (np.r_[0:4, 8:12], np.arange(4, 8)),
    )
    ds = Dataset("declared", X, y, "classification", declared)

    _, _, observations = run.evaluate(["linear"], [ds], splits=1, seed=99, fold_indices=(1,))

    assert set(observations["declared"]["linear"]) == {1}


def test_paired_deltas_average_repeats_within_each_task():
    observations = {
        "easy": {"tabpvn": {0: 0.8, 1: 0.9}, "tabpfn": {0: 0.7, 1: 0.8}},
        "hard": {"tabpvn": {0: 0.5}, "tabpfn": {0: 0.6}},
    }

    result = run.paired_deltas(observations, ["tabpvn", "tabpfn"], "tabpfn")["tabpvn"]

    # Task deltas are +0.10 and -0.10; repeated folds do not get extra task weight.
    assert result["tasks"] == 2
    assert result["wins"] == 1
    assert result["delta"] == pytest.approx(0.0)


def test_relative_rmse_deltas_are_scale_normalized_per_task():
    observations = {
        "large_scale": {"tabpvn": {0: -2.0}, "hgb": {0: -4.0}},
        "small_scale": {"tabpvn": {0: -3.0}, "hgb": {0: -2.0}},
    }

    result = run.paired_relative_rmse_deltas(observations, ["tabpvn", "hgb"], "hgb")["tabpvn"]

    assert result["tasks"] == 2
    assert result["wins"] == 1
    assert result["delta"] == pytest.approx(0.0)


def test_evaluate_emits_fold_record_and_can_resume_without_refitting(monkeypatch):
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.array([0, 1] * 6)
    split = ((np.arange(0, 8), np.arange(8, 12)),)
    dataset = Dataset("checkpointed", X, y, "classification", split)

    class Estimator:
        def fit(self, _X, _y):
            return self

        def predict_proba(self, features):
            probability = 1.0 / (1.0 + np.exp(-features[:, 0]))
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(models, "build", lambda _name, _task: Estimator())
    emitted = []
    results, timing, observations = run.evaluate(
        ["fake"], [dataset], splits=0, seed=0, record_callback=emitted.append
    )

    assert results["checkpointed"]["fake"] == observations["checkpointed"]["fake"][0]
    assert len(timing["fake"]["fit"]) == 1
    assert len(emitted) == 1
    assert emitted[0]["status"] == "ok"
    assert emitted[0]["metric"] == "roc_auc"

    cached = {run._fold_record_key(emitted[0]): emitted[0]}
    monkeypatch.setattr(
        models,
        "build",
        lambda _name, _task: (_ for _ in ()).throw(AssertionError("resume refit")),
    )
    resumed, resumed_timing, _ = run.evaluate(["fake"], [dataset], splits=0, seed=0, completed_records=cached)

    assert resumed == results
    assert resumed_timing["fake"]["fit"][0] == pytest.approx(float(emitted[0]["fit_seconds"]))


def test_fold_records_round_trip_atomically(tmp_path):
    path = tmp_path / "arena.folds.csv"
    record = {
        "dataset": "task",
        "model": "tabpvn",
        "fold": 2,
        "task": "regression",
        "metric": "neg_rmse",
        "score": "-1.25",
        "fit_seconds": "2.5",
        "inference_seconds": "0.1",
        "status": "ok",
        "error": "",
        "implementation": "fingerprint",
    }
    records = {run._fold_record_key(record): record}

    run._write_fold_records(path, records)
    loaded = run._read_fold_records(path)

    assert loaded[run._fold_record_key(record)] == {key: str(value) for key, value in record.items()}
    assert not (tmp_path / "arena.folds.csv.tmp").exists()
    assert run._fold_record_path(tmp_path / "arena.csv").name == "arena.folds.csv"


def test_fold_encoder_fits_categories_only_on_training_rows():
    X_train = pd.DataFrame({"value": [0.0, 1.0, np.nan, 2.0], "kind": ["a", "b", "a", None]})
    X_test = pd.DataFrame({"value": [3.0, np.nan], "kind": ["unseen", "a"]})
    y = np.array([0, 1, 0, 1])

    estimator = models.build("hgb", "classification").fit(X_train, y)
    proba = estimator.predict_proba(X_test)

    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_fold_encoder_handles_pandas_category_columns():
    X_train = pd.DataFrame(
        {
            "value": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "month": pd.Categorical(["Aug", "Sep", "Aug", None, "Oct", "Sep"]),
            "weekend": [False, True, False, True, False, True],
        }
    )
    X_test = pd.DataFrame(
        {
            "value": [6.0, 7.0],
            "month": pd.Categorical(["Nov", "Aug"]),
            "weekend": [True, False],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    estimator = models.build("hgb", "classification").fit(X_train, y)
    proba = estimator.predict_proba(X_test)

    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_declared_regression_overrides_discrete_target_heuristic_through_preprocessing():
    X = pd.DataFrame(
        {
            "value": np.linspace(0.0, 1.0, 84),
            "kind": pd.Categorical(np.resize(["a", "b", "c"], 84)),
        }
    )
    y = np.resize(np.arange(3, 10), len(X))
    model = TabPVN(
        task="regression",
        seed=5,
        boost={"rounds": 30, "lr": 0.05, "depth": 3, "leaf": 8, "patience": 5, "refit": False},
    ).fit(X, y)

    assert model.mode == "regression"
    assert model._prep._target_is_classification is False
    assert model.predict(X.iloc[:5]).dtype.kind == "f"


def test_tabpvn_rejects_unknown_declared_task():
    with pytest.raises(ValueError, match="task must be"):
        TabPVN(task="ordinal")


def test_target_encoding_is_stable_and_rejects_invalid_regression_target():
    assert np.array_equal(_encode_target(["yes", "no", "yes"], "classification"), [1, 0, 1])
    with np.testing.assert_raises(ValueError):
        _encode_target([1.0, np.nan], "regression")
