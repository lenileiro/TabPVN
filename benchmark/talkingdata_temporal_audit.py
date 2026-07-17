"""Chronological Kaggle TalkingData audit for TabPVN's automatic event path.

The public sample is a case-control subset of the original competition data:
all attributed clicks were retained while most negatives were discarded. This
script therefore measures architecture behavior on future rows, not a Kaggle
leaderboard estimate. ``attributed_time`` is never loaded because it reveals the
target directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)

from tabpvn import TabPVN
from tabpvn.temporal import TemporalLaplaceMap
from tabpvn.validation import FutureValidation

DEFAULT_DATA = Path("data/kaggle/talkingdata-adtracking-fraud-detection/train_sample.csv")
FEATURE_COLUMNS = ("ip", "app", "device", "os", "channel", "click_time")
TARGET = "is_attributed"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _load_window(path: Path, rows: int, train_fraction: float):
    frame = pd.read_csv(
        path,
        usecols=[*FEATURE_COLUMNS, TARGET],
        parse_dates=["click_time"],
    )
    if not frame["click_time"].is_monotonic_increasing:
        frame = frame.sort_values("click_time", kind="stable")
    if rows > 0 and len(frame) > rows:
        frame = frame.iloc[-rows:]
    frame = frame.reset_index(drop=True)
    requested = int(np.clip(round(train_fraction * len(frame)), 1, len(frame) - 1))
    boundary = frame["click_time"].iloc[requested]
    cut = int(frame["click_time"].searchsorted(boundary, side="left"))
    if cut == 0 or cut == len(frame):
        raise ValueError("timestamp boundary did not produce non-empty train and test partitions")
    train = frame.iloc[:cut].reset_index(drop=True)
    test = frame.iloc[cut:].reset_index(drop=True)
    if train["click_time"].max() >= test["click_time"].min():
        raise AssertionError("chronological split leaked an equal timestamp across the boundary")
    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError("both chronological partitions must contain both target classes")
    return train, test, boundary


def _scores(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "log_loss": float(log_loss(y, np.c_[1.0 - probability, probability], labels=[0, 1])),
    }


def _hgb_matrix(frame: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    timestamp = frame["click_time"]
    elapsed_hours = (timestamp - origin).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    return np.column_stack(
        [frame[column].to_numpy(dtype=float) for column in FEATURE_COLUMNS[:-1]]
        + [
            elapsed_hours,
            timestamp.dt.hour.to_numpy(dtype=float),
            timestamp.dt.dayofweek.to_numpy(dtype=float),
        ]
    )


def _evaluate_hgb(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    origin = train["click_time"].min()
    X_train = _hgb_matrix(train, origin)
    X_test = _hgb_matrix(test, origin)
    y_train = train[TARGET].to_numpy(dtype=int)
    y_test = test[TARGET].to_numpy(dtype=int)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=400,
        l2_regularization=1.0,
        random_state=0,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability = model.predict_proba(X_test)[:, 1]
    probability_seconds = time.perf_counter() - started
    prediction = (probability >= 0.5).astype(int)
    return {
        "model": "hgb",
        **_scores(y_test, probability, prediction),
        "fit_seconds": fit_seconds,
        "predict_proba_seconds": probability_seconds,
    }


def _evaluate_tabpvn(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    X_train = train.loc[:, FEATURE_COLUMNS]
    X_test = test.loc[:, FEATURE_COLUMNS]
    y_train = train[TARGET].to_numpy(dtype=int)
    y_test = test[TARGET].to_numpy(dtype=int)
    model = TabPVN(seed=0, task="classification")
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = model.predict_proba(X_test)
    probability_seconds = time.perf_counter() - started
    classes = np.asarray(model.classes_)
    positive = int(np.flatnonzero(classes == 1)[0])
    probability = probabilities[:, positive]
    prediction = classes[probabilities.argmax(axis=1)].astype(int)
    return {
        "model": "tabpvn",
        **_scores(y_test, probability, prediction),
        "fit_seconds": fit_seconds,
        "predict_proba_seconds": probability_seconds,
        "temporal_selected": bool(model.temporal_selected_),
        "event_schema": model.event_schema_,
        "event_discovery_report": model.event_discovery_report_,
        "validation_report": model.validation_report_,
    }


def _evaluate_tabpvn_temporal_probe(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Force the rejected IP-history candidate without changing the default gate."""
    X_train = train.loc[:, FEATURE_COLUMNS]
    X_test = test.loc[:, FEATURE_COLUMNS]
    y_train = train[TARGET].to_numpy(dtype=int)
    y_test = test[TARGET].to_numpy(dtype=int)
    temporal = TemporalLaplaceMap("ip", "click_time")
    started = time.perf_counter()
    augmented_train = temporal.fit_augment(X_train)
    augmented_test = temporal.augment(X_test)
    validation, _ = FutureValidation.from_timestamps(X_train["click_time"]).sorted()
    model = TabPVN(seed=0, task="classification")
    model._pending_validation = validation
    try:
        model._fit_pipeline(augmented_train, y_train)
    finally:
        model._pending_validation = None
        model._fit_validation = None
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = model.predict_proba(augmented_test)
    probability_seconds = time.perf_counter() - started
    classes = np.asarray(model.classes_)
    positive = int(np.flatnonzero(classes == 1)[0])
    probability = probabilities[:, positive]
    prediction = classes[probabilities.argmax(axis=1)].astype(int)
    return {
        "model": "tabpvn_temporal_probe",
        "research_only": True,
        "forced_schema": {"entity": "ip", "timestamp": "click_time", "value_columns": []},
        **_scores(y_test, probability, prediction),
        "fit_seconds": fit_seconds,
        "predict_proba_seconds": probability_seconds,
        "validation_report": model.validation_report_,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--models", default="hgb,tabpvn")
    parser.add_argument("--out", type=Path, default=Path("results/talkingdata_temporal_500k.json"))
    args = parser.parse_args()
    if args.rows < 2:
        parser.error("--rows must be at least 2")
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")

    train, test, boundary = _load_window(args.data, args.rows, args.train_fraction)
    result: dict[str, Any] = {
        "dataset": "Kaggle matleonard/feature-engineering-data train_sample.csv",
        "source": "TalkingData AdTracking sample with downsampled negatives",
        "excluded_leakage_columns": ["attributed_time"],
        "window_rows": int(len(train) + len(test)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(train[TARGET].mean()),
        "test_positive_rate": float(test[TARGET].mean()),
        "train_end": train["click_time"].max(),
        "test_start": test["click_time"].min(),
        "requested_boundary": boundary,
        "models": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in [value.strip() for value in args.models.split(",") if value.strip()]:
        print(f"running {name} on {len(train):,} past / {len(test):,} future rows", flush=True)
        if name == "hgb":
            measured = _evaluate_hgb(train, test)
        elif name == "tabpvn":
            measured = _evaluate_tabpvn(train, test)
        elif name == "tabpvn_temporal_probe":
            measured = _evaluate_tabpvn_temporal_probe(train, test)
        else:
            parser.error(f"unknown model {name!r}; choose hgb,tabpvn,tabpvn_temporal_probe")
        result["models"][name] = measured
        args.out.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
        print(
            f"{name}: auc={measured['roc_auc']:.6f} ap={measured['average_precision']:.6f} "
            f"accuracy={measured['accuracy']:.6f} fit={measured['fit_seconds']:.2f}s",
            flush=True,
        )
        gc.collect()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
