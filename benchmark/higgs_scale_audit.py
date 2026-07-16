"""Run the packaged TabPVN default on the local million-row HIGGS split.

This is a scale audit outside the official TabArena v0.1 task list.  The local
Kaggle artifact provides a fixed 1,000,000-row train file and a disjoint
50,000-row test file; column zero is the binary target and the remaining 28
columns are numeric features.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from tabpvn import TabPVN


def _peak_rss_gib():
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_used = raw if sys.platform == "darwin" else raw * 1024.0
    return bytes_used / (1024.0**3)


def _load(path):
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 29 or not np.isfinite(values).all():
        raise ValueError(f"expected a finite 29-column HIGGS matrix, got {values.shape}")
    target = values[:, 0].astype(np.int8, copy=True)
    if not np.isin(target, (0, 1)).all():
        raise ValueError("HIGGS target must contain only 0 and 1")
    return values, values[:, 1:], target


def audit(train_path, test_path):
    started = time.perf_counter()
    train_values, X_train, y_train = _load(train_path)
    test_values, X_test, y_test = _load(test_path)
    load_seconds = time.perf_counter() - started

    model = TabPVN(seed=0, task="classification")
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probability = model.predict_proba(X_test)
    probability_seconds = time.perf_counter() - started
    started = time.perf_counter()
    decision = model.predict(X_test)
    decision_seconds = time.perf_counter() - started

    classes = np.asarray(model.classes_)
    probability_decision = classes[probability.argmax(1)]
    fit_rows = getattr(model._pred, "fit_rows_", None)
    verifier = getattr(model._pred, "ver_", None)
    record = {
        "dataset": "kaggle_higgs_1m",
        "arena_membership": False,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "features": X_train.shape[1],
        "load_seconds": load_seconds,
        "fit_seconds": fit_seconds,
        "predict_proba_seconds": probability_seconds,
        "predict_seconds": decision_seconds,
        "roc_auc": float(roc_auc_score(y_test, probability[:, 1])),
        "probability_argmax_accuracy": float(accuracy_score(y_test, probability_decision)),
        "decision_accuracy": float(accuracy_score(y_test, decision)),
        "log_loss": float(log_loss(y_test, probability, labels=classes)),
        "peak_rss_gib": _peak_rss_gib(),
        "fitted_rows": len(y_train) if fit_rows is None else len(fit_rows),
        "verifier_rows": 0 if verifier is None else len(verifier),
        "numeric_interval_selected": model._numeric_interval is not None,
        "categorical_posterior_selected": model._category_posterior is not None,
        "resolved_boost": json.dumps(model.boost_, sort_keys=True),
    }
    # Keep the loaded owners alive until every model read and metric has finished.
    del train_values, test_values
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default="data/kaggle/dodolong__higgs1m/higgs-train-1m.csv",
    )
    parser.add_argument(
        "--test",
        default="data/kaggle/dodolong__higgs1m/higgs-test.csv",
    )
    parser.add_argument("--out", default="results/tabpvn_higgs_1m_current.csv")
    args = parser.parse_args()

    record = audit(Path(args.train), Path(args.test))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
