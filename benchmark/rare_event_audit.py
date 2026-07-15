"""Bounded synthetic audit for TabPVN's rare-event training path.

This isolates the case-control reservoir from auto-configuration: both models
receive the same tree budget and fit cap, while only the rare-event evidence
policy changes. Average precision is the primary metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tabpvn.base import _fit_binary_thresholds, _fit_temperature
from tabpvn.certified_boost import AdditiveCertifiedClassifier


def _positive_probability(model, X, temperature=1.0):
    scores = model._scores(X) / temperature
    exp_scores = np.exp(scores - scores.max(1, keepdims=True))
    positive_column = list(model.classes_).index(1)
    return (exp_scores / exp_scores.sum(1, keepdims=True))[:, positive_column]


def run_audit(train_rows=100_000, test_rows=50_000, fit_cap=10_000, rounds=160, seed=91):
    X, y = make_classification(
        n_samples=train_rows + test_rows,
        n_features=30,
        n_informative=12,
        n_redundant=8,
        n_clusters_per_class=3,
        weights=[0.995, 0.005],
        class_sep=1.2,
        flip_y=0.0,
        random_state=seed,
    )
    order = np.random.default_rng(seed).permutation(len(y))
    train, test = order[:train_rows], order[train_rows:]
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    model_seed = 11
    common = {
        "rounds": rounds,
        "lr": 0.05,
        "depth": 6,
        "leaf": 20,
        "holdout": 0.05,
        "patience": 20,
        "refit": False,
        "fit_cap": fit_cap,
        "seed": model_seed,
    }
    variants = (
        ("proportional", {}),
        (
            "rare_event",
            {
                "rare_event": True,
                "rare_min_events": min(1_000, max(1, fit_cap // 4)),
                "min_verifier_events": min(200, max(1, fit_cap // 20)),
            },
        ),
    )
    rows = []
    for name, extra in variants:
        started = time.perf_counter()
        model = AdditiveCertifiedClassifier(**common, **extra).fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        probability = _positive_probability(model, X_test)
        sampled_y = y_train if model.fit_rows_ is None else y_train[model.fit_rows_]
        row = {
            "variant": name,
            "seed": seed,
            "model_seed": model_seed,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "train_event_rate": float(y_train.mean()),
            "test_events": int(y_test.sum()),
            "fit_cap": fit_cap,
            "rounds": rounds,
            "sample_events": int(sampled_y.sum()),
            "stages": int(len(model.trees_)),
            "fit_seconds": fit_seconds,
            "test_average_precision": float(average_precision_score(y_test, probability)),
            "test_roc_auc": float(roc_auc_score(y_test, probability)),
        }
        if name == "rare_event":
            verifier = model.ver_
            verifier_scores = model._scores(X_train[verifier])
            temperature = _fit_temperature(
                verifier_scores, y_train[verifier], model.classes_, model.ver_weight_
            )
            verifier_probability = _positive_probability(model, X_train[verifier], temperature)
            verifier_proba = np.column_stack([1.0 - verifier_probability, verifier_probability])
            _balanced, threshold, report = _fit_binary_thresholds(
                verifier_proba,
                y_train[verifier],
                model.classes_,
                rare_class=1,
                sample_weight=model.ver_weight_,
            )
            probability = _positive_probability(model, X_test, temperature)
            prediction = probability >= (0.5 if threshold is None else threshold)
            row.update(
                {
                    "rare_threshold": threshold,
                    "test_f1": float(f1_score(y_test, prediction)),
                    "test_precision": float(precision_score(y_test, prediction, zero_division=0)),
                    "test_recall": float(recall_score(y_test, prediction)),
                    "verifier_crossfit_f1": report["weighted_f1"],
                    "verifier_conservative_f1": report["conservative_f1"],
                }
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=100_000)
    parser.add_argument("--test-rows", type=int, default=50_000)
    parser.add_argument("--fit-cap", type=int, default=10_000)
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = run_audit(args.train_rows, args.test_rows, args.fit_cap, args.rounds, args.seed)
    print(json.dumps(rows, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
