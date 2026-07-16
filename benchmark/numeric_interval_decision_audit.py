"""Targeted TabArena audit for the numeric interval decision and rank head.

The calibrated probability, public rank, and accuracy decision surfaces are
evaluated from one fitted model. The default runs only the motivating maternal-health fold;
``--sentinels`` adds five compact rejection checks without invoking TabPFN or a
full Arena suite.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from benchmark.datasets import tabarena_suite
from tabpvn import TabPVN


def _auc(target, probability):
    if probability.shape[1] == 2:
        return float(roc_auc_score(target, probability[:, 1]))
    return float(roc_auc_score(target, probability, multi_class="ovo", average="macro"))


def audit(dataset, fold_index=0):
    train, test = dataset.splits[fold_index]
    model = TabPVN(seed=0, task="classification")
    started = time.perf_counter()
    model.fit(dataset.X.iloc[train], dataset.y[train])
    fit_seconds = time.perf_counter() - started

    calibrated_probability = model.predict_calibrated_proba(dataset.X.iloc[test])
    probability = model.predict_proba(dataset.X.iloc[test])
    classes = np.asarray(model.classes_)
    calibrated_labels = classes[calibrated_probability.argmax(1)]
    probability_labels = classes[probability.argmax(1)]
    decision_labels = model.predict(dataset.X.iloc[test])
    hidden_probability = probability
    triple_fallback_overrides = 0
    interval_report = {}
    if model._numeric_interval is not None:
        encoded = model._X(dataset.X.iloc[test])
        hidden_probability = model._numeric_interval.combine(
            calibrated_probability, encoded, model._numeric_interval_w
        )
        interval_report = model._numeric_interval.report()
        changed = np.flatnonzero(decision_labels != calibrated_labels)
        triple_fallback_overrides = sum(
            model._numeric_interval.evidence(
                encoded[row : row + 1],
                0,
                calibrated_probability[row : row + 1],
                model._numeric_interval_w,
            ).get("decision_mode")
            == model._numeric_interval.TRIPLE_FALLBACK
            for row in changed
        )
    gate = model.numeric_interval_report_[-1]
    baseline_gate = model.numeric_interval_report_[0]
    return {
        "dataset": dataset.name,
        "fold": fold_index,
        "train_rows": len(train),
        "test_rows": len(test),
        "fit_seconds": fit_seconds,
        "selected": model._numeric_interval is not None,
        "permission": gate.get("permission", ""),
        "aggregation": gate.get("aggregation", ""),
        "smoothing": gate.get("smoothing", ""),
        "weight": gate.get("weight", ""),
        "gate_reason": gate.get("reason", ""),
        "oof_base_accuracy": baseline_gate.get("mean_score", ""),
        "oof_decision_accuracy": gate.get("mean_score", ""),
        "oof_base_rank_auc": gate.get("base_rank_auc", ""),
        "oof_interval_rank_auc": gate.get("rank_auc", ""),
        "oof_rank_auc_delta": gate.get("rank_auc_delta", ""),
        "oof_fold_rank_auc_delta": gate.get("fold_rank_auc_delta", ""),
        "calibrated_argmax_accuracy": float(accuracy_score(dataset.y[test], calibrated_labels)),
        "probability_argmax_accuracy": float(accuracy_score(dataset.y[test], probability_labels)),
        "decision_accuracy": float(accuracy_score(dataset.y[test], decision_labels)),
        "accuracy_delta": float(
            accuracy_score(dataset.y[test], decision_labels)
            - accuracy_score(dataset.y[test], calibrated_labels)
        ),
        "calibrated_auc": _auc(dataset.y[test], calibrated_probability),
        "arena_auc": _auc(dataset.y[test], probability),
        "hidden_posterior_auc": _auc(dataset.y[test], hidden_probability),
        "decision_overrides": int(np.sum(decision_labels != calibrated_labels)),
        "triple_fallback_overrides": int(triple_fallback_overrides),
        "triple_families": interval_report.get("triple_families", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sentinels", action="store_true")
    parser.add_argument(
        "--out",
        default="results/tabpvn_numeric_interval_decision_audit.csv",
    )
    args = parser.parse_args()
    names = ["maternal_health_risk"]
    if args.sentinels:
        names.extend(
            [
                "diabetes",
                "qsar-biodeg",
                "blood-transfusion-service-center",
                "credit-g",
                "Marketing_Campaign",
            ]
        )
    records = [
        audit(dataset, fold_index=args.fold) for dataset in tabarena_suite(size="le10k", dataset_names=names)
    ]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(
            f"{record['dataset']}: selected={record['selected']} "
            f"accuracy={record['calibrated_argmax_accuracy']:.4f}"
            f"->{record['decision_accuracy']:.4f} "
            f"rank_auc={record['calibrated_auc']:.6f}"
            f"->{record['arena_auc']:.6f} "
            f"fit={record['fit_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
