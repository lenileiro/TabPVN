"""Fast transfer audit for TabPVN's categorical posterior challenger.

The baseline and challenger share one fitted TabPVN model. Baseline metrics use
the probability stack immediately before the posterior update, so this isolates
the architecture change without paying for or confounding a second model fit.
The synthetic cases separately exercise class changes, rank-only permission,
bounded sequential pooling, and sparse hierarchical backoff; Telco is the
fail-closed dense-table sentinel.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpvn import TabPVN


def categorical_pair_lookup(seed=41, rows=1_200):
    rng = np.random.default_rng(seed)
    levels = 8
    left = rng.integers(levels, size=rows)
    right = rng.integers(levels, size=rows)
    channel = rng.integers(5, size=rows)
    contract = rng.integers(4, size=rows)
    lookup = rng.integers(3, size=(levels, levels))
    target = lookup[left, right]
    noisy = rng.random(rows) < 0.04
    target[noisy] = rng.integers(3, size=int(noisy.sum()))
    features = pd.DataFrame(
        {
            "account_type": [f"a{value}" for value in left],
            "region": [f"r{value}" for value in right],
            "channel": [f"c{value}" for value in channel],
            "contract": [f"d{value}" for value in contract],
        }
    )
    for index in range(12):
        features[f"noise_{index}"] = rng.normal(size=rows)
    return features, target


def categorical_pair_risk(seed=13, rows=3_600):
    """Imbalanced pair-specific risk where useful ranking stays below 0.5."""
    rng = np.random.default_rng(seed)
    levels = 10
    account = rng.integers(levels, size=rows)
    region = rng.integers(levels, size=rows)
    risk = rng.uniform(0.025, 0.28, size=(levels, levels))
    target = (rng.random(rows) < risk[account, region]).astype(int)
    features = pd.DataFrame(
        {
            "account": [f"a{value}" for value in account],
            "region": [f"r{value}" for value in region],
            "channel": [f"c{value}" for value in rng.integers(5, size=rows)],
        }
    )
    for index in range(10):
        features[f"noise_{index}"] = rng.normal(size=rows)
    return features, target


def categorical_evidence_stack(seed=27, rows=3_600):
    """Several independent weak facts whose joint posterior is stronger."""
    rng = np.random.default_rng(seed)
    target = rng.integers(2, size=rows)
    codes = np.column_stack([np.where(rng.random(rows) < 0.64, target, 1 - target) for _ in range(6)])
    features = pd.DataFrame(
        {
            f"signal_{group}": [f"level_{value}" for value in codes[:, group]]
            for group in range(codes.shape[1])
        }
    )
    for index in range(12):
        features[f"noise_{index}"] = rng.normal(size=rows)
    return features, target


def categorical_sparse_backoff(seed=81, rows=3_600):
    """Sparse pair cells backed by stable single-category parent effects."""
    rng = np.random.default_rng(seed)
    groups, levels = 4, 20
    effects = rng.normal(0.0, 0.9, size=(groups, levels))
    codes = rng.integers(levels, size=(rows, groups))
    score = sum(effects[group, codes[:, group]] for group in range(groups))
    probability = 1.0 / (1.0 + np.exp(-score))
    target = (rng.random(rows) < probability).astype(int)
    features = pd.DataFrame(
        {f"signal_{group}": [f"level_{value}" for value in codes[:, group]] for group in range(groups)}
    )
    for index in range(8):
        features[f"noise_{index}"] = rng.normal(size=rows)
    return features, target


def telco_churn():
    path = Path("data/kaggle/blastchar__telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    target = (frame.pop("Churn") == "Yes").astype(int).to_numpy()
    return frame.drop(columns=["customerID"]), target


def _auc(target, probability):
    if probability.shape[1] == 2:
        return float(roc_auc_score(target, probability[:, 1]))
    return float(roc_auc_score(target, probability, multi_class="ovo", average="macro"))


def audit(name, features, target, seed):
    train, test = train_test_split(
        np.arange(len(target)),
        test_size=0.30 if name.startswith("categorical_") else 0.25,
        random_state=seed,
        stratify=target,
    )
    model = TabPVN(seed=0, task="classification")
    started = time.perf_counter()
    model.fit(features.iloc[train], target[train])
    fit_seconds = time.perf_counter() - started
    encoded = model._X(features.iloc[test])
    baseline = model._blended_proba(encoded, include_posterior=False)
    final = model._blended_proba(encoded, include_posterior=True)
    baseline_accuracy = float(accuracy_score(target[test], baseline.argmax(1)))
    final_accuracy = float(accuracy_score(target[test], final.argmax(1)))
    baseline_auc, final_auc = _auc(target[test], baseline), _auc(target[test], final)
    baseline_loss = float(log_loss(target[test], baseline))
    final_loss = float(log_loss(target[test], final))
    gate = model.category_posterior_report_[-1]
    return {
        "dataset": name,
        "train_rows": len(train),
        "test_rows": len(test),
        "fit_seconds": fit_seconds,
        "posterior_selected": model._category_posterior is not None,
        "posterior_permission": getattr(model, "_category_posterior_permission", None),
        "posterior_aggregation": getattr(model, "_category_posterior_aggregation", None),
        "posterior_smoothing": getattr(model, "_category_posterior_smoothing", None),
        "posterior_weight": getattr(model, "_category_posterior_w", ""),
        "gate_reason": gate.get("reason", ""),
        "base_accuracy": baseline_accuracy,
        "final_accuracy": final_accuracy,
        "accuracy_delta": final_accuracy - baseline_accuracy,
        "base_auc": baseline_auc,
        "final_auc": final_auc,
        "auc_delta": final_auc - baseline_auc,
        "base_log_loss": baseline_loss,
        "final_log_loss": final_loss,
        "log_loss_delta": final_loss - baseline_loss,
        "test_overrides": int(np.sum(baseline.argmax(1) != final.argmax(1))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="results/tabpvn_categorical_posterior_audit.csv",
    )
    parser.add_argument("--include-telco", action="store_true")
    args = parser.parse_args()

    datasets = [
        ("categorical_pair_lookup", *categorical_pair_lookup()),
        ("categorical_pair_risk", *categorical_pair_risk()),
        ("categorical_evidence_stack", *categorical_evidence_stack()),
        ("categorical_sparse_backoff", *categorical_sparse_backoff()),
    ]
    if args.include_telco:
        telco = telco_churn()
        if telco is not None:
            datasets.append(("telco_customer_churn", *telco))
    records = [
        audit(name, features, target, seed=19 if name.startswith("categorical_") else 17)
        for name, features, target in datasets
    ]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(
            f"{record['dataset']}: selected={record['posterior_selected']} "
            f"permission={record['posterior_permission']} "
            f"aggregation={record['posterior_aggregation']} "
            f"smoothing={record['posterior_smoothing']} "
            f"accuracy={record['base_accuracy']:.4f}->{record['final_accuracy']:.4f} "
            f"auc={record['base_auc']:.4f}->{record['final_auc']:.4f} "
            f"fit={record['fit_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
