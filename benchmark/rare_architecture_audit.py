"""Bounded ablation for TabPVN's proof-carrying rare-event architecture.

The target combines a six-fact exact-cardinality concept with a continuous
tail condition.  Accuracy is intentionally not reported because a constant
negative classifier exceeds 98% on this task; average precision is the primary
rare-event metric.  Both variants use the same fixed booster budget so the
audit isolates AP checkpointing and replayable symbolic predicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from tabpvn import TabPVN


class _FixedAuditBudget(TabPVN):
    def __init__(self, *, rounds=60, seed=0):
        super().__init__(seed=seed)
        self.audit_rounds = int(rounds)

    def _auto_tune(self, X, y):
        return {
            "rounds": self.audit_rounds,
            "lr": 0.08,
            "depth": 2,
            "leaf": 8,
            "patience": 15,
        }


class _NoRareArchitecture(_FixedAuditBudget):
    def _auto_rare_interactions(self, X, y, boost):
        return None


def _dataset(train_rows, test_rows, seed):
    rng = np.random.default_rng(seed)
    total = int(train_rows + test_rows)
    facts = rng.integers(0, 2, size=(total, 6)).astype(float)
    numeric = rng.random((total, 8))
    X = np.column_stack([facts, numeric])
    y = ((facts.sum(axis=1) == 3) & (numeric[:, 0] > 0.95)).astype(int)
    order = rng.permutation(total)
    return X[order[:train_rows]], y[order[:train_rows]], X[order[train_rows:]], y[order[train_rows:]]


def run_audit(train_rows=25_000, test_rows=25_000, rounds=60, seed=119):
    X_train, y_train, X_test, y_test = _dataset(train_rows, test_rows, seed)
    variants = (
        ("no_rare_architecture", _NoRareArchitecture),
        ("rare_architecture", _FixedAuditBudget),
    )
    rows = []
    for name, estimator in variants:
        started = time.perf_counter()
        model = estimator(rounds=rounds, seed=0).fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        positive = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
        probability = model.predict_proba(X_test)[:, positive]
        gate = {entry["name"]: entry for entry in model.candidate_report_}
        architecture = (model.rare_event_report_ or {}).get("architecture_gate", {})
        rows.append(
            {
                "variant": name,
                "seed": seed,
                "train_rows": int(train_rows),
                "test_rows": int(test_rows),
                "train_events": int(y_train.sum()),
                "test_events": int(y_test.sum()),
                "rounds": int(rounds),
                "fit_seconds": float(fit_seconds),
                "test_average_precision": float(average_precision_score(y_test, probability)),
                "test_roc_auc": float(roc_auc_score(y_test, probability)),
                "derived_features": int(len(model.interaction_features_)),
                "rules_selected": bool(gate.get("rare_symbolic_predicate_boost", {}).get("selected", False)),
                "proposal_objective": architecture.get("proposal_objective"),
                "fold_booster_fits": architecture.get("fold_booster_fits"),
                "certify": float(model.certify(X_test[:100])),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=25_000)
    parser.add_argument("--test-rows", type=int, default=25_000)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=119)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = run_audit(args.train_rows, args.test_rows, args.rounds, args.seed)
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
