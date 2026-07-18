"""Probe one cross-fitted certified expert for the weakest multiclass pair.

The complete one-vs-one ensemble loses useful incumbent evidence.  This audit
keeps the selected TabPVN probability stack intact and lets one binary
proof-tree booster reallocate only the probability mass of the weakest class
pair.  Pair selection for each OOF fold uses the other folds' leak-free base
predictions, so the scored fold does not choose its own expert.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from benchmark.affine_composition_probe import _evaluate
from benchmark.datasets import tabarena_suite
from benchmark.pairwise_boost_probe import _probability
from benchmark.rank_temperature_probe import _RankTemperatureProbeTabPVN
from tabpvn.base import _classification_rank_score, _preserve_certified_class

DEFAULT_CASES = tuple(("maternal_health_risk", fold) for fold in range(3))
COMPOSITIONS = (
    ("arithmetic_quarter", "arithmetic", 0.25),
    ("arithmetic_half", "arithmetic", 0.5),
    ("geometric_quarter", "geometric", 0.25),
    ("geometric_half", "geometric", 0.5),
)


def _pair_rank(target, probability, pair) -> float:
    first, second = pair
    rows = (target == first) | (target == second)
    if not rows.any() or np.unique(target[rows]).size != 2:
        return float("inf")
    first_auc = roc_auc_score(target[rows] == first, probability[rows, first])
    second_auc = roc_auc_score(target[rows] == second, probability[rows, second])
    return float(0.5 * (first_auc + second_auc))


def _weakest_pair(target, probability, rows, n_classes):
    ranked = [
        (_pair_rank(target[rows], probability[rows], pair), pair)
        for pair in combinations(range(n_classes), 2)
    ]
    return min(ranked, key=lambda item: (item[0], item[1]))[1]


def _fit_pair(owner, X, labels, classes, pair):
    first_label, second_label = classes[list(pair)]
    rows = (labels == first_label) | (labels == second_label)
    leaf = max(10, min(30, int(rows.sum()) // 40))
    classifier = owner._classifier(
        rounds=500,
        lr=0.05,
        depth=6,
        leaf=leaf,
        patience=20,
        validation_metric="auc",
        stratified_holdout=True,
        refit=True,
    )
    return owner._fit_certified(classifier, X[rows], labels[rows])


def _pair_member(model, X, classes, pair):
    probability = _probability(model, X)
    model_index = {label: index for index, label in enumerate(model.classes_)}
    first_label, second_label = classes[list(pair)]
    return probability[:, [model_index[first_label], model_index[second_label]]]


def _compose_pair(base, member, pair, composition, weight):
    candidate = np.asarray(base, dtype=float).copy()
    pair = np.asarray(pair, dtype=np.int64)
    mass = candidate[:, pair].sum(axis=1, keepdims=True)
    incumbent = np.divide(
        candidate[:, pair],
        mass,
        out=np.full((len(candidate), 2), 0.5, dtype=float),
        where=mass > 0.0,
    )
    if composition == "arithmetic":
        conditional = (1.0 - weight) * incumbent + weight * member
    elif composition == "geometric":
        tiny = np.finfo(float).tiny
        log_conditional = (1.0 - weight) * np.log(np.clip(incumbent, tiny, 1.0)) + weight * np.log(
            np.clip(member, tiny, 1.0)
        )
        log_conditional -= log_conditional.max(axis=1, keepdims=True)
        conditional = np.exp(log_conditional)
        conditional /= conditional.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unsupported pair composition: {composition}")
    candidate[:, pair] = mass * conditional
    return _preserve_certified_class(base, candidate)


def audit(dataset, fold_index: int) -> list[dict[str, object]]:
    train, test = dataset.splits[fold_index]
    model = _RankTemperatureProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], dataset.y[train])
    probe = model.rank_temperature_probe_

    classes = np.asarray(model.classes_)
    class_index = {label: index for index, label in enumerate(classes)}
    labels = np.asarray(dataset.y[train])
    oof_target = np.asarray([class_index[label] for label in labels], dtype=np.int32)
    heldout_target = np.asarray([class_index[label] for label in dataset.y[test]], dtype=np.int32)
    X_train = model._X(dataset.X.iloc[train])
    X_test = model._X(dataset.X.iloc[test])
    base = np.asarray(probe["probability"], dtype=float)
    evidence_rows = np.asarray(probe["evidence_rows"], dtype=np.int64)
    splits = tuple(probe["splits"])

    fold_members = []
    fold_pairs = []
    tree_counts = []
    for fold_train, fold_valid in splits:
        selection_rows = np.setdiff1d(
            evidence_rows,
            np.asarray(fold_valid, dtype=np.int64),
            assume_unique=False,
        )
        pair = _weakest_pair(oof_target, base, selection_rows, len(classes))
        pair_model = _fit_pair(
            model,
            X_train[fold_train],
            labels[fold_train],
            classes,
            pair,
        )
        fold_pairs.append(pair)
        fold_members.append((fold_valid, pair, _pair_member(pair_model, X_train[fold_valid], classes, pair)))
        tree_counts.append(len(pair_model.trees_))

    deploy_pair = _weakest_pair(oof_target, base, evidence_rows, len(classes))
    full_model = _fit_pair(model, X_train, labels, classes, deploy_pair)
    heldout_member = _pair_member(full_model, X_test, classes, deploy_pair)
    heldout_base = model.predict_proba(dataset.X.iloc[test])
    heldout_base_rank = _classification_rank_score(heldout_target, heldout_base)

    records: list[dict[str, object]] = []
    for name, composition, weight in COMPOSITIONS:
        candidate = base.copy()
        for fold_valid, pair, member in fold_members:
            candidate[fold_valid] = _compose_pair(base[fold_valid], member, pair, composition, weight)
        evaluation = _evaluate(
            oof_target,
            base,
            candidate,
            splits,
            evidence_rows,
        )
        heldout = _compose_pair(
            heldout_base,
            heldout_member,
            deploy_pair,
            composition,
            weight,
        )
        records.append(
            {
                "dataset": dataset.name,
                "fold": fold_index,
                "candidate": name,
                "composition": composition,
                "weight": weight,
                "rows": len(train),
                "features": X_train.shape[1],
                "classes": len(classes),
                "fold_pairs": [list(pair) for pair in fold_pairs],
                "deploy_pair": list(deploy_pair),
                "mean_oof_pair_trees": float(np.mean(tree_counts)),
                "full_pair_trees": len(full_model.trees_),
                **evaluation,
                "heldout_rank_auc": _classification_rank_score(heldout_target, heldout),
                "heldout_rank_auc_delta": (
                    _classification_rank_score(heldout_target, heldout) - heldout_base_rank
                ),
                "heldout_accuracy_delta": float(
                    np.mean(heldout.argmax(axis=1) == heldout_target)
                    - np.mean(heldout_base.argmax(axis=1) == heldout_target)
                ),
                "heldout_pair_rank_delta": (
                    _pair_rank(heldout_target, heldout, deploy_pair)
                    - _pair_rank(heldout_target, heldout_base, deploy_pair)
                ),
            }
        )
    return records


def _parse_cases(value: str | None) -> tuple[tuple[str, int], ...]:
    if not value:
        return DEFAULT_CASES
    cases = []
    for item in value.split(","):
        name, separator, fold = item.rpartition(":")
        if not separator or not name:
            raise ValueError("cases must use dataset:fold entries")
        cases.append((name, int(fold)))
    return tuple(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", help="Comma-separated dataset:fold entries")
    parser.add_argument("--out", default="results/tabpvn_weak_pair_boost_probe.csv")
    args = parser.parse_args()
    cases = _parse_cases(args.cases)
    datasets = {
        dataset.name: dataset
        for dataset in tabarena_suite(
            size="all",
            dataset_names=list(dict.fromkeys(name for name, _fold in cases)),
        )
    }
    records = [record for name, fold in cases for record in audit(datasets[name], fold)]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    for record in records:
        print(
            f"{record['dataset']} fold {record['fold']} {record['candidate']}: "
            f"pairs={record['fold_pairs']} deploy={record['deploy_pair']} "
            f"OOF={record['oof_rank_auc_delta']:+.6f} "
            f"heldout={record['heldout_rank_auc_delta']:+.6f} "
            f"pair={record['heldout_pair_rank_delta']:+.6f}"
        )


if __name__ == "__main__":
    main()
