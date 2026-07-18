"""Probe a bounded certified pairwise multiclass read.

For compact multiclass tables, one depth-four binary certified booster is fit
for each class pair.  Query probabilities are the normalized sum of each
pair's two probabilities.  The combiner has no fitted parameters, and every
contribution remains an ordinary additive proof tree.  This module is
research-only until shared OOF evidence and untouched official folds agree.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from benchmark.affine_composition_probe import _evaluate
from benchmark.datasets import tabarena_suite
from benchmark.global_shape_probe import _GlobalShapeProbeTabPVN
from tabpvn.base import _classification_rank_score, _preserve_certified_class

DEFAULT_CASES = tuple(("maternal_health_risk", fold) for fold in range(3))


def _probability(model, X):
    scores = model._scores(X)
    exponential = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exponential / exponential.sum(axis=1, keepdims=True)


def _fit_pairwise(owner, X, y, classes):
    models = []
    for first, second in combinations(classes, 2):
        rows = (y == first) | (y == second)
        classifier = owner._classifier(
            rounds=240,
            lr=0.05,
            depth=4,
            leaf=12,
            patience=20,
            validation_metric="auc",
            stratified_holdout=True,
            refit=True,
        )
        models.append(
            (
                first,
                second,
                owner._fit_certified(classifier, X[rows], y[rows]),
            )
        )
    return tuple(models)


def _pairwise_vote(models, X, classes):
    class_index = {label: index for index, label in enumerate(classes)}
    votes: NDArray[np.float64] = np.zeros((len(X), len(classes)), dtype=float)
    for first, second, model in models:
        probability = _probability(model, X)
        model_index = {label: index for index, label in enumerate(model.classes_)}
        votes[:, class_index[first]] += probability[:, model_index[first]]
        votes[:, class_index[second]] += probability[:, model_index[second]]
    return votes / votes.sum(axis=1, keepdims=True)


def _pairwise_logit(models, X, classes):
    """Least-squares class logits from the complete graph of pair margins."""
    class_index = {label: index for index, label in enumerate(classes)}
    scores: NDArray[np.float64] = np.zeros((len(X), len(classes)), dtype=float)
    for first, second, model in models:
        probability = _probability(model, X)
        model_index = {label: index for index, label in enumerate(model.classes_)}
        difference = np.log(np.clip(probability[:, model_index[first]], 1e-12, 1.0)) - np.log(
            np.clip(probability[:, model_index[second]], 1e-12, 1.0)
        )
        scores[:, class_index[first]] += difference
        scores[:, class_index[second]] -= difference
    scores /= len(classes)
    exponential = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exponential / exponential.sum(axis=1, keepdims=True)


def audit(dataset, fold_index):
    train, test = dataset.splits[fold_index]
    model = _GlobalShapeProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], dataset.y[train])
    probe = model.global_shape_probe_
    X_train = model._X(dataset.X.iloc[train])
    X_test = model._X(dataset.X.iloc[test])
    labels = np.asarray(dataset.y[train])
    classes = np.asarray(model.classes_)
    class_index = {label: index for index, label in enumerate(classes)}
    encoded_train = np.asarray([class_index[label] for label in labels], dtype=np.int32)
    encoded_test = np.asarray(
        [class_index[label] for label in np.asarray(dataset.y[test])],
        dtype=np.int32,
    )
    splits = tuple(probe["splits"])
    evidence_rows = np.asarray(probe["evidence_rows"], dtype=np.int64)
    base = np.asarray(probe["base"], dtype=float)

    oof_pairwise = np.zeros_like(base)
    oof_pairwise_logit = np.zeros_like(base)
    covered: NDArray[np.bool_] = np.zeros(len(labels), dtype=bool)
    tree_counts = []
    for fold_train, fold_valid in splits:
        pair_models = _fit_pairwise(
            model,
            X_train[fold_train],
            labels[fold_train],
            classes,
        )
        oof_pairwise[fold_valid] = _pairwise_vote(
            pair_models,
            X_train[fold_valid],
            classes,
        )
        oof_pairwise_logit[fold_valid] = _pairwise_logit(
            pair_models,
            X_train[fold_valid],
            classes,
        )
        covered[fold_valid] = True
        tree_counts.append(sum(len(pair_model.trees_) for _a, _b, pair_model in pair_models))
    if not np.all(covered[evidence_rows]):
        raise ValueError("pairwise probe requires fold-local evidence for every scored row")

    full_models = _fit_pairwise(model, X_train, labels, classes)
    heldout_pairwise = _pairwise_vote(full_models, X_test, classes)
    heldout_pairwise_logit = _pairwise_logit(full_models, X_test, classes)
    heldout_base = model._blended_proba(
        X_test,
        include_prior_rank=False,
        include_affine_rank=False,
    )
    heldout_base_score = _classification_rank_score(encoded_test, heldout_base)

    records = []
    candidates = (
        ("vote_direct_rank", oof_pairwise, heldout_pairwise, 1.0),
        ("vote_half_blend", oof_pairwise, heldout_pairwise, 0.5),
        ("logit_direct_rank", oof_pairwise_logit, heldout_pairwise_logit, 1.0),
        ("logit_half_blend", oof_pairwise_logit, heldout_pairwise_logit, 0.5),
    )
    for authority, oof_member, heldout_member, blend in candidates:
        candidate = (1.0 - blend) * base + blend * oof_member
        projected = _preserve_certified_class(base, candidate)
        evaluation = _evaluate(
            encoded_train,
            base,
            projected,
            splits,
            evidence_rows,
        )
        heldout = (1.0 - blend) * heldout_base + blend * heldout_member
        heldout_projected = _preserve_certified_class(heldout_base, heldout)
        heldout_score = _classification_rank_score(encoded_test, heldout_projected)
        records.append(
            {
                "dataset": dataset.name,
                "fold": int(fold_index),
                "candidate": f"pairwise_{authority}",
                "blend": float(blend),
                "rows": int(len(train)),
                "features": int(X_train.shape[1]),
                "classes": int(len(classes)),
                "pair_models": int(len(full_models)),
                "mean_oof_pair_trees": float(np.mean(tree_counts)),
                "full_pair_trees": int(sum(len(pair_model.trees_) for _a, _b, pair_model in full_models)),
                **evaluation,
                "heldout_base_rank_auc": heldout_base_score,
                "heldout_rank_auc": heldout_score,
                "heldout_rank_auc_delta": heldout_score - heldout_base_score,
            }
        )
    return records


def _parse_cases(value):
    if not value:
        return DEFAULT_CASES
    cases = []
    for item in value.split(","):
        name, separator, fold = item.rpartition(":")
        if not separator or not name:
            raise ValueError("cases must use dataset:fold entries")
        cases.append((name, int(fold)))
    return tuple(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", help="Comma-separated dataset:fold entries")
    parser.add_argument(
        "--out",
        default="results/tabpvn_pairwise_boost_probe.csv",
    )
    args = parser.parse_args()
    cases = _parse_cases(args.cases)
    datasets = {
        dataset.name: dataset
        for dataset in tabarena_suite(
            size="all",
            dataset_names=sorted({name for name, _fold in cases}),
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
            f"{record['dataset']} fold={record['fold']} {record['candidate']} "
            f"oof={record['oof_rank_auc_delta']:+.6f} "
            f"heldout={record['heldout_rank_auc_delta']:+.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
