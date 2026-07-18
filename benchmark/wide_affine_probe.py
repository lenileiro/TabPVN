"""Probe a fold-local screened affine read on very wide binary tables.

The production booster already has a deterministic 256-feature screen whose
proposal uses fit-side class moments and whose admission uses untouched rows.
This research audit derives that screen independently inside each shared OOF
fold, fits an explicit affine read only on the fold's pool, and evaluates the
result against the exact incumbent probability stack.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark.affine_composition_probe import _evaluate
from benchmark.datasets import tabarena_suite
from benchmark.global_shape_probe import _fold_priors, _GlobalShapeProbeTabPVN
from tabpvn.base import (
    _AFFINE_RANK_INVERSE_REGULARIZATION,
    _affine_rank_weight,
    _classification_rank_score,
    _preserve_certified_class,
)
from tabpvn.proposers.affine import AffineLogitRead

DEFAULT_CASES = tuple(("Bioresponse", fold) for fold in range(3))


def audit(dataset, fold_index: int) -> list[dict[str, object]]:
    train, test = dataset.splits[fold_index]
    train_target = np.asarray(dataset.y[train])
    test_target = np.asarray(dataset.y[test])
    model = _GlobalShapeProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], train_target)
    probe = model.global_shape_probe_
    classes = np.asarray(model.classes_)
    class_index = {value: index for index, value in enumerate(classes)}
    encoded_train = np.asarray([class_index[value] for value in train_target], dtype=np.int32)
    encoded_test = np.asarray([class_index[value] for value in test_target], dtype=np.int32)
    X_train = model._X(dataset.X.iloc[train])
    X_test = model._X(dataset.X.iloc[test])
    splits = tuple(probe["splits"])
    evidence_rows = np.asarray(probe["evidence_rows"], dtype=np.int64)
    base = np.asarray(probe["base"], dtype=float)
    affine_probability = base.copy()
    fold_pools = []
    covered: np.ndarray = np.zeros(len(train), dtype=bool)
    for fold_train, fold_valid in splits:
        pool = model._wide_feature_pool(X_train[fold_train], train_target[fold_train])
        member = AffineLogitRead(
            inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
            seed=0,
        ).fit(
            X_train[fold_train][:, pool],
            train_target[fold_train],
            classes=classes,
        )
        affine_probability[fold_valid] = member.proba(X_train[fold_valid][:, pool])
        fold_pools.append(pool)
        covered[fold_valid] = True
    if not np.all(covered[evidence_rows]):
        raise ValueError("wide affine evidence rows require fold-local predictions")

    full_pool = model._wide_feature_pool(X_train, train_target)
    full_member = AffineLogitRead(
        inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
        seed=0,
    ).fit(X_train[:, full_pool], train_target, classes=classes)
    heldout_affine = full_member.proba(X_test[:, full_pool])
    heldout_base = model._blended_proba(
        X_test,
        include_prior_rank=False,
        include_affine_rank=False,
    )
    heldout_base_rank = _classification_rank_score(encoded_test, heldout_base)
    heldout_base_accuracy = float(np.mean(heldout_base.argmax(axis=1) == encoded_test))
    fold_prior = _fold_priors(encoded_train, splits, len(classes))
    deployment_prior = np.bincount(encoded_train, minlength=len(classes)).astype(float)
    deployment_prior /= deployment_prior.sum()
    weight = _affine_rank_weight(len(train))
    fold_overlap = [
        len(np.intersect1d(pool, full_pool)) / len(np.union1d(pool, full_pool)) for pool in fold_pools
    ]

    records = []
    for composition in ("arithmetic", "prior_ratio"):
        candidate = AffineLogitRead.combine(
            base,
            affine_probability,
            weight,
            composition=composition,
            prior=fold_prior,
        )
        evaluation = _evaluate(encoded_train, base, candidate, splits, evidence_rows)
        projected = _preserve_certified_class(base, candidate)
        projected_evaluation = _evaluate(
            encoded_train,
            base,
            projected,
            splits,
            evidence_rows,
        )
        heldout = AffineLogitRead.combine(
            heldout_base,
            heldout_affine,
            weight,
            composition=composition,
            prior=deployment_prior,
        )
        heldout_projected = _preserve_certified_class(heldout_base, heldout)
        records.append(
            {
                "dataset": dataset.name,
                "fold": fold_index,
                "train_rows": len(train),
                "source_features": X_train.shape[1],
                "selected_features": len(full_pool),
                "fold_pool_jaccard": fold_overlap,
                "weight": weight,
                "composition": composition,
                **evaluation,
                "projected_oof_rank_auc_delta": projected_evaluation["oof_rank_auc_delta"],
                "projected_fold_rank_auc_delta": projected_evaluation["fold_rank_auc_delta"],
                "projected_rank_selected": projected_evaluation["rank_selected"],
                "heldout_rank_auc_delta": (
                    _classification_rank_score(encoded_test, heldout) - heldout_base_rank
                ),
                "projected_heldout_rank_auc_delta": (
                    _classification_rank_score(encoded_test, heldout_projected) - heldout_base_rank
                ),
                "heldout_accuracy_delta": (
                    float(np.mean(heldout.argmax(axis=1) == encoded_test)) - heldout_base_accuracy
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
    parser.add_argument("--out", default="results/tabpvn_wide_affine_probe.csv")
    args = parser.parse_args()
    cases = _parse_cases(args.cases)
    datasets = {
        dataset.name: dataset
        for dataset in tabarena_suite(
            size="le10k",
            dataset_names=[name for name, _fold in cases],
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
            f"{record['dataset']} fold {record['fold']} {record['composition']}: "
            f"OOF={record['projected_oof_rank_auc_delta']:+.6f} "
            f"folds={record['projected_fold_rank_auc_delta']} "
            f"selected={record['projected_rank_selected']} "
            f"heldout={record['projected_heldout_rank_auc_delta']:+.6f}"
        )


if __name__ == "__main__":
    main()
