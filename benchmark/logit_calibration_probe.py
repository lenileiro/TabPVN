"""Probe an explicit cross-fitted affine read of final log probabilities.

This is matrix scaling, not another feature learner.  A strongly regularized
multinomial affine map reads only TabPVN's selected log-probability vector.  It
is cross-fitted over the shared OOF stack and projected to preserve every
incumbent class before rank scoring.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark.affine_composition_probe import _evaluate
from benchmark.datasets import tabarena_suite
from benchmark.rank_temperature_probe import _RankTemperatureProbeTabPVN
from benchmark.weak_pair_boost_probe import _pair_rank, _weakest_pair
from tabpvn.base import (
    _AFFINE_RANK_INVERSE_REGULARIZATION,
    _classification_rank_score,
    _preserve_certified_class,
)
from tabpvn.proposers.affine import AffineLogitRead

DEFAULT_CASES = tuple(("maternal_health_risk", fold) for fold in range(3))
COMPOSITIONS = (
    ("arithmetic_quarter", "arithmetic", 0.25),
    ("arithmetic_half", "arithmetic", 0.5),
    ("geometric_quarter", "geometric", 0.25),
    ("geometric_half", "geometric", 0.5),
    ("direct", "arithmetic", 1.0),
)


def _log_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    logged = np.log(np.clip(probability, np.finfo(float).tiny, 1.0))
    return logged - logged.mean(axis=1, keepdims=True)


def _fit_read(probability, target, rows, n_classes):
    return AffineLogitRead(
        inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
        seed=0,
    ).fit(
        _log_probability(probability[rows]),
        target[rows],
        classes=np.arange(n_classes),
    )


def _compose(base, calibrated, composition, weight):
    if composition == "arithmetic":
        candidate = (1.0 - weight) * base + weight * calibrated
    elif composition == "geometric":
        tiny = np.finfo(float).tiny
        logged = (1.0 - weight) * np.log(np.clip(base, tiny, 1.0)) + weight * np.log(
            np.clip(calibrated, tiny, 1.0)
        )
        logged -= logged.max(axis=1, keepdims=True)
        candidate = np.exp(logged)
        candidate /= candidate.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unsupported calibration composition: {composition}")
    return _preserve_certified_class(base, candidate)


def audit(dataset, fold_index: int) -> list[dict[str, object]]:
    train, test = dataset.splits[fold_index]
    model = _RankTemperatureProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], dataset.y[train])
    probe = model.rank_temperature_probe_

    classes = np.asarray(model.classes_)
    class_index = {label: index for index, label in enumerate(classes)}
    oof_target = np.asarray([class_index[label] for label in dataset.y[train]], dtype=np.int32)
    heldout_target = np.asarray([class_index[label] for label in dataset.y[test]], dtype=np.int32)
    base = np.asarray(probe["probability"], dtype=float)
    evidence_rows = np.asarray(probe["evidence_rows"], dtype=np.int64)
    splits = tuple(probe["splits"])
    calibrated = base.copy()

    for _fold_train, fold_valid in splits:
        calibration_rows = np.setdiff1d(
            evidence_rows,
            np.asarray(fold_valid, dtype=np.int64),
            assume_unique=False,
        )
        read = _fit_read(base, oof_target, calibration_rows, len(classes))
        calibrated[fold_valid] = read.proba(_log_probability(base[fold_valid]))

    full_read = _fit_read(base, oof_target, evidence_rows, len(classes))
    heldout_base = model.predict_proba(dataset.X.iloc[test])
    heldout_calibrated = full_read.proba(_log_probability(heldout_base))
    heldout_base_rank = _classification_rank_score(heldout_target, heldout_base)
    weak_pair = _weakest_pair(oof_target, base, evidence_rows, len(classes))

    records: list[dict[str, object]] = []
    for name, composition, weight in COMPOSITIONS:
        candidate = _compose(base, calibrated, composition, weight)
        evaluation = _evaluate(
            oof_target,
            base,
            candidate,
            splits,
            evidence_rows,
        )
        heldout = _compose(
            heldout_base,
            heldout_calibrated,
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
                "classes": len(classes),
                "inverse_regularization": _AFFINE_RANK_INVERSE_REGULARIZATION,
                "class_preserved": bool(
                    np.array_equal(base.argmax(axis=1), candidate.argmax(axis=1))
                    and np.array_equal(heldout_base.argmax(axis=1), heldout.argmax(axis=1))
                ),
                "weak_pair": list(weak_pair),
                **evaluation,
                "heldout_rank_auc": _classification_rank_score(heldout_target, heldout),
                "heldout_rank_auc_delta": (
                    _classification_rank_score(heldout_target, heldout) - heldout_base_rank
                ),
                "heldout_pair_rank_delta": (
                    _pair_rank(heldout_target, heldout, weak_pair)
                    - _pair_rank(heldout_target, heldout_base, weak_pair)
                ),
                "coef": full_read.coef_.tolist(),
                "intercept": full_read.intercept_.tolist(),
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
    parser.add_argument("--out", default="results/tabpvn_logit_calibration_probe.csv")
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
            f"OOF={record['oof_rank_auc_delta']:+.6f} "
            f"folds={record['fold_rank_auc_delta']} "
            f"heldout={record['heldout_rank_auc_delta']:+.6f} "
            f"pair={record['heldout_pair_rank_delta']:+.6f}"
        )


if __name__ == "__main__":
    main()
