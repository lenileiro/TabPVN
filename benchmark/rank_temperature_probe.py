"""Probe a final class-preserving temperature on multiclass rank probabilities.

The booster temperature is selected for log-loss before TabPVN's admitted
probability members.  This research-only audit asks whether a fixed normalized
power transform of the final public probability stack improves macro OVO AUC.
The transform preserves every argmax and adds no learner or serving state.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark.affine_composition_probe import _evaluate
from benchmark.datasets import tabarena_suite
from tabpvn import TabPVN
from tabpvn.base import _classification_rank_score, _preserve_certified_class

DEFAULT_CASES = tuple(("maternal_health_risk", fold) for fold in range(3))
TEMPERATURES = (0.5, 0.75, 1.25, 2.0)


def _rank_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Normalize ``probability ** (1 / temperature)`` row by row."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    probability = np.asarray(probability, dtype=float)
    log_probability = np.log(np.clip(probability, np.finfo(float).tiny, 1.0))
    shifted = log_probability / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    transformed = np.exp(shifted)
    transformed /= transformed.sum(axis=1, keepdims=True)
    return _preserve_certified_class(probability, transformed)


class _RankTemperatureProbeTabPVN(TabPVN):
    """Capture the final OOF rank surface without changing production gates."""

    def _multiclass_prior_rank_gate(self, y, precomp):
        selected = super()._multiclass_prior_rank_gate(y, precomp)
        probability = self._prior_rank_oof_proba
        if probability is None:
            probability = self._oof_probability_stack(precomp)
        self.rank_temperature_probe_ = {
            "probability": np.asarray(probability, dtype=float).copy(),
            "splits": tuple(precomp["splits"]),
            "evidence_rows": np.asarray(precomp["evidence_rows"], dtype=np.int64),
            "prior_rank_selected": bool(selected),
        }
        return selected


def audit(dataset, fold_index: int) -> list[dict[str, object]]:
    train, test = dataset.splits[fold_index]
    model = _RankTemperatureProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], dataset.y[train])

    probe = model.rank_temperature_probe_
    classes = np.asarray(model.classes_)
    class_index = {value: index for index, value in enumerate(classes)}
    oof_target = np.asarray([class_index[value] for value in dataset.y[train]], dtype=np.int32)
    heldout_target = np.asarray([class_index[value] for value in dataset.y[test]], dtype=np.int32)
    oof_base = probe["probability"]
    heldout_base = model.predict_proba(dataset.X.iloc[test])
    heldout_base_rank = _classification_rank_score(heldout_target, heldout_base)
    heldout_base_prediction = heldout_base.argmax(axis=1)

    records: list[dict[str, object]] = []
    for temperature in TEMPERATURES:
        oof_candidate = _rank_temperature(oof_base, temperature)
        evaluation = _evaluate(
            oof_target,
            oof_base,
            oof_candidate,
            probe["splits"],
            probe["evidence_rows"],
        )
        heldout_candidate = _rank_temperature(heldout_base, temperature)
        class_preserved = bool(
            np.array_equal(oof_base.argmax(axis=1), oof_candidate.argmax(axis=1))
            and np.array_equal(heldout_base_prediction, heldout_candidate.argmax(axis=1))
        )
        records.append(
            {
                "dataset": dataset.name,
                "fold": fold_index,
                "temperature": temperature,
                "rows": len(train),
                "features": model.n_input_features_,
                "classes": len(classes),
                "prior_rank_selected": probe["prior_rank_selected"],
                "class_preserved": class_preserved,
                **evaluation,
                "heldout_rank_auc": _classification_rank_score(heldout_target, heldout_candidate),
                "heldout_rank_auc_delta": (
                    _classification_rank_score(heldout_target, heldout_candidate) - heldout_base_rank
                ),
                "heldout_accuracy_delta": float(
                    np.mean(heldout_candidate.argmax(axis=1) == heldout_target)
                    - np.mean(heldout_base_prediction == heldout_target)
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
    parser.add_argument("--out", default="results/tabpvn_rank_temperature_probe.csv")
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
            f"{record['dataset']} fold {record['fold']} T={record['temperature']}: "
            f"OOF={record['oof_rank_auc_delta']:+.6f} "
            f"folds={record['fold_rank_auc_delta']} "
            f"heldout={record['heldout_rank_auc_delta']:+.6f} "
            f"class_preserved={record['class_preserved']}"
        )


if __name__ == "__main__":
    main()
