"""Probe transparent affine/base composition rules on TabPVN's shared OOF stack.

The production gate receives the exact probability stack after every earlier
admitted proposer.  This audit captures that stack in-place, so alternative
compositions require only cheap affine fits and never retrain the booster.
Nothing in this module changes the deployed estimator.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark.datasets import tabarena_suite
from tabpvn import TabPVN
from tabpvn.base import (
    _AFFINE_DECISION_MAX_RANK_REGRESSION,
    _AFFINE_DECISION_MIN_ACCURACY_GAIN,
    _AFFINE_DECISION_MIN_PAIRED_Z,
    _AFFINE_RANK_INVERSE_REGULARIZATION,
    _AFFINE_RANK_MIN_FOLD_GAIN,
    _AFFINE_RANK_MIN_GAIN,
    _affine_rank_weight,
    _classification_rank_score,
    _preserve_certified_class,
)
from tabpvn.proposers.affine import AffineLogitRead

DEFAULT_CASES = (
    ("credit-g", 0),
    ("credit-g", 1),
    ("credit-g", 2),
    ("qsar-biodeg", 0),
    ("coil2000_insurance_policies", 1),
    ("taiwanese_bankruptcy_prediction", 0),
)


def _normalize_log_probability(log_probability: np.ndarray) -> np.ndarray:
    shifted = log_probability - log_probability.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def _compositions(
    base: np.ndarray,
    affine: np.ndarray,
    prior: np.ndarray,
    weight: float,
) -> dict[str, np.ndarray]:
    tiny = np.finfo(float).tiny
    log_base = np.log(np.clip(base, tiny, 1.0))
    log_affine = np.log(np.clip(affine, tiny, 1.0))
    return {
        "arithmetic": AffineLogitRead.combine(base, affine, weight),
        "geometric": _normalize_log_probability((1.0 - weight) * log_base + weight * log_affine),
        "product": _normalize_log_probability(log_base + weight * log_affine),
        "prior_ratio": AffineLogitRead.combine(
            base,
            affine,
            weight,
            composition="prior_ratio",
            prior=prior,
        ),
    }


class _CompositionProbeTabPVN(TabPVN):
    """Capture affine fold reads at the production gate without changing it."""

    def _global_affine_rank_gate(self, X, y, precomp):
        classes = np.asarray(self._pred.classes_)
        labels = np.asarray(y)
        base = self._oof_probability_stack(precomp).copy()
        class_index = {value: index for index, value in enumerate(classes)}
        encoded = np.asarray([class_index[value] for value in labels], dtype=np.int32)
        full_counts = np.bincount(encoded, minlength=len(classes)).astype(float)
        deployment_prior = full_counts / full_counts.sum()
        affine = base.copy()
        oof_prior = np.tile(deployment_prior, (len(labels), 1))
        for train, valid in precomp["splits"]:
            member = AffineLogitRead(
                inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
                seed=self.seed,
            ).fit(X[train], labels[train], classes=classes)
            affine[valid] = member.proba(X[valid])
            fold_counts = np.bincount(encoded[train], minlength=len(classes)).astype(float)
            oof_prior[valid] = fold_counts / fold_counts.sum()
        self.affine_composition_probe_ = {
            "base": base,
            "affine": affine,
            "oof_prior": oof_prior,
            "deployment_prior": deployment_prior,
            "splits": tuple(precomp["splits"]),
            "evidence_rows": np.asarray(precomp["evidence_rows"], dtype=np.int64),
        }
        return super()._global_affine_rank_gate(X, y, precomp)


def _log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    return float(-np.log(np.clip(probability[np.arange(len(target)), target], 1e-300, 1.0)).mean())


def _evaluate(
    target: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    splits,
    rows: np.ndarray,
) -> dict[str, object]:
    base_prediction = base.argmax(1)
    candidate_prediction = candidate.argmax(1)
    base_correct = base_prediction == target
    candidate_correct = candidate_prediction == target
    base_accuracy = float(base_correct[rows].mean())
    accuracy = float(candidate_correct[rows].mean())
    wins = int(np.sum(candidate_correct[rows] & ~base_correct[rows]))
    losses = int(np.sum(~candidate_correct[rows] & base_correct[rows]))
    paired_z = (wins - losses) / np.sqrt(max(wins + losses, 1))
    fold_net_wins = []
    fold_accuracy_delta = []
    fold_rank_delta = []
    base_rank = _classification_rank_score(target[rows], base[rows])
    rank = _classification_rank_score(target[rows], candidate[rows])
    for _train, valid in splits:
        fold_wins = int(np.sum(candidate_correct[valid] & ~base_correct[valid]))
        fold_losses = int(np.sum(~candidate_correct[valid] & base_correct[valid]))
        fold_net_wins.append(fold_wins - fold_losses)
        fold_accuracy_delta.append((fold_wins - fold_losses) / len(valid))
        fold_rank_delta.append(
            _classification_rank_score(target[valid], candidate[valid])
            - _classification_rank_score(target[valid], base[valid])
        )
    base_loss = _log_loss(target[rows], base[rows])
    loss = _log_loss(target[rows], candidate[rows])
    majority = sum(net > 0 for net in fold_net_wins) >= (len(splits) + 1) // 2
    decision_selected = bool(
        accuracy - base_accuracy >= _AFFINE_DECISION_MIN_ACCURACY_GAIN
        and wins > losses
        and paired_z >= _AFFINE_DECISION_MIN_PAIRED_Z
        and all(net >= 0 for net in fold_net_wins)
        and majority
        and rank >= base_rank - _AFFINE_DECISION_MAX_RANK_REGRESSION
        and loss <= base_loss + 1.0 / np.sqrt(len(rows))
    )
    rank_selected = bool(
        rank - base_rank >= _AFFINE_RANK_MIN_GAIN
        and all(delta >= _AFFINE_RANK_MIN_FOLD_GAIN for delta in fold_rank_delta)
    )
    return {
        "oof_accuracy": accuracy,
        "oof_accuracy_delta": accuracy - base_accuracy,
        "oof_wins": wins,
        "oof_losses": losses,
        "oof_paired_z": float(paired_z),
        "fold_net_wins": fold_net_wins,
        "fold_accuracy_delta": fold_accuracy_delta,
        "oof_rank_auc": rank,
        "oof_rank_auc_delta": rank - base_rank,
        "fold_rank_auc_delta": fold_rank_delta,
        "oof_log_loss_delta": loss - base_loss,
        "decision_selected": decision_selected,
        "rank_selected": rank_selected,
    }


def audit(dataset, fold_index: int) -> list[dict[str, object]]:
    train, test = dataset.splits[fold_index]
    model = _CompositionProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], dataset.y[train])
    probe = model.affine_composition_probe_
    classes = np.asarray(model.classes_)
    class_index = {value: index for index, value in enumerate(classes)}
    oof_target = np.asarray([class_index[value] for value in dataset.y[train]], dtype=np.int32)
    weight = _affine_rank_weight(len(train))

    encoded_test = model._X(dataset.X.iloc[test])
    heldout_base = model._blended_proba(
        encoded_test,
        include_prior_rank=False,
        include_interval_rank=False,
        include_affine_rank=False,
    )
    affine = getattr(model, "_affine_rank", None)
    if affine is None:
        affine = AffineLogitRead(
            inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
            seed=model.seed,
        ).fit(model._X(dataset.X.iloc[train]), dataset.y[train], classes=classes)
    heldout_affine = affine.proba(encoded_test)
    heldout_target = np.asarray([class_index[value] for value in dataset.y[test]], dtype=np.int32)
    heldout_base_accuracy = float(np.mean(heldout_base.argmax(1) == heldout_target))
    heldout_base_rank = _classification_rank_score(heldout_target, heldout_base)

    records = []
    oof_candidates = _compositions(probe["base"], probe["affine"], probe["oof_prior"], weight)
    heldout_candidates = _compositions(
        heldout_base,
        heldout_affine,
        probe["deployment_prior"],
        weight,
    )
    for name, candidate in oof_candidates.items():
        evaluation = _evaluate(
            oof_target,
            probe["base"],
            candidate,
            probe["splits"],
            probe["evidence_rows"],
        )
        projected = _preserve_certified_class(probe["base"], candidate)
        projected_evaluation = _evaluate(
            oof_target,
            probe["base"],
            projected,
            probe["splits"],
            probe["evidence_rows"],
        )
        heldout = heldout_candidates[name]
        records.append(
            {
                "dataset": dataset.name,
                "fold": fold_index,
                "composition": name,
                "weight": weight,
                **evaluation,
                "projected_oof_rank_auc_delta": projected_evaluation["oof_rank_auc_delta"],
                "projected_fold_rank_auc_delta": projected_evaluation["fold_rank_auc_delta"],
                "projected_rank_selected": projected_evaluation["rank_selected"],
                "heldout_accuracy_delta": (
                    float(np.mean(heldout.argmax(1) == heldout_target)) - heldout_base_accuracy
                ),
                "heldout_rank_auc_delta": (
                    _classification_rank_score(heldout_target, heldout) - heldout_base_rank
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
    parser.add_argument("--out", default="results/tabpvn_affine_composition_probe.csv")
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
            f"OOF acc={record['oof_accuracy_delta']:+.6f} "
            f"rank={record['oof_rank_auc_delta']:+.6f} "
            f"decision={record['decision_selected']} rank_gate={record['rank_selected']} "
            f"heldout acc={record['heldout_accuracy_delta']:+.6f} "
            f"rank={record['heldout_rank_auc_delta']:+.6f}"
        )


if __name__ == "__main__":
    main()
