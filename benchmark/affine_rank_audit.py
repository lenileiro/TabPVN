"""Targeted direct ablation for the OOF-gated global affine read.

Each task is fitted once.  The official holdout is scored with the deployed
surface and with only the affine read bypassed, isolating its effect without a
second booster fit.  TabPFN is deliberately outside this inner audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from benchmark.datasets import tabarena_suite
from tabpvn import TabPVN, TargetAttestation

DEFAULT_CASES = (
    ("credit-g", 0),
    ("qsar-biodeg", 0),
    ("coil2000_insurance_policies", 1),
    ("maternal_health_risk", 0),
)


def _rank_score(target, probability) -> float:
    if probability.shape[1] == 2:
        return float(roc_auc_score(target, probability[:, 1]))
    return float(roc_auc_score(target, probability, multi_class="ovo", average="macro"))


def _plain(value):
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def audit(dataset, fold_index: int) -> tuple[dict[str, object], dict[str, object] | None]:
    train, test = dataset.splits[fold_index]
    model = TabPVN(seed=0, task="classification")
    started = time.perf_counter()
    model.fit(dataset.X.iloc[train], dataset.y[train])
    fit_seconds = time.perf_counter() - started

    encoded = model._X(dataset.X.iloc[test])
    started = time.perf_counter()
    deployed = model.predict_proba(dataset.X.iloc[test])
    inference_seconds = time.perf_counter() - started
    without_affine = model._blended_proba(encoded, include_affine_rank=False)
    baseline_decision_probability = model._blended_proba(
        encoded,
        include_prior_rank=False,
        include_interval_rank=False,
        include_affine_rank=False,
    )
    if model._numeric_interval is not None:
        baseline_decision_probability = model._numeric_interval.combine(
            baseline_decision_probability,
            encoded,
            model._numeric_interval_w,
        )
    classes = np.asarray(model.classes_)
    baseline_prediction = classes[baseline_decision_probability.argmax(1)]
    deployed_prediction = model.predict(dataset.X.iloc[test])
    target = np.asarray(dataset.y[test])
    baseline_correct = baseline_prediction == target
    deployed_correct = deployed_prediction == target
    heldout_wins = int(np.sum(deployed_correct & ~baseline_correct))
    heldout_losses = int(np.sum(~deployed_correct & baseline_correct))
    corrected = np.flatnonzero(deployed_correct & ~baseline_correct)
    proof_sample = None
    if len(corrected):
        local_row = int(corrected[0])
        subject = f"{dataset.name}/fold-{fold_index}/row-{int(test[local_row])}"
        response = model.proof(
            dataset.X.iloc[test],
            local_row,
            attestation=TargetAttestation(
                value=_plain(target[local_row]),
                source="TabArena official held-out target",
                subject=subject,
            ),
        )
        proof_sample = {
            "dataset": dataset.name,
            "fold": fold_index,
            "heldout_row": int(test[local_row]),
            "target": _plain(target[local_row]),
            "baseline_prediction": _plain(baseline_prediction[local_row]),
            "prediction": _plain(deployed_prediction[local_row]),
            "affine_evidence": model.affine_evidence(dataset.X.iloc[test], local_row),
            "proof_response": response,
            "reason": model.reason(dataset.X.iloc[test], local_row),
        }
    gate = model.affine_rank_report_[-1]
    baseline_score = _rank_score(dataset.y[test], without_affine)
    deployed_score = _rank_score(dataset.y[test], deployed)
    return {
        "dataset": dataset.name,
        "fold": fold_index,
        "train_rows": len(train),
        "test_rows": len(test),
        "selected": model._affine_rank is not None,
        "permission": gate.get("permission", ""),
        "composition": gate.get("composition", ""),
        "weight": gate.get("weight", ""),
        "gate_reason": gate.get("reason", ""),
        "oof_baseline_auc": model.affine_rank_report_[0].get("mean_score", ""),
        "oof_affine_auc": gate.get("mean_score", ""),
        "oof_auc_delta": gate.get("rank_auc_delta", ""),
        "oof_fold_auc_delta": gate.get("fold_auc_delta", ""),
        "oof_log_loss_delta": gate.get("log_loss_delta", ""),
        "oof_baseline_accuracy": gate.get("baseline_oof_accuracy", ""),
        "oof_affine_accuracy": gate.get("decision_oof_accuracy", ""),
        "oof_accuracy_delta": gate.get("accuracy_gain", ""),
        "oof_fold_accuracy_delta": gate.get("fold_accuracy_delta", ""),
        "oof_paired_z": gate.get("paired_z", ""),
        "heldout_baseline_auc": baseline_score,
        "heldout_affine_auc": deployed_score,
        "heldout_auc_delta": deployed_score - baseline_score,
        "heldout_baseline_accuracy": float(np.mean(baseline_prediction == dataset.y[test])),
        "heldout_affine_accuracy": float(np.mean(deployed_prediction == dataset.y[test])),
        "heldout_accuracy_delta": float(
            np.mean(deployed_prediction == dataset.y[test]) - np.mean(baseline_prediction == dataset.y[test])
        ),
        "heldout_wins": heldout_wins,
        "heldout_losses": heldout_losses,
        "proof_row": "" if proof_sample is None else proof_sample["heldout_row"],
        "proof_verified": (
            "" if proof_sample is None else TabPVN.check_proof(proof_sample["proof_response"])
        ),
        "class_preserved": bool(np.array_equal(without_affine.argmax(1), deployed.argmax(1))),
        "kernel_certified": float(model.certify(dataset.X.iloc[test], sample=min(256, len(test)))),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
    }, proof_sample


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
    parser.add_argument("--out", default="results/tabpvn_affine_rank_audit.csv")
    parser.add_argument("--proof-out", help="Optional JSON path for corrected-row affine proofs")
    args = parser.parse_args()
    cases = _parse_cases(args.cases)
    datasets = {
        dataset.name: dataset
        for dataset in tabarena_suite(
            size="le10k",
            dataset_names=[name for name, _fold in cases],
        )
    }
    audited = [audit(datasets[name], fold) for name, fold in cases]
    records = [record for record, _proof in audited]
    proofs = [proof for _record, proof in audited if proof is not None]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if args.proof_out:
        proof_output = Path(args.proof_out)
        proof_output.parent.mkdir(parents=True, exist_ok=True)
        with proof_output.open("w") as handle:
            json.dump(proofs, handle, indent=2)
    for record in records:
        print(
            f"{record['dataset']} fold {record['fold']}: selected={record['selected']} "
            f"OOF={record['oof_auc_delta']} "
            f"heldout={record['heldout_baseline_auc']:.6f}"
            f"->{record['heldout_affine_auc']:.6f} "
            f"accuracy={record['heldout_baseline_accuracy']:.6f}"
            f"->{record['heldout_affine_accuracy']:.6f} "
            f"permission={record['permission']} "
            f"composition={record['composition']} "
            f"certify={record['kernel_certified']:.1f} "
            f"fit={record['fit_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
