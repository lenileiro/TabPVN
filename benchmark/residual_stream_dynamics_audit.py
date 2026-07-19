"""Audit one TabArena fold's additive residual trajectory.

This probe runs only the phase-adaptive TabPVN candidate.  It records verifier
dynamics from the model's existing fit, so no second evidence model is trained.

Example:
    uv run python -m benchmark.residual_stream_dynamics_audit \
      --dataset maternal_health_risk --fold 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from benchmark.datasets import tabarena_suite
from benchmark.experiments.models import build
from tabpvn.residual_dynamics import summarize_dynamics


def _take_rows(table, rows):
    return table.iloc[rows] if hasattr(table, "iloc") else table[rows]


def _score(target: np.ndarray, probability: np.ndarray) -> float:
    if probability.shape[1] == 2:
        return float(roc_auc_score(target, probability[:, 1]))
    return float(roc_auc_score(target, probability, multi_class="ovo", average="macro"))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit(dataset_name: str, fold: int, model_name: str = "tabpvn_adaptive_hard_pair") -> dict[str, object]:
    dataset = tabarena_suite(size="all", dataset_names=[dataset_name])[0]
    if dataset.task != "classification":
        raise ValueError("residual-stream hard-pair audit requires a classification task")
    if dataset.splits is None or not 0 <= fold < len(dataset.splits):
        raise ValueError(f"fold must select one of {len(dataset.splits or ())} official folds")
    train, test = dataset.splits[fold]
    estimator = build(model_name, dataset.task)
    started = time.perf_counter()
    estimator.fit(_take_rows(dataset.X, train), dataset.y[train])
    fit_seconds = time.perf_counter() - started
    test_features = _take_rows(dataset.X, test)
    probability = estimator.predict_proba(test_features)
    predictor = estimator._pred
    encoded_test = estimator._X(test_features)
    booster_scores = predictor._scores(encoded_test)
    raw_booster_probability = np.exp(booster_scores - booster_scores.max(axis=1, keepdims=True))
    raw_booster_probability /= raw_booster_probability.sum(axis=1, keepdims=True)
    calibrated_scores = booster_scores / getattr(estimator, "_temp", 1.0)
    calibrated_booster_probability = np.exp(calibrated_scores - calibrated_scores.max(axis=1, keepdims=True))
    calibrated_booster_probability /= calibrated_booster_probability.sum(axis=1, keepdims=True)
    records = list(predictor.residual_dynamics_ or ())
    selected_rounds = len(predictor.trees_) // max(1, len(predictor.classes_))
    selected_records = records[:selected_rounds]
    return {
        "dataset": dataset.name,
        "model": model_name,
        "fold": int(fold),
        "score": _score(dataset.y[test], probability),
        "raw_booster_score": _score(dataset.y[test], raw_booster_probability),
        "calibrated_booster_score": _score(dataset.y[test], calibrated_booster_probability),
        "fit_seconds": float(fit_seconds),
        "source_rows": int(len(dataset.y)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "selected_rounds": int(selected_rounds),
        "tree_count": int(len(predictor.trees_)),
        "boost_config": _jsonable(estimator.boost_),
        "interaction_features": list(getattr(estimator, "interaction_features_", ())),
        "temperature": float(getattr(estimator, "_temp", 1.0)),
        "probability_members": {
            "no_signal_prior": getattr(estimator, "_no_signal_prior", None) is not None,
            "smooth": getattr(estimator, "_smooth", None) is not None,
            "category_memory": getattr(estimator, "_category_memory", None) is not None,
            "proof_path_memory": getattr(estimator, "_proof_path_memory", None) is not None,
            "category_posterior": getattr(estimator, "_category_posterior", None) is not None,
            "numeric_interval_rank": getattr(estimator, "_numeric_interval_permission", None)
            == "decision_and_rank",
            "affine_rank": getattr(estimator, "_affine_rank_permission", None),
            "prior_rank_strength": float(getattr(estimator, "_prior_rank_strength", 0.0)),
        },
        "deployed_pair_growth_schedule": _jsonable(getattr(predictor, "pair_growth_schedule_", ())),
        "summary": summarize_dynamics(selected_records),
        "attempted_summary": summarize_dynamics(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="maternal_health_risk")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=("tabpvn_adaptive_hard_pair", "tabpvn_hard_pair_best_first"),
        default="tabpvn_adaptive_hard_pair",
    )
    parser.add_argument("--show-records", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/tabpvn_residual_stream_dynamics_audit.json"),
    )
    args = parser.parse_args()
    result = audit(args.dataset, args.fold, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary = result["summary"]
    print(
        f"{result['dataset']} fold={result['fold']} score={result['score']:.6f} "
        f"fit={result['fit_seconds']:.2f}s selected_rounds={result['selected_rounds']} "
        f"hard_pair_rounds={summary['hard_pair_rounds']}"
    )
    if args.show_records:
        for record in result["records"]:
            print(
                f"round={record['round']:>3} growth={record['growth_phase']:<10} "
                f"pair={record['hard_pair']} next={record['next_phase']:<10} "
                f"d_loss={record['pair_loss_change']:+.6f} "
                f"rank={record['effective_rank']:.3f} innovation={record['innovation_ratio']:.3f} "
                f"blocked={','.join(record['blocked_by']) or '-'}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
