"""Benchmark runner.

Evaluates a set of models across a dataset suite using repeated stratified splits,
records per-(dataset, model) scores, and produces an average-rank leaderboard —
the standard way to compare tabular models across heterogeneous datasets (lower
average rank = better; this is what critical-difference diagrams are built on).

Usage:
    uv run python -m benchmark.experiments.run --suite sklearn --models rf,linear
    uv run python -m benchmark.experiments.run --suite sklearn --models tabpvn,tabpfn --splits 3
    uv run python -m benchmark.experiments.run --suite openml --models rf,xgboost --out results/run.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from benchmark import datasets as ds_mod
from benchmark.experiments import models as model_mod

_FOLD_RECORD_FIELDS = (
    "dataset",
    "model",
    "fold",
    "task",
    "metric",
    "score",
    "fit_seconds",
    "inference_seconds",
    "status",
    "error",
    "implementation",
)


def _fold_record_key(record):
    return (
        str(record["dataset"]),
        str(record["model"]),
        int(record["fold"]),
        str(record["metric"]),
    )


def _fold_record_path(output: str | Path) -> Path:
    output = Path(output)
    suffix = output.suffix or ".csv"
    return output.with_name(f"{output.stem}.folds{suffix}")


def _read_fold_records(path: str | Path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {_fold_record_key(row): row for row in rows}


def _write_fold_records(path: str | Path, records):
    """Atomically persist fold-level progress so an interrupted suite can resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    rows = sorted(records.values(), key=_fold_record_key)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FOLD_RECORD_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in _FOLD_RECORD_FIELDS} for row in rows)
    os.replace(temporary, path)


def _implementation_fingerprint(model_name: str) -> str:
    """Fingerprint benchmark semantics and the model implementation.

    Resume checkpoints without this exact fingerprint are stale by definition.
    This prevents a successful historical fold from silently standing in for
    changed source code, which previously made targeted reruns disagree with an
    apparently current aggregate artifact.
    """
    root = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__).resolve(),
        root / "benchmark" / "datasets.py",
        root / "benchmark" / "experiments" / "models.py",
    ]
    if model_name.startswith("tabpvn"):
        files.extend(path for path in sorted((root / "tabpvn").rglob("*.py")) if "tests" not in path.parts)
        files.extend(path for path in sorted((root / "core").rglob("*.py")) if "tests" not in path.parts)
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    digest.update(f"python={sys.version_info[:3]}".encode())
    for distribution in ("numpy", "scikit-learn", model_name):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        digest.update(f"{distribution}={version}".encode())
    for path in files:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _predict_and_score(
    task: str,
    estimator,
    X_test,
    y_test,
    classification_metric: str = "roc_auc",
) -> tuple[float, float]:
    """Return (score, predict_seconds).

    Score is higher-is-better for both tasks: selected ranking metric (clf), negative RMSE (reg).
    predict_seconds is the wall-clock of the prediction call only — the meaningful
    *inference cost* axis, since ICL models (TabPFN et al.) do all their work here,
    not in fit(). This is what we Pareto against accuracy.
    """
    t0 = time.perf_counter()
    if task == "classification":
        proba = estimator.predict_proba(X_test)
    else:
        pred = estimator.predict(X_test)
    predict_s = time.perf_counter() - t0

    if task == "classification":
        if classification_metric == "average_precision":
            if proba.shape[1] != 2:
                raise ValueError("average_precision benchmark mode currently requires a binary task")
            score = float(average_precision_score(y_test, proba[:, 1]))
        elif proba.shape[1] == 2:
            score = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            score = float(roc_auc_score(y_test, proba, multi_class="ovo", average="macro"))
    else:
        score = -float(mean_squared_error(y_test, pred) ** 0.5)
    return score, predict_s


def _take_rows(X, indices):
    """Index ndarray and pandas feature tables without changing their schema."""
    return X.iloc[indices] if hasattr(X, "iloc") else X[indices]


def _splits_for_dataset(ds: ds_mod.Dataset, splits: int, seed: int):
    """Return official task folds when available, otherwise reproducible random splits."""
    if ds.splits is not None:
        return ds.splits if splits <= 0 else ds.splits[:splits]
    count = 3 if splits <= 0 else splits
    strat = ds.y if ds.task == "classification" else None
    return tuple(
        train_test_split(np.arange(len(ds.y)), test_size=0.25, random_state=seed + s, stratify=strat)
        for s in range(count)
    )


def evaluate(
    model_names: list[str],
    suite: list[ds_mod.Dataset],
    splits: int,
    seed: int,
    fold_indices: tuple[int, ...] | None = None,
    classification_metric: str = "roc_auc",
    completed_records=None,
    record_callback=None,
    implementation_fingerprints=None,
):
    # results[dataset][model] = mean score over splits (or None if unavailable/failed)
    results: dict[str, dict[str, float | None]] = {}
    # observations[dataset][model][split] = score. Keeping split identity makes
    # pairwise deltas robust when a backend fails only on some task folds.
    observations: dict[str, dict[str, dict[int, float]]] = {}
    # timing[model] = {"fit": [...], "predict": [...]} — seconds summed over splits/datasets
    timing: dict[str, dict[str, list[float]]] = {m: {"fit": [], "predict": []} for m in model_names}
    completed_records = completed_records or {}
    implementation_fingerprints = implementation_fingerprints or {
        name: _implementation_fingerprint(name) for name in model_names
    }

    for ds in suite:
        results[ds.name] = {}
        observations[ds.name] = {}
        # Targeted promotion checks index the original official folds, rather
        # than the prefix selected by ``--splits``.
        dataset_splits = _splits_for_dataset(ds, 0 if fold_indices is not None else splits, seed)
        if fold_indices is None:
            selected_splits = tuple(enumerate(dataset_splits))
        else:
            unavailable = [index for index in fold_indices if index >= len(dataset_splits)]
            if unavailable:
                raise ValueError(
                    f"{ds.name} has {len(dataset_splits)} available splits; cannot select {unavailable}"
                )
            selected_splits = tuple((index, dataset_splits[index]) for index in fold_indices)
        for name in model_names:
            per_split: list[float] = []
            per_split_map: dict[int, float] = {}
            fit_s = predict_s = 0.0
            try:
                for split_id, (train, test) in selected_splits:
                    metric = classification_metric if ds.task == "classification" else "neg_rmse"
                    key = (ds.name, name, int(split_id), metric)
                    cached = completed_records.get(key)
                    implementation = implementation_fingerprints[name]
                    if (
                        cached is not None
                        and cached.get("status") == "ok"
                        and cached.get("implementation") == implementation
                    ):
                        score = float(cached["score"])
                        fold_fit_s = float(cached.get("fit_seconds", 0.0) or 0.0)
                        fold_predict_s = float(cached.get("inference_seconds", 0.0) or 0.0)
                        fit_s += fold_fit_s
                        predict_s += fold_predict_s
                        per_split.append(score)
                        per_split_map[split_id] = score
                        print(f"  [resume] {ds.name} | {name} | fold={split_id}")
                        continue
                    if cached is not None and cached.get("status") == "ok":
                        print(f"  [stale] {ds.name} | {name} | fold={split_id}; rerunning")
                    Xtr, Xte = _take_rows(ds.X, train), _take_rows(ds.X, test)
                    ytr, yte = ds.y[train], ds.y[test]
                    fold_fit_s = fold_predict_s = 0.0
                    try:
                        est = model_mod.build(name, ds.task)
                        tf = time.perf_counter()
                        est.fit(Xtr, ytr)
                        fold_fit_s = time.perf_counter() - tf
                        score, fold_predict_s = _predict_and_score(
                            ds.task,
                            est,
                            Xte,
                            yte,
                            classification_metric=classification_metric,
                        )
                    except Exception as error:
                        if record_callback is not None:
                            record_callback(
                                {
                                    "dataset": ds.name,
                                    "model": name,
                                    "fold": int(split_id),
                                    "task": ds.task,
                                    "metric": metric,
                                    "score": "",
                                    "fit_seconds": f"{fold_fit_s:.9f}",
                                    "inference_seconds": f"{fold_predict_s:.9f}",
                                    "status": (
                                        "unavailable"
                                        if isinstance(error, model_mod.ModelUnavailable)
                                        else "failed"
                                    ),
                                    "error": f"{type(error).__name__}: {error}",
                                    "implementation": implementation,
                                }
                            )
                        raise
                    fit_s += fold_fit_s
                    predict_s += fold_predict_s
                    per_split.append(score)
                    per_split_map[split_id] = score
                    if record_callback is not None:
                        record_callback(
                            {
                                "dataset": ds.name,
                                "model": name,
                                "fold": int(split_id),
                                "task": ds.task,
                                "metric": metric,
                                "score": f"{score:.12g}",
                                "fit_seconds": f"{fold_fit_s:.9f}",
                                "inference_seconds": f"{fold_predict_s:.9f}",
                                "status": "ok",
                                "error": "",
                                "implementation": implementation,
                            }
                        )
                results[ds.name][name] = mean(per_split)
                observations[ds.name][name] = per_split_map
                timing[name]["fit"].append(fit_s)
                timing[name]["predict"].append(predict_s)
            except model_mod.ModelUnavailable as e:
                results[ds.name][name] = None
                print(f"  [skip] {name} on {ds.name}: {e}")
            except Exception as e:  # a model that errors on a dataset scores nothing
                results[ds.name][name] = None
                print(f"  [fail] {name} on {ds.name}: {type(e).__name__}: {e}")
            status = results[ds.name][name]
            if status is not None:
                print(f"  {ds.name:>16} | {name:>10} | score={status:+.4f} | infer={predict_s:6.3f}s")
    return results, timing, observations


def average_ranks(results: dict[str, dict[str, float | None]], model_names: list[str]):
    """Average rank per model across datasets where it produced a score (1 = best)."""
    ranks: dict[str, list[float]] = {m: [] for m in model_names}
    for _ds, scores in results.items():
        present = [(m, s) for m, s in scores.items() if s is not None]
        if len(present) < 2:
            continue
        present.sort(key=lambda kv: kv[1], reverse=True)  # higher score = rank 1
        # dense ranking with ties averaged
        for m, s in present:
            tied = [j for j, (_m, _s) in enumerate(present) if _s == s]
            ranks[m].append(1 + mean(tied))
    return {m: (mean(r) if r else float("nan")) for m, r in ranks.items()}


def paired_deltas(observations, model_names: list[str], reference: str):
    """Paired per-task score deltas and a task-level normal-approximation CI.

    We average repeated folds within each task before computing uncertainty. That
    keeps an easy task with ten repeats from dominating a hard task with three.
    """
    if reference not in model_names:
        return {}
    out = {}
    for name in model_names:
        if name == reference:
            continue
        deltas = []
        for scores in observations.values():
            ours, ref = scores.get(name, {}), scores.get(reference, {})
            shared = sorted(set(ours) & set(ref))
            if shared:
                deltas.append(mean(ours[i] - ref[i] for i in shared))
        if not deltas:
            continue
        delta = mean(deltas)
        se = stdev(deltas) / (len(deltas) ** 0.5) if len(deltas) > 1 else float("nan")
        out[name] = {
            "delta": delta,
            "ci_low": delta - 1.96 * se if np.isfinite(se) else float("nan"),
            "ci_high": delta + 1.96 * se if np.isfinite(se) else float("nan"),
            "wins": int(sum(d > 0 for d in deltas)),
            "tasks": len(deltas),
        }
    return out


def paired_relative_rmse_deltas(observations, model_names: list[str], reference: str):
    """Paired relative RMSE reductions; positive means lower error than reference."""
    if reference not in model_names:
        return {}
    out = {}
    for name in model_names:
        if name == reference:
            continue
        deltas = []
        for scores in observations.values():
            ours, ref = scores.get(name, {}), scores.get(reference, {})
            shared = sorted(set(ours) & set(ref))
            if shared:
                fold_deltas = []
                for index in shared:
                    reference_rmse = -float(ref[index])
                    if reference_rmse > 0:
                        fold_deltas.append((float(ours[index]) - float(ref[index])) / reference_rmse)
                if fold_deltas:
                    deltas.append(mean(fold_deltas))
        if not deltas:
            continue
        delta = mean(deltas)
        se = stdev(deltas) / (len(deltas) ** 0.5) if len(deltas) > 1 else float("nan")
        out[name] = {
            "delta": delta,
            "ci_low": delta - 1.96 * se if np.isfinite(se) else float("nan"),
            "ci_high": delta + 1.96 * se if np.isfinite(se) else float("nan"),
            "wins": int(sum(value > 0 for value in deltas)),
            "tasks": len(deltas),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["sklearn", "openml", "tabarena"], default="sklearn")
    ap.add_argument("--models", default="rf,linear", help="comma-separated registry names")
    ap.add_argument(
        "--splits",
        type=int,
        default=0,
        help="split cap (0: all official TabArena folds; 3 random splits for non-task suites)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--classification-metric",
        choices=["roc_auc", "average_precision"],
        default="roc_auc",
        help="ranking metric for classification; use average_precision for binary rare-event tasks",
    )
    ap.add_argument(
        "--fold-indices",
        default="",
        help="comma-separated official fold indices; evaluate only these folds",
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse successful fold records from the sibling *.folds.csv checkpoint",
    )
    ap.add_argument(
        "--reference",
        default="tabpfn",
        help="reference model for paired task-level deltas (empty to suppress)",
    )
    # tabarena-only filters
    ap.add_argument("--ta-size", choices=["all", "le10k", "10k-100k", "gt100k"], default="all")
    ap.add_argument("--ta-runnable", choices=["any", "tabpfnv2", "tabicl"], default="any")
    ap.add_argument("--ta-max", type=int, default=0, help="cap #datasets (0 = no cap)")
    ap.add_argument(
        "--ta-datasets",
        default="",
        help="comma-separated ordered TabArena dataset names for a fixed promotion slice",
    )
    args = ap.parse_args()

    try:
        fold_indices = tuple(int(value) for value in args.fold_indices.split(",") if value)
    except ValueError:
        ap.error("--fold-indices must be a comma-separated list of non-negative integers")
    if any(index < 0 for index in fold_indices) or len(set(fold_indices)) != len(fold_indices):
        ap.error("--fold-indices must contain unique non-negative integers")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.suite == "sklearn":
        suite = ds_mod.sklearn_suite()
    elif args.suite == "openml":
        suite = ds_mod.openml_suite()
    else:
        names = [name.strip() for name in args.ta_datasets.split(",") if name.strip()] or None
        suite = ds_mod.tabarena_suite(
            size=args.ta_size,
            runnable=None if args.ta_runnable == "any" else args.ta_runnable,
            max_datasets=args.ta_max or None,
            dataset_names=names,
        )

    split_note = "official task folds" if args.suite == "tabarena" and args.splits <= 0 else args.splits
    print(
        f"suite={args.suite} ({len(suite)} datasets) models={model_names} "
        f"splits={split_note} classification_metric={args.classification_metric}\n"
    )
    fold_records = {}
    fold_path = None
    if args.out:
        fold_path = _fold_record_path(args.out)
        if args.resume:
            fold_records = _read_fold_records(fold_path)
            if fold_records:
                print(f"resuming {len(fold_records)} recorded folds from {fold_path}\n")

    def checkpoint(record):
        fold_records[_fold_record_key(record)] = record
        _write_fold_records(fold_path, fold_records)

    results, timing, observations = evaluate(
        model_names,
        suite,
        args.splits,
        args.seed,
        fold_indices=fold_indices or None,
        classification_metric=args.classification_metric,
        completed_records=fold_records,
        record_callback=(checkpoint if fold_path is not None else None),
    )

    print("\n=== leaderboard: average rank (lower better) + inference cost ===")
    ranks = average_ranks(results, model_names)
    for m, r in sorted(ranks.items(), key=lambda kv: (np.isnan(kv[1]), kv[1])):
        fit_t = sum(timing[m]["fit"])
        infer_t = sum(timing[m]["predict"])
        print(f"  {m:>10} | avg_rank={r:.3f} | infer={infer_t:7.2f}s | fit={fit_t:7.2f}s")

    if args.reference:
        task_by_dataset = {dataset.name: dataset.task for dataset in suite}
        classification = {
            name: scores for name, scores in observations.items() if task_by_dataset[name] == "classification"
        }
        regression = {
            name: scores for name, scores in observations.items() if task_by_dataset[name] == "regression"
        }
        deltas = paired_deltas(classification, model_names, args.reference)
        if deltas:
            print(f"\n=== paired classification delta vs {args.reference} (higher is better) ===")
            for name, s in sorted(deltas.items()):
                ci = f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]" if np.isfinite(s["ci_low"]) else "n/a"
                print(
                    f"  {name:>10} - {args.reference:<10} | delta={s['delta']:+.4f} "
                    f"| 95% CI={ci} | wins={s['wins']}/{s['tasks']} tasks"
                )
        relative = paired_relative_rmse_deltas(regression, model_names, args.reference)
        if relative:
            print(f"\n=== paired relative RMSE reduction vs {args.reference} (higher is better) ===")
            for name, s in sorted(relative.items()):
                ci = (
                    f"[{100 * s['ci_low']:+.2f}%, {100 * s['ci_high']:+.2f}%]"
                    if np.isfinite(s["ci_low"])
                    else "n/a"
                )
                print(
                    f"  {name:>10} - {args.reference:<10} | reduction={100 * s['delta']:+.2f}% "
                    f"| 95% CI={ci} | wins={s['wins']}/{s['tasks']} tasks"
                )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "model", "score"])
            for dsn, scores in results.items():
                for m, s in scores.items():
                    w.writerow([dsn, m, "" if s is None else f"{s:.6f}"])
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
