"""Audit TabPVN on every million-row dataset in the BeyondArena core suite.

BeyondArena's current million-row slice contains eleven datasets.  Each has one
core split, so this script evaluates the packaged zero-knob default exactly once
per dataset.  Datasets are downloaded one at a time from the immutable Data
Foundry versions on Hugging Face and each fit runs in a fresh subprocess.  This
keeps results checkpointed when a later task times out or exceeds local memory.

The preflight is intentionally conservative.  TabPVN currently materializes a
dense float64 matrix, and its leak-safe target encoding can transiently hold two
such matrices.  Tasks that cannot fit the configured memory budget are recorded
as capacity failures instead of risking an operating-system kill.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from tabpvn import TabPVN

REPOSITORY = "TabArena/BeyondArena"
GIB = float(1024**3)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    relative_path: str
    task: str
    problem_type: str
    metric: str
    split_regime: str
    target: str
    train_rows: int
    test_rows: int
    features: int
    official_encoded_features: int
    text_columns: int
    numeric_columns: int
    missing_fraction: float


SPECS = (
    DatasetSpec(
        "mercari_price_suggestion_1m",
        "mercari_price_suggestion/versions/019d736f-26fd-73bd-8853-dc219e5f4ed5",
        "regression",
        "regression",
        "root_mean_squared_error",
        "iid",
        "price",
        1_000_000,
        250_000,
        6,
        130,
        4,
        0,
        0.081118,
    ),
    DatasetSpec(
        "climate_model_weather_forecasting_1m",
        "climate_model_weather_forecasting/versions/019d7379-7906-707e-95e9-f479afb2e2c1",
        "regression",
        "regression",
        "root_mean_squared_error",
        "temporal",
        "fact_temperature",
        1_000_000,
        250_000,
        100,
        109,
        0,
        95,
        0.007938,
    ),
    DatasetSpec(
        "cooking_time_1m",
        "cooking_time/versions/019d737d-b4b1-7c24-8930-70a2c242b3f9",
        "regression",
        "regression",
        "root_mean_squared_error",
        "temporal",
        "cooking_time_minutes",
        1_000_000,
        250_000,
        196,
        205,
        0,
        186,
        0.080519,
    ),
    DatasetSpec(
        "delivery_eta_1m",
        "delivery_eta/versions/019d7382-df54-7dd3-8198-0567d7499858",
        "regression",
        "regression",
        "root_mean_squared_error",
        "temporal",
        "delivery_eta_minutes",
        1_000_000,
        250_000,
        225,
        234,
        0,
        221,
        0.171878,
    ),
    DatasetSpec(
        "home_credit_default_stability_1m",
        "home_credit_default_stability/versions/019d7383-f45a-72f7-ac99-09447cf6d41f",
        "classification",
        "binary",
        "roc_auc",
        "temporal",
        "target",
        1_000_000,
        224_927,
        711,
        720,
        0,
        597,
        0.328045,
    ),
    DatasetSpec(
        "consumer_complaints_1m",
        "consumer_complaints/versions/019d738b-4c6e-751f-972a-5f63b1508f70",
        "classification",
        "multiclass",
        "log_loss",
        "temporal",
        "Company response to consumer",
        1_000_000,
        226_140,
        12,
        114,
        3,
        0,
        0.160960,
    ),
    DatasetSpec(
        "lending_club_1m",
        "lending_club/versions/019d738c-5edd-7437-9116-e3bc87a8b0c5",
        "classification",
        "binary",
        "roc_auc",
        "temporal",
        "Default",
        814_751,
        250_000,
        96,
        278,
        5,
        83,
        0.322620,
    ),
    DatasetSpec(
        "sepsis_prediction_1m",
        "sepsis_prediction/versions/019d7391-f36e-72f8-89d7-f7ca71725034",
        "classification",
        "binary",
        "roc_auc",
        "grouped",
        "SepsisLabel",
        970_175,
        258_511,
        43,
        42,
        0,
        38,
        0.667262,
    ),
    DatasetSpec(
        "maps_router_eta_1m",
        "maps_router_eta/versions/019d7407-606f-7147-b041-c7d0a3847c71",
        "regression",
        "regression",
        "root_mean_squared_error",
        "temporal",
        "target_log_spkm",
        1_000_000,
        250_000,
        988,
        997,
        0,
        985,
        0.046504,
    ),
    DatasetSpec(
        "amex_non_iid_1m",
        "amex_non_iid/versions/019d7455-0e4e-7261-9842-93177684d486",
        "classification",
        "binary",
        "roc_auc",
        "grouped",
        "target",
        1_000_350,
        249_255,
        190,
        198,
        0,
        177,
        0.085851,
    ),
    DatasetSpec(
        "electric_motor_temperature_prediction",
        "electric_motor_temperature_prediction/019d7392-e21f-7e60-9efb-2141c3513fd9",
        "regression",
        "regression",
        "root_mean_squared_error",
        "grouped",
        "permanent_magnet_temperature",
        968_996,
        327_320,
        110,
        109,
        0,
        109,
        0.0,
    ),
)
SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


def _peak_rss_gib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_used = raw if sys.platform == "darwin" else raw * 1024.0
    return bytes_used / GIB


def _estimated_dense_width(spec: DatasetSpec) -> int:
    token_width = 300 if spec.task == "classification" else 1_200
    text_delta = spec.text_columns * (token_width - 32)
    missing_indicators = spec.numeric_columns if spec.missing_fraction > 0 else 0
    return spec.official_encoded_features + text_delta + missing_indicators


def _memory_preflight(spec: DatasetSpec) -> dict[str, float | int]:
    width = _estimated_dense_width(spec)
    matrix_gib = spec.train_rows * width * 8.0 / GIB
    # fit_transform may hold its initial and OOF-target-encoded transforms together.
    estimated_peak_gib = 2.0 * matrix_gib + 2.0
    return {
        "estimated_dense_features": width,
        "estimated_dense_matrix_gib": matrix_gib,
        "estimated_peak_gib": estimated_peak_gib,
    }


def _download(spec: DatasetSpec, cache_dir: Path) -> tuple[Path, Path, float]:
    started = time.perf_counter()
    local_dir = cache_dir / spec.name
    data = hf_hub_download(
        REPOSITORY,
        f"{spec.relative_path}/dataset.parquet",
        repo_type="dataset",
        local_dir=local_dir,
    )
    split = hf_hub_download(
        REPOSITORY,
        f"{spec.relative_path}/experiment_metadata.predictive-ml-splits-mold-v1.json",
        repo_type="dataset",
        local_dir=local_dir,
    )
    return Path(data), Path(split), time.perf_counter() - started


def _row_selection(indices: list[int], expected_rows: int, label: str) -> tuple[int, int] | np.ndarray:
    if len(indices) != expected_rows:
        raise ValueError(f"{label} split has {len(indices):,} rows; expected {expected_rows:,}")
    if not indices:
        raise ValueError(f"{label} split is empty")
    values = np.asarray(indices, dtype=np.int64)
    gaps = np.diff(values)
    if np.any(gaps <= 0):
        raise ValueError(f"{label} split indices must be strictly increasing")
    if np.all(gaps == 1):
        return int(values[0]), int(values[-1] + 1)
    return values


def _selection_values(selection: tuple[int, int] | np.ndarray) -> np.ndarray:
    if isinstance(selection, tuple):
        return np.arange(selection[0], selection[1], dtype=np.int64)
    return selection


def _official_selections(
    spec: DatasetSpec, split_path: Path
) -> tuple[tuple[int, int] | np.ndarray, tuple[int, int] | np.ndarray]:
    with split_path.open() as handle:
        metadata = json.load(handle)
    train_indices, test_indices = metadata["splits"]["0"]["0"]
    train = _row_selection(train_indices, spec.train_rows, "train")
    test = _row_selection(test_indices, spec.test_rows, "test")
    if np.intersect1d(
        _selection_values(train),
        _selection_values(test),
        assume_unique=True,
    ).size:
        raise ValueError("official train and test ranges overlap")
    return train, test


def _read_rows(path: Path, selection: tuple[int, int] | np.ndarray):
    parquet = pq.ParquetFile(path)
    contiguous = isinstance(selection, tuple)
    if contiguous:
        wanted_start, wanted_stop = selection
        requested_rows = wanted_stop - wanted_start
    else:
        indices = selection
        wanted_start, wanted_stop = int(indices[0]), int(indices[-1] + 1)
        requested_rows = len(indices)
    chunks = []
    row_start = 0
    for row_group in range(parquet.num_row_groups):
        rows = parquet.metadata.row_group(row_group).num_rows
        row_stop = row_start + rows
        overlap_start = max(row_start, wanted_start)
        overlap_stop = min(row_stop, wanted_stop)
        if overlap_start < overlap_stop:
            chunk = parquet.read_row_group(row_group)
            if contiguous:
                chunks.append(chunk.slice(overlap_start - row_start, overlap_stop - overlap_start))
            else:
                first = int(np.searchsorted(indices, row_start, side="left"))
                last = int(np.searchsorted(indices, row_stop, side="left"))
                if first < last:
                    local_indices = pa.array(indices[first:last] - row_start)
                    chunks.append(chunk.take(local_indices))
        row_start = row_stop
        if row_start >= wanted_stop:
            break
    if not chunks:
        raise ValueError("row selection is outside parquet file")
    table = pa.concat_tables(chunks)
    if table.num_rows != requested_rows:
        raise ValueError(f"read {table.num_rows:,} rows for {requested_rows:,} requested rows")
    frame = table.to_pandas()
    del table, chunks
    return frame


def _common_record(spec: DatasetSpec) -> dict:
    return {
        "dataset": spec.name,
        "arena": "BeyondArena",
        "arena_subset": "core_1m_plus",
        "official_core_split": "r0f0",
        "split_regime": spec.split_regime,
        "problem_type": spec.problem_type,
        "objective_metric": spec.metric,
        "train_rows": spec.train_rows,
        "test_rows": spec.test_rows,
        "features": spec.features,
        **_memory_preflight(spec),
    }


def _metric_record(spec: DatasetSpec, model: TabPVN, y_true, prediction) -> dict:
    if spec.problem_type == "binary":
        classes = np.asarray(model.classes_)
        positive = y_true == classes[1]
        labels = classes[prediction.argmax(axis=1)]
        return {
            "objective_value": float(roc_auc_score(positive, prediction[:, 1])),
            "objective_higher_is_better": True,
            "roc_auc": float(roc_auc_score(positive, prediction[:, 1])),
            "accuracy": float(accuracy_score(y_true, labels)),
            "log_loss": float(log_loss(y_true, prediction, labels=classes)),
        }
    if spec.problem_type == "multiclass":
        classes = np.asarray(model.classes_)
        loss = float(log_loss(y_true, prediction, labels=classes))
        labels = classes[prediction.argmax(axis=1)]
        return {
            "objective_value": loss,
            "objective_higher_is_better": False,
            "accuracy": float(accuracy_score(y_true, labels)),
            "log_loss": loss,
        }
    rmse = float(mean_squared_error(y_true, prediction) ** 0.5)
    return {
        "objective_value": rmse,
        "objective_higher_is_better": False,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def _audit_one(spec: DatasetSpec, cache_dir: Path) -> dict:
    record = _common_record(spec)
    record["status"] = "running"
    data_path, split_path, download_seconds = _download(spec, cache_dir)
    record["download_seconds"] = download_seconds
    record["parquet_gib"] = data_path.stat().st_size / GIB
    train_selection, test_selection = _official_selections(spec, split_path)

    started = time.perf_counter()
    train = _read_rows(data_path, train_selection)
    if spec.target not in train:
        raise ValueError(f"target column {spec.target!r} is missing")
    y_train = train.pop(spec.target).to_numpy()
    record["train_load_seconds"] = time.perf_counter() - started

    model = TabPVN(seed=0, task=spec.task)
    print(f"[{spec.name}] fitting {len(train):,} x {train.shape[1]:,}", flush=True)
    started = time.perf_counter()
    model.fit(train, y_train)
    record["fit_seconds"] = time.perf_counter() - started
    del train, y_train
    gc.collect()

    started = time.perf_counter()
    test = _read_rows(data_path, test_selection)
    y_test = test.pop(spec.target).to_numpy()
    record["test_load_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    prediction = model.predict_proba(test) if spec.task == "classification" else model.predict(test)
    record["predict_seconds"] = time.perf_counter() - started
    record.update(_metric_record(spec, model, y_test, prediction))
    record.update(
        {
            "status": "ok",
            "peak_rss_gib": _peak_rss_gib(),
            "resolved_boost": json.dumps(model.boost_, sort_keys=True),
            "numeric_interval_selected": model._numeric_interval is not None,
            "categorical_posterior_selected": model._category_posterior is not None,
        }
    )
    return record


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")


def _run_worker(spec: DatasetSpec, cache_dir: Path, result_path: Path) -> int:
    try:
        record = _audit_one(spec, cache_dir)
    except Exception as error:  # keep a failed task from erasing completed checkpoints
        record = _common_record(spec)
        record.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=12),
                "peak_rss_gib": _peak_rss_gib(),
            }
        )
        _write_json(result_path, record)
        traceback.print_exc()
        return 1
    _write_json(result_path, record)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0


def _read_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["dataset"]: dict(row) for row in csv.DictReader(handle)}


def _write_csv(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [records[spec.name] for spec in SPECS if spec.name in records]
    fieldnames = []
    for record in ordered:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def _selected_specs(names: list[str]) -> list[DatasetSpec]:
    if not names or names == ["all"]:
        return list(SPECS)
    unknown = sorted(set(names) - SPEC_BY_NAME.keys())
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")
    return [SPEC_BY_NAME[name] for name in names]


def _orchestrate(args) -> int:
    output = Path(args.out)
    cache_dir = Path(args.cache_dir)
    records = _read_existing(output)
    result_dir = cache_dir / "worker-results"
    selected = _selected_specs(args.dataset)

    for spec in selected:
        if spec.name in records and not args.rerun:
            print(f"[{spec.name}] checkpoint exists; skipping", flush=True)
            continue
        preflight = _memory_preflight(spec)
        if preflight["estimated_peak_gib"] > args.memory_budget_gib and not args.force_memory:
            record = _common_record(spec)
            record.update(
                {
                    "status": "skipped_memory_preflight",
                    "error": (
                        f"estimated {preflight['estimated_peak_gib']:.2f} GiB exceeds "
                        f"{args.memory_budget_gib:.2f} GiB budget"
                    ),
                }
            )
            records[spec.name] = record
            _write_csv(output, records)
            print(f"[{spec.name}] {record['error']}", flush=True)
            continue

        worker_result = result_dir / f"{spec.name}.json"
        worker_result.unlink(missing_ok=True)
        print(f"[{spec.name}] downloading immutable dataset version", flush=True)
        try:
            _, _, download_seconds = _download(spec, cache_dir)
        except Exception as error:
            record = _common_record(spec)
            record.update(
                {
                    "status": "download_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            records[spec.name] = record
            _write_csv(output, records)
            print(f"[{spec.name}] download failed: {error}", flush=True)
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--dataset",
            spec.name,
            "--cache-dir",
            str(cache_dir),
            "--worker-result",
            str(worker_result),
        ]
        print(
            f"[{spec.name}] starting worker; estimated peak {preflight['estimated_peak_gib']:.2f} GiB",
            flush=True,
        )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                check=False,
                timeout=args.timeout_seconds,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = None
        elapsed = time.perf_counter() - started

        if worker_result.exists():
            record = json.loads(worker_result.read_text())
        else:
            record = _common_record(spec)
            status = "timeout" if returncode is None else "worker_killed"
            record.update(
                {
                    "status": status,
                    "error": (
                        f"worker exceeded {args.timeout_seconds}s"
                        if returncode is None
                        else f"worker exited with code {returncode} before writing a result"
                    ),
                }
            )
        # Dataset transfer is benchmark setup, not model runtime. The worker's
        # cache lookup takes milliseconds; retain the real setup duration here.
        record["download_seconds"] = download_seconds
        record["wall_seconds"] = elapsed
        records[spec.name] = record
        _write_csv(output, records)
        print(f"[{spec.name}] status={record['status']}; checkpointed {output}", flush=True)

        if not args.keep_downloads:
            shutil.rmtree(cache_dir / spec.name, ignore_errors=True)

    print(f"wrote {output}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--out", default="results/tabpvn_beyondarena_1m_current.csv")
    parser.add_argument("--cache-dir", default="/tmp/tabpvn-beyondarena-1m")
    parser.add_argument("--memory-budget-gib", type=float, default=12.0)
    parser.add_argument("--timeout-seconds", type=int, default=1_200)
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--force-memory", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    selected = _selected_specs(args.dataset)
    if args.worker:
        if len(selected) != 1 or not args.worker_result:
            raise ValueError("worker mode requires one --dataset and --worker-result")
        return _run_worker(selected[0], Path(args.cache_dir), Path(args.worker_result))
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
