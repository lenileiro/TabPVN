"""Run Google Research TabFM on the African Credit Scoring challenge.

The challenge has repeated lender rows for one loan decision. This runner
reuses the loan-level feature table and deterministic competition invariants
from ``african_credit_challenge.py`` while replacing TabPVN with TabFM.

TabFM is a large in-context model. The default here is intentionally bounded
for a 16 GB Apple Silicon machine: one ensemble member, a rare-event-aware
1,024-loan context, cached context states, and chunked MPS inference.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from benchmark import african_credit_challenge as challenge

DEFAULT_TABFM_REPO = Path(os.environ.get("TABFM_REPO", "/tmp/google-tabfm"))
DEFAULT_SUBMISSION = Path("results/african_credit_tabfm_submission.csv")
DEFAULT_REPORT = Path("results/african_credit_tabfm_report.json")
DEFAULT_CONTEXT_ROWS = 1_024
DEFAULT_VALIDATION_ROWS = 4_096
DEFAULT_QUERY_ROWS = 256


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _cached_checkpoint() -> Path | None:
    snapshots = Path.home() / ".cache/huggingface/hub/models--google--tabfm-1.0.0-pytorch/snapshots"
    if not snapshots.exists():
        return None
    candidates = [
        path
        for path in snapshots.iterdir()
        if (path / "classification/model.safetensors").exists()
        or (path / "classification/pytorch_model.bin").exists()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _load_tabfm(
    tabfm_repo: Path,
    checkpoint: Path | None,
    requested_device: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    if not (tabfm_repo / "tabfm/__init__.py").exists():
        raise FileNotFoundError(f"TabFM source not found at {tabfm_repo}; clone google-research/tabfm there")
    sys.path.insert(0, str(tabfm_repo))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import tabfm  # type: ignore[import-not-found]
    import torch
    from tabfm.src.pytorch.model import TabFM  # type: ignore[import-not-found]

    if requested_device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = requested_device

    checkpoint = checkpoint or _cached_checkpoint()
    started = time.perf_counter()
    legacy_weights = None if checkpoint is None else checkpoint / "classification/pytorch_model.bin"
    if legacy_weights is not None and legacy_weights.exists():
        config_path = checkpoint / "classification/config.json"
        config = json.loads(config_path.read_text())
        for key in ("model_type", "version", "task", "framework"):
            config.pop(key, None)
        config["is_classifier"] = True

        # The July 2 checkpoint predates TabFM's safetensors loader. Meta-device
        # construction plus mmap avoids holding two 6.1 GB float32 copies.
        state = torch.load(legacy_weights, map_location="cpu", weights_only=True, mmap=True)
        with torch.device("meta"):
            model = TabFM(**config)
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to(device=device, dtype=torch.bfloat16).eval()
        del state
        gc.collect()
    else:
        model = tabfm.tabfm_v1_0_0_pytorch.load(
            model_type="classification",
            checkpoint_path=None if checkpoint is None else str(checkpoint),
            device=device,
            dtype=torch.bfloat16,
            use_cache=False,
        )
    if device == "mps":
        torch.mps.synchronize()

    return (
        tabfm,
        torch,
        model,
        {
            "repository": str(tabfm_repo),
            "version": tabfm.__version__,
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "device": device,
            "dtype": "bfloat16",
            "load_seconds": time.perf_counter() - started,
        },
    )


def _context_and_validation_indices(
    target: np.ndarray,
    context_rows: int,
    validation_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if context_rows < 2 or validation_rows < 2:
        raise ValueError("context and validation must each contain at least two rows")
    if context_rows + validation_rows > len(target):
        raise ValueError("context_rows + validation_rows exceeds the training rows")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_rows, random_state=seed)
    context_pool, validation = next(splitter.split(np.zeros(len(target)), target))
    positive_pool = context_pool[target[context_pool] == 1]
    negative_pool = context_pool[target[context_pool] == 0]

    positive_rows = min(len(positive_pool), max(64, context_rows // 8))
    negative_rows = context_rows - positive_rows
    if negative_rows > len(negative_pool):
        raise ValueError("not enough negative rows for the requested context")

    rng = np.random.default_rng(seed)
    context = np.concatenate(
        [
            rng.choice(positive_pool, positive_rows, replace=False),
            rng.choice(negative_pool, negative_rows, replace=False),
        ]
    )
    rng.shuffle(context)
    return (
        context,
        validation,
        {
            "context_rows": int(len(context)),
            "context_events": int(target[context].sum()),
            "context_event_rate": float(target[context].mean()),
            "validation_rows": int(len(validation)),
            "validation_events": int(target[validation].sum()),
            "validation_event_rate": float(target[validation].mean()),
            "context_selection": "stratified rare-event context from the non-validation pool",
        },
    )


def _classifier(tabfm: Any, model: Any, seed: int) -> Any:
    return tabfm.TabFMClassifier(
        model=model,
        n_estimators=1,
        batch_size=1,
        random_state=seed,
        cache_context=True,
        maybe_quantize_kv_cache=True,
        keep_cache_on_device=True,
    )


def _positive_probability(classifier: Any, features: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(classifier.predict_proba(features), dtype=float)
    classes = np.asarray(classifier.classes_)
    position = np.flatnonzero(classes == 1)
    if len(position) != 1:
        raise ValueError(f"expected binary classes containing 1, got {classes.tolist()}")
    return probability[:, int(position[0])]


def _predict_in_chunks(
    classifier: Any,
    features: pd.DataFrame,
    chunk_rows: int,
    label: str,
) -> tuple[np.ndarray, float]:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    output = np.empty(len(features), dtype=float)
    started = time.perf_counter()
    chunks = max(1, (len(features) + chunk_rows - 1) // chunk_rows)
    for chunk_index, start in enumerate(range(0, len(features), chunk_rows), start=1):
        stop = min(start + chunk_rows, len(features))
        output[start:stop] = _positive_probability(classifier, features.iloc[start:stop])
        if chunk_index == chunks or chunk_index % 10 == 0:
            print(f"{label}: {stop:,}/{len(features):,} rows", flush=True)
    return output, time.perf_counter() - started


def _select_f1_threshold(
    target: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        target,
        probability,
        sample_weight=sample_weight,
    )
    score = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_score = float(score.max())
    # Prefer the highest equally scoring threshold so a tie does not inflate
    # false positives on the public leaderboard.
    best = int(np.flatnonzero(np.isclose(score, best_score, rtol=0.0, atol=1e-12))[-1])
    return float(thresholds[best]), best_score


def _validation_report(
    target: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
    predict_seconds: float,
) -> tuple[float, dict[str, Any]]:
    threshold, _ = _select_f1_threshold(target, probability, sample_weight)
    prediction = probability >= threshold
    argmax = probability >= 0.5
    return threshold, {
        "threshold": threshold,
        "f1": float(f1_score(target, prediction, sample_weight=sample_weight)),
        "precision": float(precision_score(target, prediction, sample_weight=sample_weight, zero_division=0)),
        "recall": float(recall_score(target, prediction, sample_weight=sample_weight, zero_division=0)),
        "argmax_f1": float(f1_score(target, argmax, sample_weight=sample_weight)),
        "average_precision": float(average_precision_score(target, probability, sample_weight=sample_weight)),
        "roc_auc": float(roc_auc_score(target, probability, sample_weight=sample_weight)),
        "predicted_default_loans": int(prediction.sum()),
        "actual_default_loans": int(target.sum()),
        "predict_seconds": predict_seconds,
    }


def _test_loans(loan: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    ids = pd.DataFrame({challenge.LOAN_ID: test[challenge.LOAN_ID].unique()})
    result = ids.merge(loan, on=challenge.LOAN_ID, how="left", validate="one_to_one")
    if result["country_id"].isna().any():
        raise AssertionError("not every test loan is present in the feature table")
    return result


def _release_classifier(classifier: Any, torch: Any, device: str) -> None:
    del classifier
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=challenge.DEFAULT_DATA_DIR)
    parser.add_argument("--tabfm-repo", type=Path, default=DEFAULT_TABFM_REPO)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--context-rows", type=int, default=DEFAULT_CONTEXT_ROWS)
    parser.add_argument("--validation-rows", type=int, default=DEFAULT_VALIDATION_ROWS)
    parser.add_argument("--query-rows", type=int, default=DEFAULT_QUERY_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    run_started = time.perf_counter()
    train, test, sample = challenge._load_data(args.data_dir)
    loan, data_report = challenge._build_loan_table(train, test)
    train_loans = loan[loan[challenge.TARGET].notna()].reset_index(drop=True)
    test_loans = _test_loans(loan, test)
    target = train_loans[challenge.TARGET].to_numpy(dtype=int)
    context_rows, validation_rows, sampling_report = _context_and_validation_indices(
        target,
        context_rows=args.context_rows,
        validation_rows=args.validation_rows,
        seed=args.seed,
    )

    tabfm, torch, model, runtime_report = _load_tabfm(
        args.tabfm_repo,
        args.checkpoint,
        args.device,
    )
    print(
        f"loaded TabFM {runtime_report['version']} on {runtime_report['device']} in "
        f"{runtime_report['load_seconds']:.1f}s",
        flush=True,
    )

    known_target = train_loans.set_index(challenge.LOAN_ID)[challenge.TARGET].astype(int)
    known = test_loans[challenge.LOAN_ID].isin(known_target.index)
    kenya_unknown = test_loans["country_id"].eq("Kenya") & ~known
    ghana_unknown = test_loans["country_id"].eq("Ghana") & ~known
    other_unknown = ~known & ~kenya_unknown & ~ghana_unknown
    ghana_type3 = ghana_unknown & test_loans["loan_type"].eq("Type_3")
    portable_unknown = (ghana_unknown & ~ghana_type3) | other_unknown

    prediction = np.zeros(len(test_loans), dtype=int)
    probability = np.zeros(len(test_loans), dtype=float)
    source = np.full(len(test_loans), "", dtype=object)
    prediction[known] = test_loans.loc[known, challenge.LOAN_ID].map(known_target).to_numpy(int)
    probability[known] = prediction[known]
    source[known] = "known_train_loan"

    validation_target = target[validation_rows]
    validation_weight = train_loans.iloc[validation_rows]["loan_lender_rows"].to_numpy(float)

    print("fitting cached rich TabFM context", flush=True)
    rich = _classifier(tabfm, model, args.seed)
    fit_started = time.perf_counter()
    rich.fit(challenge._features(train_loans.iloc[context_rows], portable=False), target[context_rows])
    rich_fit_seconds = time.perf_counter() - fit_started
    rich_validation_probability, rich_validation_seconds = _predict_in_chunks(
        rich,
        challenge._features(train_loans.iloc[validation_rows], portable=False),
        args.query_rows,
        "rich validation",
    )
    rich_threshold, rich_report = _validation_report(
        validation_target,
        rich_validation_probability,
        validation_weight,
        rich_validation_seconds,
    )
    print(f"rich validation f1={rich_report['f1']:.6f} threshold={rich_threshold:.6g}", flush=True)
    if kenya_unknown.any():
        kenya_probability, kenya_seconds = _predict_in_chunks(
            rich,
            challenge._features(test_loans.loc[kenya_unknown], portable=False),
            args.query_rows,
            "Kenya test",
        )
        probability[kenya_unknown] = kenya_probability
        prediction[kenya_unknown] = (kenya_probability >= rich_threshold).astype(int)
        source[kenya_unknown] = "tabfm_rich"
    else:
        kenya_seconds = 0.0
    _release_classifier(rich, torch, runtime_report["device"])

    print("fitting cached portable TabFM context", flush=True)
    portable = _classifier(tabfm, model, args.seed + 1)
    fit_started = time.perf_counter()
    portable.fit(
        challenge._features(train_loans.iloc[context_rows], portable=True),
        target[context_rows],
    )
    portable_fit_seconds = time.perf_counter() - fit_started
    portable_validation_probability, portable_validation_seconds = _predict_in_chunks(
        portable,
        challenge._features(train_loans.iloc[validation_rows], portable=True),
        args.query_rows,
        "portable validation",
    )
    portable_threshold, portable_report = _validation_report(
        validation_target,
        portable_validation_probability,
        validation_weight,
        portable_validation_seconds,
    )
    print(
        f"portable validation f1={portable_report['f1']:.6f} threshold={portable_threshold:.6g}",
        flush=True,
    )
    if portable_unknown.any():
        portable_probability, portable_seconds = _predict_in_chunks(
            portable,
            challenge._features(test_loans.loc[portable_unknown], portable=True),
            args.query_rows,
            "portable test",
        )
        probability[portable_unknown] = portable_probability
        prediction[portable_unknown] = (portable_probability >= portable_threshold).astype(int)
        source[portable_unknown] = "tabfm_portable"
    else:
        portable_seconds = 0.0
    _release_classifier(portable, torch, runtime_report["device"])

    ghana_type3_last = ghana_type3 & test_loans["customer_is_last_visible_loan"].eq(1.0)
    prediction[ghana_type3] = 0
    prediction[ghana_type3_last] = 1
    source[ghana_type3] = "ghana_type3_nonfinal_rule"
    source[ghana_type3_last] = "ghana_type3_final_rule"
    if np.any(source == ""):
        raise AssertionError("not every test loan received a prediction source")

    loan_prediction = pd.DataFrame(
        {
            challenge.LOAN_ID: test_loans[challenge.LOAN_ID],
            challenge.TARGET: prediction,
            "probability": probability,
            "source": source,
            "country_id": test_loans["country_id"],
            "loan_type": test_loans["loan_type"],
        }
    )
    submission_report = challenge._write_submission(
        test,
        sample,
        loan_prediction,
        args.submission,
    )
    report = {
        "competition": "Zindi African Credit Scoring Challenge",
        "metric": "F1",
        "model": "Google Research TabFM v1.0.0 weights",
        "seed": args.seed,
        "data": data_report,
        "sampling": sampling_report,
        "runtime": {
            **runtime_report,
            "n_estimators": 1,
            "query_rows": args.query_rows,
            "rich_context_fit_seconds": rich_fit_seconds,
            "portable_context_fit_seconds": portable_fit_seconds,
            "kenya_predict_seconds": kenya_seconds,
            "portable_test_predict_seconds": portable_seconds,
            "total_seconds": time.perf_counter() - run_started,
        },
        "validation": {
            "geometry": "fixed stratified loan holdout; metrics weighted by submission rows",
            "rich": rich_report,
            "portable": portable_report,
        },
        "submission": {
            **submission_report,
            "known_train_test_loans": int(known.sum()),
            "ghana_type3_loans": int(ghana_type3.sum()),
            "ghana_type3_final_loans": int(ghana_type3_last.sum()),
            "country_summary": (
                loan_prediction.groupby("country_id")[challenge.TARGET]
                .agg(["count", "sum", "mean"])
                .reset_index()
                .to_dict(orient="records")
            ),
            "source_summary": (
                loan_prediction.groupby("source")[challenge.TARGET]
                .agg(["count", "sum", "mean"])
                .reset_index()
                .to_dict(orient="records")
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(
        f"wrote {args.submission} with {submission_report['predicted_defaults']:,} defaults",
        flush=True,
    )
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
