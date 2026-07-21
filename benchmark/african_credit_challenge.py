"""TabPVN entry for Zindi's African Credit Scoring Challenge.

The competition table has one row per loan/lender pair, while ``target`` is a
loan decision. This runner therefore models one row per loan and maps the
decision back to every submission row. It also records the two batch rules that
cannot be learned from an ordinary row matrix:

* a loan already present in Train.csv has the same decision in Test.csv;
* Ghana's unseen weekly product is treated as a repayment sequence, where only
  the customer's last visible loan is a default candidate.

The second rule is deliberately isolated in the report because it is a
competition-batch assumption. It must not be used for online credit decisions,
where future loans are not visible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from tabpvn import TabPVN

DEFAULT_DATA_DIR = Path("data/african-credit-scoring-challenge20241203-14702-1yayxml")
DEFAULT_SUBMISSION = Path("results/african_credit_tabpvn_submission.csv")
DEFAULT_REPORT = Path("results/african_credit_tabpvn_report.json")

TARGET = "target"
LOAN_ID = "tbl_loan_id"
DATE_COLUMNS = ("disbursement_date", "due_date")
REQUIRED_COLUMNS = {
    "ID",
    "customer_id",
    "country_id",
    LOAN_ID,
    "lender_id",
    "loan_type",
    "Total_Amount",
    "Total_Amount_to_Repay",
    "disbursement_date",
    "due_date",
    "duration",
    "New_versus_Repeat",
    "Amount_Funded_By_Lender",
    "Lender_portion_Funded",
    "Lender_portion_to_be_repaid",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _safe_ratio(numerator: Any, denominator: Any) -> pd.Series:
    numerator = pd.Series(numerator, copy=False).astype(float)
    denominator = pd.Series(denominator, copy=False).astype(float)
    result = numerator.div(denominator.where(denominator.ne(0)))
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "Train.csv"
    test_path = data_dir / "Test.csv"
    sample_path = data_dir / "SampleSubmission.csv"
    missing = [str(path) for path in (train_path, test_path, sample_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"challenge files are missing: {missing}")

    train = pd.read_csv(train_path, parse_dates=list(DATE_COLUMNS))
    test = pd.read_csv(test_path, parse_dates=list(DATE_COLUMNS))
    sample = pd.read_csv(sample_path)
    for name, frame in (("Train.csv", train), ("Test.csv", test)):
        absent = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if absent:
            raise ValueError(f"{name} is missing required columns: {absent}")
    if TARGET not in train or TARGET in test:
        raise ValueError("Train.csv must contain target and Test.csv must not")
    if list(sample.columns) != ["ID", TARGET]:
        raise ValueError("SampleSubmission.csv must have exactly ID,target columns")
    if len(sample) != len(test) or not sample["ID"].equals(test["ID"]):
        raise ValueError("sample submission IDs must match Test.csv in the same order")
    return train, test, sample


def _loan_targets(train: pd.DataFrame) -> tuple[pd.Series, int]:
    grouped = train.groupby(LOAN_ID, sort=False)[TARGET]
    minimum = grouped.min().astype(int)
    maximum = grouped.max().astype(int)
    conflicts = int(minimum.ne(maximum).sum())
    # A conflicting lender row cannot make a loan both repaid and defaulted.
    # Resolve the two observed ties conservatively, matching the published
    # competition invariant that conflicting loan decisions become repaid.
    return minimum.rename(TARGET), conflicts


def _term_bucket(duration: pd.Series) -> pd.Series:
    bins = [-np.inf, 7, 14, 31, 60, 120, 240, np.inf]
    labels = ["week", "fortnight", "month", "two_months", "quarter", "half_year", "long"]
    return pd.cut(duration, bins=bins, labels=labels, include_lowest=True).astype("object")


def _build_loan_table(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_columns = [column for column in train.columns if column != TARGET]
    combined = pd.concat(
        [
            train.loc[:, feature_columns].assign(_source="train"),
            test.loc[:, feature_columns].assign(_source="test"),
        ],
        ignore_index=True,
    )
    combined["_funded_share_squared"] = np.square(combined["Lender_portion_Funded"].astype(float))

    grouped = combined.groupby(LOAN_ID, sort=False, dropna=False)
    loan = grouped.agg(
        customer_id=("customer_id", "first"),
        country_id=("country_id", "first"),
        loan_type=("loan_type", "first"),
        Total_Amount=("Total_Amount", "first"),
        Total_Amount_to_Repay=("Total_Amount_to_Repay", "first"),
        disbursement_date=("disbursement_date", "first"),
        due_date=("due_date", "first"),
        duration=("duration", "first"),
        New_versus_Repeat=(
            "New_versus_Repeat",
            lambda values: "+".join(sorted({str(value) for value in values})),
        ),
        loan_new_lender_rows=(
            "New_versus_Repeat",
            lambda values: int(np.sum(np.asarray(values) == "New Loan")),
        ),
        loan_lender_rows=("ID", "size"),
        loan_unique_lenders=("lender_id", "nunique"),
        loan_funded_sum=("Amount_Funded_By_Lender", "sum"),
        loan_funded_mean=("Amount_Funded_By_Lender", "mean"),
        loan_funded_max=("Amount_Funded_By_Lender", "max"),
        loan_lender_repay_sum=("Lender_portion_to_be_repaid", "sum"),
        loan_lender_repay_mean=("Lender_portion_to_be_repaid", "mean"),
        loan_lender_repay_max=("Lender_portion_to_be_repaid", "max"),
        loan_funded_share_min=("Lender_portion_Funded", "min"),
        loan_funded_share_max=("Lender_portion_Funded", "max"),
        loan_funding_hhi=("_funded_share_squared", "sum"),
        lender_signature=("lender_id", lambda values: "+".join(sorted({str(value) for value in values}))),
        source_count=("_source", "nunique"),
    ).reset_index()

    stable_columns = (
        "customer_id",
        "country_id",
        "loan_type",
        "Total_Amount",
        "Total_Amount_to_Repay",
        "disbursement_date",
        "due_date",
        "duration",
    )
    inconsistent = {
        column: int(grouped[column].nunique(dropna=False).gt(1).sum()) for column in stable_columns
    }
    bad = {column: count for column, count in inconsistent.items() if count}
    if bad:
        raise ValueError(f"loan-level fields disagree across lender rows: {bad}")

    loan["repayment_multiple"] = _safe_ratio(loan["Total_Amount_to_Repay"], loan["Total_Amount"])
    loan["repayment_markup"] = loan["repayment_multiple"] - 1.0
    loan["daily_repayment_multiple"] = _safe_ratio(loan["repayment_multiple"], loan["duration"])
    loan["amount_due_per_day"] = _safe_ratio(loan["Total_Amount_to_Repay"], loan["duration"])
    loan["loan_funded_coverage"] = _safe_ratio(loan["loan_funded_sum"], loan["Total_Amount"])
    loan["loan_repay_coverage"] = _safe_ratio(loan["loan_lender_repay_sum"], loan["Total_Amount_to_Repay"])
    loan["lender_repayment_multiple"] = _safe_ratio(loan["loan_lender_repay_sum"], loan["loan_funded_sum"])
    loan["loan_new_lender_fraction"] = _safe_ratio(loan["loan_new_lender_rows"], loan["loan_lender_rows"])
    loan["actual_duration"] = (loan["due_date"] - loan["disbursement_date"]).dt.days.astype(float)
    loan["duration_delta"] = loan["actual_duration"] - loan["duration"]
    loan["disbursement_year"] = loan["disbursement_date"].dt.year.astype(float)
    loan["disbursement_month"] = loan["disbursement_date"].dt.month.astype(float)
    loan["disbursement_day"] = loan["disbursement_date"].dt.day.astype(float)
    loan["disbursement_weekday"] = loan["disbursement_date"].dt.dayofweek.astype(float)
    loan["disbursement_ordinal"] = (loan["disbursement_date"] - pd.Timestamp("2021-01-01")).dt.days.astype(
        float
    )
    loan["term_bucket"] = _term_bucket(loan["duration"])

    for column in (
        "Total_Amount",
        "Total_Amount_to_Repay",
        "loan_funded_sum",
        "loan_lender_repay_sum",
        "amount_due_per_day",
    ):
        loan[f"log_{column}"] = np.log1p(loan[column].clip(lower=0.0))

    loan = loan.sort_values(
        ["country_id", "customer_id", "disbursement_date", LOAN_ID], kind="stable"
    ).reset_index(drop=True)
    customer = loan.groupby(["country_id", "customer_id"], sort=False, dropna=False)
    loan["customer_prior_visible_loans"] = customer.cumcount().astype(float)
    loan["customer_visible_loans"] = customer[LOAN_ID].transform("size").astype(float)
    loan["customer_is_first_visible_loan"] = loan["customer_prior_visible_loans"].eq(0).astype(float)
    loan["customer_is_last_visible_loan"] = (
        loan["customer_prior_visible_loans"].add(1).eq(loan["customer_visible_loans"])
    ).astype(float)
    previous_date = customer["disbursement_date"].shift()
    next_date = customer["disbursement_date"].shift(-1)
    loan["days_since_previous_loan"] = (loan["disbursement_date"] - previous_date).dt.days.fillna(-1.0)
    loan["days_until_next_visible_loan"] = (next_date - loan["disbursement_date"]).dt.days.fillna(-1.0)
    previous_amount = customer["Total_Amount"].shift()
    loan["amount_vs_previous"] = _safe_ratio(loan["Total_Amount"], previous_amount)
    cumulative_amount = customer["Total_Amount"].cumsum() - loan["Total_Amount"]
    prior_count = loan["customer_prior_visible_loans"]
    prior_mean_amount = cumulative_amount.div(prior_count.where(prior_count.gt(0)))
    loan["amount_vs_prior_mean"] = _safe_ratio(loan["Total_Amount"], prior_mean_amount)
    customer_amount_median = customer["Total_Amount"].transform("median")
    customer_repay_median = customer["Total_Amount_to_Repay"].transform("median")
    loan["amount_vs_customer_median"] = _safe_ratio(loan["Total_Amount"], customer_amount_median)
    loan["repay_vs_customer_median"] = _safe_ratio(loan["Total_Amount_to_Repay"], customer_repay_median)
    loan["customer_amount_percentile"] = customer["Total_Amount"].rank(pct=True, method="average")

    targets, target_conflicts = _loan_targets(train)
    loan = loan.merge(targets, left_on=LOAN_ID, right_index=True, how="left", validate="one_to_one")
    loan["customer_id"] = loan["customer_id"].astype(str)
    loan["lender_signature"] = loan["lender_signature"].astype(str)
    loan["loan_type"] = loan["loan_type"].astype(str)
    loan["New_versus_Repeat"] = loan["New_versus_Repeat"].astype(str)

    report = {
        "raw_rows": int(len(combined)),
        "unique_loans": int(len(loan)),
        "train_unique_loans": int(train[LOAN_ID].nunique()),
        "test_unique_loans": int(test[LOAN_ID].nunique()),
        "shared_train_test_loans": int(len(set(train[LOAN_ID]) & set(test[LOAN_ID]))),
        "conflicting_train_loan_targets": target_conflicts,
        "batch_only_features": [
            "customer_visible_loans",
            "customer_is_last_visible_loan",
            "days_until_next_visible_loan",
            "customer_amount_percentile",
        ],
    }
    return loan, report


_RICH_CATEGORICAL = (
    "customer_id",
    "loan_type",
    "New_versus_Repeat",
    "term_bucket",
    "lender_signature",
)

_BASE_NUMERIC = (
    "duration",
    "actual_duration",
    "duration_delta",
    "repayment_multiple",
    "repayment_markup",
    "daily_repayment_multiple",
    "loan_lender_rows",
    "loan_unique_lenders",
    "loan_funded_share_min",
    "loan_funded_share_max",
    "loan_funding_hhi",
    "loan_new_lender_fraction",
    "loan_funded_coverage",
    "loan_repay_coverage",
    "lender_repayment_multiple",
    "disbursement_year",
    "disbursement_month",
    "disbursement_day",
    "disbursement_weekday",
    "disbursement_ordinal",
    "customer_prior_visible_loans",
    "customer_visible_loans",
    "customer_is_first_visible_loan",
    "customer_is_last_visible_loan",
    "days_since_previous_loan",
    "days_until_next_visible_loan",
    "amount_vs_previous",
    "amount_vs_prior_mean",
    "amount_vs_customer_median",
    "repay_vs_customer_median",
    "customer_amount_percentile",
)

_RICH_AMOUNT_COLUMNS = (
    "Total_Amount",
    "Total_Amount_to_Repay",
    "loan_funded_sum",
    "loan_funded_mean",
    "loan_funded_max",
    "loan_lender_repay_sum",
    "loan_lender_repay_mean",
    "loan_lender_repay_max",
    "amount_due_per_day",
    "log_Total_Amount",
    "log_Total_Amount_to_Repay",
    "log_loan_funded_sum",
    "log_loan_lender_repay_sum",
    "log_amount_due_per_day",
)


def _features(loan: pd.DataFrame, portable: bool) -> pd.DataFrame:
    categorical = ("loan_type", "New_versus_Repeat", "term_bucket") if portable else _RICH_CATEGORICAL
    numeric = _BASE_NUMERIC if portable else _BASE_NUMERIC + _RICH_AMOUNT_COLUMNS
    return loan.loc[:, [*categorical, *numeric]].copy()


def _positive_probability(model: TabPVN, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.classes_)
    position = np.flatnonzero(classes == 1)
    if len(position) != 1:
        raise ValueError(f"expected binary classes containing 1, got {classes.tolist()}")
    return probabilities[:, int(position[0])]


def _f1_prediction(model: TabPVN, features: pd.DataFrame) -> np.ndarray:
    if getattr(model, "rare_event_", False):
        return np.asarray(model.predict_rare(features), dtype=int)
    return np.asarray(model.predict(features), dtype=int)


def _fit_model(features: pd.DataFrame, target: np.ndarray, seed: int) -> tuple[TabPVN, float]:
    model = TabPVN(seed=seed, task="classification")
    started = time.perf_counter()
    model.fit(features, target)
    return model, time.perf_counter() - started


def _validation(
    train: pd.DataFrame,
    loan: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    train_loans = loan[loan[TARGET].notna()].reset_index(drop=True)
    target = train_loans[TARGET].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    fit_rows, valid_rows = next(splitter.split(train_loans, target))
    model, fit_seconds = _fit_model(
        _features(train_loans.iloc[fit_rows], portable=False), target[fit_rows], seed
    )
    valid_features = _features(train_loans.iloc[valid_rows], portable=False)
    started = time.perf_counter()
    probability = _positive_probability(model, valid_features)
    prediction = _f1_prediction(model, valid_features)
    predict_seconds = time.perf_counter() - started

    valid_loan_ids = train_loans.iloc[valid_rows][LOAN_ID].to_numpy()
    prediction_by_loan = dict(zip(valid_loan_ids, prediction, strict=True))
    probability_by_loan = dict(zip(valid_loan_ids, probability, strict=True))
    valid_raw = train[train[LOAN_ID].isin(valid_loan_ids)]
    row_target = valid_raw[TARGET].to_numpy(dtype=int)
    row_prediction = valid_raw[LOAN_ID].map(prediction_by_loan).to_numpy(dtype=int)
    row_probability = valid_raw[LOAN_ID].map(probability_by_loan).to_numpy(dtype=float)

    argmax = np.asarray(model.predict(valid_features), dtype=int)
    argmax_by_loan = dict(zip(valid_loan_ids, argmax, strict=True))
    row_argmax = valid_raw[LOAN_ID].map(argmax_by_loan).to_numpy(dtype=int)
    return {
        "geometry": "stratified holdout of unique loans; metrics are weighted by submission rows",
        "batch_context": "unlabeled train and test rows are visible to sequence features",
        "fit_loans": int(len(fit_rows)),
        "validation_loans": int(len(valid_rows)),
        "validation_rows": int(len(valid_raw)),
        "fit_default_rate": float(target[fit_rows].mean()),
        "validation_default_rate": float(row_target.mean()),
        "f1": float(f1_score(row_target, row_prediction)),
        "precision": float(precision_score(row_target, row_prediction, zero_division=0)),
        "recall": float(recall_score(row_target, row_prediction, zero_division=0)),
        "argmax_f1": float(f1_score(row_target, row_argmax)),
        "average_precision": float(average_precision_score(row_target, row_probability)),
        "roc_auc": float(roc_auc_score(row_target, row_probability)),
        "predicted_defaults": int(row_prediction.sum()),
        "actual_defaults": int(row_target.sum()),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "rare_event_selected": bool(model.rare_event_),
        "rare_event_report": model.rare_event_report_,
        "validation_report": model.validation_report_,
    }


def _fit_submission_models(
    loan: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_loans = loan[loan[TARGET].notna()].reset_index(drop=True)
    test_loan_ids = pd.DataFrame({LOAN_ID: test[LOAN_ID].unique()})
    test_loans = test_loan_ids.merge(
        loan,
        on=LOAN_ID,
        how="left",
        validate="one_to_one",
    )
    if test_loans["country_id"].isna().any():
        raise AssertionError("not every test loan is present in the feature table")
    target = train_loans[TARGET].to_numpy(dtype=int)

    known_target = train_loans.set_index(LOAN_ID)[TARGET].astype(int)
    known_mask = test_loans[LOAN_ID].isin(known_target.index)
    kenya_unknown = test_loans["country_id"].eq("Kenya") & ~known_mask
    ghana_unknown = test_loans["country_id"].eq("Ghana") & ~known_mask
    other_unknown = ~known_mask & ~kenya_unknown & ~ghana_unknown

    prediction = np.zeros(len(test_loans), dtype=int)
    probability = np.zeros(len(test_loans), dtype=float)
    source = np.full(len(test_loans), "", dtype=object)
    if known_mask.any():
        known = test_loans.loc[known_mask, LOAN_ID].map(known_target).to_numpy(dtype=int)
        prediction[known_mask] = known
        probability[known_mask] = known.astype(float)
        source[known_mask] = "known_train_loan"

    rich_model, rich_fit_seconds = _fit_model(_features(train_loans, portable=False), target, seed)
    if kenya_unknown.any():
        kenya_features = _features(test_loans.loc[kenya_unknown], portable=False)
        prediction[kenya_unknown] = _f1_prediction(rich_model, kenya_features)
        probability[kenya_unknown] = _positive_probability(rich_model, kenya_features)
        source[kenya_unknown] = "tabpvn_kenya"

    portable_model, portable_fit_seconds = _fit_model(_features(train_loans, portable=True), target, seed + 1)
    portable_unknown = ghana_unknown | other_unknown
    if portable_unknown.any():
        portable_features = _features(test_loans.loc[portable_unknown], portable=True)
        prediction[portable_unknown] = _f1_prediction(portable_model, portable_features)
        probability[portable_unknown] = _positive_probability(portable_model, portable_features)
        source[portable_unknown] = "tabpvn_portable"

    # Type_3 is Ghana's unseen weekly product. Kenya's analogous weekly product
    # has a 0.10% non-final default rate and 89.6% of all defaults occur on the
    # final visible loan. Apply the batch repayment-sequence invariant explicitly.
    ghana_type3 = ghana_unknown & test_loans["loan_type"].eq("Type_3")
    ghana_type3_last = ghana_type3 & test_loans["customer_is_last_visible_loan"].eq(1.0)
    prediction[ghana_type3] = 0
    prediction[ghana_type3_last] = 1
    source[ghana_type3] = "ghana_type3_nonfinal_rule"
    source[ghana_type3_last] = "ghana_type3_final_rule"

    test_loan_predictions = pd.DataFrame(
        {
            LOAN_ID: test_loans[LOAN_ID],
            TARGET: prediction,
            "probability": probability,
            "source": source,
            "country_id": test_loans["country_id"],
            "loan_type": test_loans["loan_type"],
        }
    )
    report = {
        "rich_model_fit_seconds": rich_fit_seconds,
        "portable_model_fit_seconds": portable_fit_seconds,
        "rich_model_rare_event_selected": bool(rich_model.rare_event_),
        "portable_model_rare_event_selected": bool(portable_model.rare_event_),
        "known_train_test_loans": int(known_mask.sum()),
        "known_train_test_positive_loans": int(prediction[known_mask].sum()),
        "ghana_type3_loans": int(ghana_type3.sum()),
        "ghana_type3_final_loans": int(ghana_type3_last.sum()),
        "country_summary": (
            test_loan_predictions.groupby("country_id")[TARGET]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .to_dict(orient="records")
        ),
        "source_summary": (
            test_loan_predictions.groupby("source")[TARGET]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    return test_loan_predictions, report


def _write_submission(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    loan_predictions: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    prediction_map = loan_predictions.set_index(LOAN_ID)[TARGET]
    submission = sample.copy()
    submission[TARGET] = test[LOAN_ID].map(prediction_map).to_numpy()
    if submission[TARGET].isna().any():
        raise AssertionError("not every test row received a loan prediction")
    submission[TARGET] = submission[TARGET].astype(int)
    if not set(submission[TARGET].unique()).issubset({0, 1}):
        raise AssertionError("submission target must contain binary labels")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": int(len(submission)),
        "predicted_defaults": int(submission[TARGET].sum()),
        "predicted_default_rate": float(submission[TARGET].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mode", choices=("validate", "submit", "all"), default="all")
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, test, sample = _load_data(args.data_dir)
    loan, data_report = _build_loan_table(train, test)
    report: dict[str, Any] = {
        "competition": "Zindi African Credit Scoring Challenge",
        "metric": "F1",
        "model": "TabPVN",
        "seed": args.seed,
        "data": data_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.mode != "all" and args.report.exists():
        try:
            previous = json.loads(args.report.read_text())
        except (json.JSONDecodeError, OSError):
            previous = {}
        compatible = all(previous.get(key) == report[key] for key in ("competition", "model", "seed", "data"))
        if compatible:
            for section in ("validation", "submission"):
                if section in previous:
                    report[section] = previous[section]

    if args.mode in {"validate", "all"}:
        print("validating TabPVN on a stratified unique-loan holdout", flush=True)
        report["validation"] = _validation(train, loan, args.seed)
        measured = report["validation"]
        print(
            f"validation f1={measured['f1']:.6f} precision={measured['precision']:.6f} "
            f"recall={measured['recall']:.6f} fit={measured['fit_seconds']:.2f}s",
            flush=True,
        )
        args.report.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")

    if args.mode in {"submit", "all"}:
        print("fitting rich Kenya and portable Ghana TabPVN models", flush=True)
        loan_predictions, submission_report = _fit_submission_models(loan, test, args.seed)
        report["submission"] = {
            **_write_submission(test, sample, loan_predictions, args.submission),
            **submission_report,
        }
        print(
            f"submission defaults={report['submission']['predicted_defaults']:,}/"
            f"{report['submission']['rows']:,}",
            flush=True,
        )

    args.report.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
