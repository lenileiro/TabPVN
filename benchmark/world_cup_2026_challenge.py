"""Causal TabPVN entry for Zindi's World Cup 2026 prediction challenge.

Only the supplied challenge files are used. Every historical feature is built
from tournaments completed before the row being predicted, and model selection
uses complete future-tournament holdouts rather than random team splits.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score, mean_squared_error

from tabpvn import TabPVN

DEFAULT_DATA_DIR = Path("data/world-cup-2026-goal-prediction-challenge20260612-26558-nuw9mm")
DEFAULT_SUBMISSION = Path("results/world_cup_2026_tabpvn_submission.csv")
DEFAULT_REPORT = Path("results/world_cup_2026_tabpvn_report.json")
BACKTEST_YEARS = (2006, 2010, 2014, 2018, 2022)

TARGET_GOALS = "total_goals"
TARGET_STAGE = "stage_label"
TARGET_MATCHES = "target_matches"
STAGE_ORDER = ("group", "roundof32", "roundof16", "qf", "sf", "runnerup", "champion")
STAGE_PROGRESS = {
    "group": 0.0,
    "roundof32": 0.2,
    "roundof16": 0.4,
    "qf": 0.6,
    "sf": 0.8,
    "runnerup": 0.9,
    "champion": 1.0,
}

_CATEGORICAL_FEATURES = ("team_code", "confederation_name", "region_name")
_HISTORY_METRICS = (
    "goals_per_match",
    "goals_against_per_match",
    "goal_difference_per_match",
    "win_rate",
    "draw_rate",
    "points_per_match",
    "progress",
    "total_goals",
    "matches_played",
)
_STAGE_COMPONENTS = (
    ("TabPVN progress", "stage_model"),
    ("World-Cup Elo", "elo"),
    ("rolling progress", "history"),
    ("best prior finish", "prior_best"),
    ("knockout pedigree", "pedigree"),
)
_GOAL_COMPONENTS = (
    "TabPVN total",
    "TabPVN log-total",
    "TabPVN square-root total",
    "TabPVN rate x assigned-stage matches",
    "TabPVN rate x expected matches",
)
_MAX_STAGE_COMPONENT_WEIGHT = 0.8
_DEFAULT_STAGE_WEIGHTS = (0.3, 0.0, 0.2, 0.3, 0.2)
_DEFAULT_GOAL_STAGE_WEIGHTS = (0.4, 0.3, 0.1, 0.0, 0.2)
_DEFAULT_GOAL_WEIGHTS = (0.0, 0.0, 0.1, 0.9, 0.0)


def _validated_selection() -> dict[str, Any]:
    """Return the fixed architecture selected by complete future-edition folds."""
    return {
        "protocol": "prevalidated complete future-tournament holdouts",
        "years": list(BACKTEST_YEARS),
        "selection_source": "fixed defaults selected by five-edition backtest",
        "stage_components": [name for name, _ in _STAGE_COMPONENTS],
        "stage_weights": _DEFAULT_STAGE_WEIGHTS,
        "stage_max_component_weight": _MAX_STAGE_COMPONENT_WEIGHT,
        "stage_tie_break": "mean rank across all stage components",
        "stage": {
            "macro_f1": 0.30416666666666664,
            "weighted_f1": 0.55625,
        },
        "goal_stage_components": [name for name, _ in _STAGE_COMPONENTS],
        "goal_stage_weights": _DEFAULT_GOAL_STAGE_WEIGHTS,
        "goal_components": list(_GOAL_COMPONENTS),
        "goal_weights": _DEFAULT_GOAL_WEIGHTS,
        "goals_rmse": 3.063154571547398,
        "meta_validation": {
            "protocol": "leave-one-edition-out selection of ensemble weights",
            "stage": {
                "macro_f1": 0.2791666666666667,
                "weighted_f1": 0.51875,
            },
            "goals_rmse": 3.1920893641717933,
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "train": data_dir / "Train.csv",
        "test": data_dir / "Test.csv",
        "sample": data_dir / "SampleSubmission.csv",
        "tournaments": data_dir / "data/tournaments.csv",
        "standings": data_dir / "data/tournament_standings.csv",
        "appearances": data_dir / "data/team_appearances.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"challenge files are missing: {missing}")
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    if list(frames["sample"].columns) != ["ID", "total_goals", "Target"]:
        raise ValueError("SampleSubmission.csv must have ID,total_goals,Target columns")
    if not frames["sample"]["ID"].equals(frames["test"]["ID"]):
        raise ValueError("sample submission IDs must match Test.csv in order")
    return frames


def _stage_label(
    stage: str,
    country: str,
    winner: str,
    position: float,
) -> str:
    if stage == "group stage":
        return "group"
    if stage == "round of 16":
        return "roundof16"
    if stage in {"quarter-finals", "second group stage"}:
        return "qf"
    if stage in {"semi-finals", "third-place match"}:
        return "sf"
    if stage == "final":
        return "champion" if country == winner else "runnerup"
    if stage == "final round":
        if position == 1:
            return "champion"
        if position == 2:
            return "runnerup"
        return "sf"
    raise ValueError(f"unsupported historical stage: {stage!r}")


def _historical_records(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train = frames["train"].copy()
    tournament = frames["tournaments"].loc[
        frames["tournaments"]["tournament_id"].isin(train["tournament_id"]),
        ["tournament_id", "winner", "count_teams"],
    ]
    standings = frames["standings"].loc[
        frames["standings"]["tournament_id"].isin(train["tournament_id"]),
        ["tournament_id", "team_code", "position"],
    ]
    appearances = frames["appearances"].loc[
        frames["appearances"]["tournament_id"].isin(train["tournament_id"])
    ]
    aggregate = (
        appearances.groupby(["tournament_id", "team_code"], sort=False)
        .agg(
            appearance_matches=("match_id", "nunique"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            wins=("win", "sum"),
            draws=("draw", "sum"),
            losses=("lose", "sum"),
        )
        .reset_index()
    )
    train = train.merge(tournament, on="tournament_id", how="left", validate="many_to_one")
    train = train.merge(
        standings,
        on=["tournament_id", "team_code"],
        how="left",
        validate="one_to_one",
    )
    train = train.merge(
        aggregate,
        on=["tournament_id", "team_code"],
        how="left",
        validate="one_to_one",
    )
    if train[["winner", "count_teams", "appearance_matches"]].isna().any().any():
        raise AssertionError("historical tournament joins are incomplete")
    if not train["appearance_matches"].astype(int).equals(train["matches_played"]):
        raise AssertionError("match appearances do not reproduce matches_played")
    if not train["goals_for"].astype(int).equals(train[TARGET_GOALS]):
        raise AssertionError("match appearances do not reproduce total_goals")

    train[TARGET_STAGE] = [
        _stage_label(stage, country, winner, position)
        for stage, country, winner, position in zip(
            train["stage_reached"],
            train["country"],
            train["winner"],
            train["position"],
            strict=True,
        )
    ]
    train["progress"] = train[TARGET_STAGE].map(STAGE_PROGRESS).astype(float)
    train["goals_per_match"] = train[TARGET_GOALS] / train["matches_played"]
    train["goals_against_per_match"] = train["goals_against"] / train["matches_played"]
    train["goal_difference_per_match"] = (train["goals_for"] - train["goals_against"]) / train[
        "matches_played"
    ]
    train["win_rate"] = train["wins"] / train["matches_played"]
    train["draw_rate"] = train["draws"] / train["matches_played"]
    train["points_per_match"] = (3 * train["wins"] + train["draws"]) / train["matches_played"]
    train["log_total_goals"] = np.log1p(train[TARGET_GOALS])
    return train.sort_values(["year", "team_code"], kind="stable").reset_index(drop=True)


def _elo_snapshots(
    records: pd.DataFrame, appearances: pd.DataFrame
) -> dict[tuple[int, str], tuple[float, int]]:
    appearances = appearances.loc[appearances["tournament_id"].isin(records["tournament_id"])]
    matches = (
        appearances.sort_values(["match_date", "match_id", "home_team"], kind="stable")
        .drop_duplicates("match_id")
        .loc[
            :,
            [
                "tournament_id",
                "match_id",
                "team_code",
                "opponent_code",
                "goals_for",
                "goals_against",
                "win",
                "draw",
            ],
        ]
    )
    year_by_tournament = records.drop_duplicates("tournament_id").set_index("tournament_id")["year"]
    ratings: dict[str, float] = {}
    games: dict[str, int] = {}
    snapshots: dict[tuple[int, str], tuple[float, int]] = {}
    for year in sorted(records["year"].unique()):
        teams = records.loc[records["year"].eq(year), "team_code"]
        for code in teams:
            snapshots[(int(year), code)] = (ratings.get(code, 1500.0), games.get(code, 0))
        tournament_ids = year_by_tournament.index[year_by_tournament.eq(year)]
        for row in matches.loc[matches["tournament_id"].isin(tournament_ids)].itertuples(index=False):
            left, right = row.team_code, row.opponent_code
            left_rating = ratings.get(left, 1500.0)
            right_rating = ratings.get(right, 1500.0)
            expected = 1.0 / (1.0 + 10.0 ** ((right_rating - left_rating) / 400.0))
            actual = 1.0 if row.win else 0.5 if row.draw else 0.0
            margin = abs(float(row.goals_for) - float(row.goals_against))
            multiplier = 1.0 + 0.20 * max(0.0, margin - 1.0)
            change = 24.0 * multiplier * (actual - expected)
            ratings[left] = left_rating + change
            ratings[right] = right_rating - change
            games[left] = games.get(left, 0) + 1
            games[right] = games.get(right, 0) + 1
    for code in set(records["team_code"]):
        snapshots[(2026, code)] = (ratings.get(code, 1500.0), games.get(code, 0))
    return snapshots


def _weighted_mean(history: pd.DataFrame, metric: str, year: int, half_life: float) -> float:
    if history.empty:
        return math.nan
    age = year - history["year"].to_numpy(float)
    weights = np.exp2(-age / half_life)
    if metric in {
        "goals_per_match",
        "goals_against_per_match",
        "goal_difference_per_match",
        "win_rate",
        "draw_rate",
        "points_per_match",
    }:
        weights = weights * history["matches_played"].to_numpy(float)
    return float(np.average(history[metric].to_numpy(float), weights=weights))


def _history_features(
    history: pd.DataFrame,
    year: int,
    prefix: str,
    *,
    summarize_editions: bool = False,
) -> dict[str, float]:
    result: dict[str, float] = {
        f"{prefix}_appearances": float(len(history)),
        f"{prefix}_matches": float(history["matches_played"].sum()),
    }
    sequence = history
    if summarize_editions and not history.empty:
        sequence = history.groupby("year", as_index=False)[list(_HISTORY_METRICS)].mean()
    for metric in _HISTORY_METRICS:
        result[f"{prefix}_{metric}_ewm12"] = _weighted_mean(sequence, metric, year, 12.0)
        result[f"{prefix}_{metric}_ewm24"] = _weighted_mean(sequence, metric, year, 24.0)
        result[f"{prefix}_{metric}_last"] = math.nan if sequence.empty else float(sequence.iloc[-1][metric])
        result[f"{prefix}_{metric}_last3"] = (
            math.nan if sequence.empty else float(sequence.tail(3)[metric].mean())
        )
    return result


def _feature_row(
    records: pd.DataFrame,
    code: str,
    confederation: str,
    region: str,
    year: int,
    tournament_teams: int,
    elo: tuple[float, int],
) -> dict[str, Any]:
    prior = records.loc[records["year"].lt(year)]
    team_history = prior.loc[prior["team_code"].eq(code)]
    confed_history = prior.loc[prior["confederation_name"].eq(confederation)]
    recent_global = prior.loc[prior["year"].ge(year - 16)]
    last_year = math.nan if team_history.empty else float(team_history["year"].max())
    result: dict[str, Any] = {
        "team_code": code,
        "confederation_name": confederation,
        "region_name": region,
        "year": float(year),
        "years_since_last": math.nan if math.isnan(last_year) else year - last_year,
        "tournament_teams": float(tournament_teams),
        "knockout_share": 32.0 / tournament_teams if tournament_teams >= 32 else 8.0 / tournament_teams,
        "elo": float(elo[0]),
        "elo_games": float(elo[1]),
        "prior_titles": float(team_history[TARGET_STAGE].eq("champion").sum()),
        "prior_finals": float(team_history[TARGET_STAGE].isin(["champion", "runnerup"]).sum()),
        "prior_semifinals": float(team_history[TARGET_STAGE].isin(["champion", "runnerup", "sf"]).sum()),
        "prior_best_progress": (math.nan if team_history.empty else float(team_history["progress"].max())),
        "global_recent_goal_rate": (
            math.nan
            if recent_global.empty
            else float(recent_global[TARGET_GOALS].sum() / recent_global["matches_played"].sum())
        ),
        "global_recent_goals_against_rate": (
            math.nan
            if recent_global.empty
            else float(recent_global["goals_against"].sum() / recent_global["matches_played"].sum())
        ),
    }
    result.update(_history_features(team_history, year, "team"))
    result.update(_history_features(confed_history, year, "confed", summarize_editions=True))
    return result


def _build_feature_table(
    records: pd.DataFrame,
    test: pd.DataFrame,
    appearances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    elo = _elo_snapshots(records, appearances)
    historical_rows = []
    for row in records.itertuples(index=False):
        feature = _feature_row(
            records,
            row.team_code,
            row.confederation_name,
            row.region_name,
            int(row.year),
            int(row.count_teams),
            elo.get((int(row.year), row.team_code), (1500.0, 0)),
        )
        feature.update(
            {
                "ID": row.ID,
                TARGET_GOALS: float(row.total_goals),
                TARGET_STAGE: row.stage_label,
                "progress": float(row.progress),
                "goals_per_match": float(row.goals_per_match),
                "log_total_goals": float(row.log_total_goals),
                "sqrt_total_goals": float(math.sqrt(row.total_goals)),
                TARGET_MATCHES: float(row.matches_played),
            }
        )
        historical_rows.append(feature)

    latest_metadata = (
        records.sort_values("year", kind="stable")
        .drop_duplicates("team_code", keep="last")
        .set_index("team_code")[["confederation_name", "region_name"]]
    )
    test_rows = []
    for row in test.itertuples(index=False):
        code = row.ID.rsplit("_", 1)[-1]
        if code in latest_metadata.index:
            confederation, region = latest_metadata.loc[code]
        else:
            confederation, region = "unseen", "unseen"
        feature = _feature_row(
            records,
            code,
            str(confederation),
            str(region),
            2026,
            len(test),
            elo.get((2026, code), (1500.0, 0)),
        )
        feature["ID"] = row.ID
        test_rows.append(feature)

    historical = pd.DataFrame(historical_rows)
    test_features = pd.DataFrame(test_rows)
    for column in _CATEGORICAL_FEATURES:
        historical[column] = historical[column].astype(str)
        test_features[column] = test_features[column].astype(str)
    return historical, test_features


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "ID",
        TARGET_GOALS,
        TARGET_STAGE,
        TARGET_MATCHES,
        "progress",
        "goals_per_match",
        "log_total_goals",
        "sqrt_total_goals",
    }
    return [column for column in frame.columns if column not in excluded]


def _fit_regression(features: pd.DataFrame, target: np.ndarray, seed: int) -> TabPVN:
    model = TabPVN(seed=seed, task="regression")
    model.fit(features, target)
    return model


def _rank_unit(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method="average") - 1.0
    return ranks / max(len(ranks) - 1, 1)


def _stage_component_matrix(fold: dict[str, Any]) -> np.ndarray:
    return np.column_stack([_rank_unit(fold[key]) for _, key in _STAGE_COMPONENTS])


def _stage_score(fold: dict[str, Any], weights: tuple[float, ...]) -> np.ndarray:
    components = _stage_component_matrix(fold)
    primary = components @ np.asarray(weights)
    return primary + 1e-9 * components.mean(axis=1)


def _stage_quotas(team_count: int) -> dict[str, int]:
    if team_count >= 48:
        return {
            "group": team_count - 32,
            "roundof32": 16,
            "roundof16": 8,
            "qf": 4,
            "sf": 2,
            "runnerup": 1,
            "champion": 1,
        }
    if team_count >= 24:
        return {
            "group": team_count - 16,
            "roundof32": 0,
            "roundof16": 8,
            "qf": 4,
            "sf": 2,
            "runnerup": 1,
            "champion": 1,
        }
    return {
        "group": team_count - 8,
        "roundof32": 0,
        "roundof16": 0,
        "qf": 4,
        "sf": 2,
        "runnerup": 1,
        "champion": 1,
    }


def _assign_stages(score: np.ndarray, team_count: int) -> np.ndarray:
    quotas = _stage_quotas(team_count)
    slots = [stage for stage in STAGE_ORDER for _ in range(quotas[stage])]
    if len(slots) != len(score):
        raise AssertionError("stage quotas must match the number of teams")
    output = np.empty(len(score), dtype=object)
    output[np.argsort(score, kind="stable")] = slots
    return output


def _stage_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed = set(actual) | set(prediction)
    labels = [stage for stage in STAGE_ORDER if stage in observed]
    return {
        "macro_f1": float(f1_score(actual, prediction, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, prediction, average="weighted", zero_division=0)),
    }


def _expected_matches(stage: np.ndarray, team_count: int) -> np.ndarray:
    if team_count >= 48:
        mapping = {
            "group": 3,
            "roundof32": 4,
            "roundof16": 5,
            "qf": 6,
            "sf": 8,
            "runnerup": 8,
            "champion": 8,
        }
    else:
        mapping = {
            "group": 3,
            "roundof32": 4,
            "roundof16": 4,
            "qf": 5,
            "sf": 7,
            "runnerup": 7,
            "champion": 7,
        }
    return np.asarray([mapping[value] for value in stage], dtype=float)


def _model_fold(
    historical: pd.DataFrame,
    year: int,
    seed: int,
) -> dict[str, Any]:
    columns = _feature_columns(historical)
    fit = historical["year"].lt(year)
    valid = historical["year"].eq(year)
    x_fit = historical.loc[fit, columns]
    x_valid = historical.loc[valid, columns]
    targets = {
        "stage": historical.loc[fit, "progress"].to_numpy(float),
        "rate": historical.loc[fit, "goals_per_match"].to_numpy(float),
        "total": historical.loc[fit, TARGET_GOALS].to_numpy(float),
        "log_total": historical.loc[fit, "log_total_goals"].to_numpy(float),
        "sqrt_total": historical.loc[fit, "sqrt_total_goals"].to_numpy(float),
        "matches": historical.loc[fit, TARGET_MATCHES].to_numpy(float),
    }
    predictions = {}
    for offset, (name, target) in enumerate(targets.items()):
        model = _fit_regression(x_fit, target, seed + offset)
        prediction = np.asarray(model.predict(x_valid), dtype=float)
        if name == "log_total":
            prediction = np.expm1(prediction)
        elif name == "sqrt_total":
            prediction = np.square(np.clip(prediction, 0.0, None))
        predictions[name] = prediction
    valid_frame = historical.loc[valid].reset_index(drop=True)
    return {
        "year": year,
        "actual_stage": valid_frame[TARGET_STAGE].to_numpy(object),
        "actual_goals": valid_frame[TARGET_GOALS].to_numpy(float),
        "stage_model": predictions["stage"],
        "elo": valid_frame["elo"].to_numpy(float),
        "history": valid_frame["team_progress_ewm12"].fillna(0.0).to_numpy(float),
        "prior_best": valid_frame["prior_best_progress"].fillna(0.0).to_numpy(float),
        "pedigree": valid_frame["prior_semifinals"].to_numpy(float),
        "rate": predictions["rate"],
        "total": predictions["total"],
        "log_total": predictions["log_total"],
        "sqrt_total": predictions["sqrt_total"],
        "matches": predictions["matches"],
        "team_count": int(valid.sum()),
    }


def _simplex_grid(step: float, dimensions: int) -> list[tuple[float, ...]]:
    units = round(1.0 / step)

    def compositions(total: int, parts: int) -> list[tuple[int, ...]]:
        if parts == 1:
            return [(total,)]
        return [(head, *tail) for head in range(total + 1) for tail in compositions(total - head, parts - 1)]

    return [tuple(value / units for value in values) for values in compositions(units, dimensions)]


def _stage_predictions(
    folds: list[dict[str, Any]], weights: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.concatenate([fold["actual_stage"] for fold in folds])
    predictions = []
    for fold in folds:
        score = _stage_score(fold, weights)
        predictions.append(_assign_stages(score, fold["team_count"]))
    return actual, np.concatenate(predictions)


def _select_stage_blend(folds: list[dict[str, Any]]) -> tuple[tuple[float, ...], dict[str, float]]:
    best: tuple[float, float, tuple[float, ...], np.ndarray] | None = None
    for weights in _simplex_grid(0.10, len(_STAGE_COMPONENTS)):
        if max(weights) > _MAX_STAGE_COMPONENT_WEIGHT:
            continue
        actual, prediction = _stage_predictions(folds, weights)
        metrics = _stage_metrics(actual, prediction)
        macro = metrics["macro_f1"]
        weighted = metrics["weighted_f1"]
        candidate = ((macro + weighted) / 2.0, macro, weights, prediction)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return best[2], _stage_metrics(actual, best[3])


def _fold_stage(fold: dict[str, Any], weights: tuple[float, ...]) -> np.ndarray:
    score = _stage_score(fold, weights)
    return _assign_stages(score, fold["team_count"])


def _goal_component_matrix(fold: dict[str, Any], stage: np.ndarray) -> np.ndarray:
    assigned_matches = _expected_matches(stage, fold["team_count"])
    maximum_matches = 8.0 if fold["team_count"] >= 48 else 7.0
    expected_matches = np.clip(fold["matches"], 3.0, maximum_matches)
    return np.column_stack(
        [
            fold["total"],
            fold["log_total"],
            fold["sqrt_total"],
            fold["rate"] * assigned_matches,
            fold["rate"] * expected_matches,
        ]
    )


def _select_goal_stage_blend(folds: list[dict[str, Any]]) -> tuple[tuple[float, ...], float]:
    actual = np.concatenate([fold["actual_goals"] for fold in folds])
    best: tuple[float, tuple[float, ...]] | None = None
    for weights in _simplex_grid(0.10, len(_STAGE_COMPONENTS)):
        predictions = []
        for fold in folds:
            stage = _fold_stage(fold, weights)
            matches = _expected_matches(stage, fold["team_count"])
            predictions.append(np.clip(fold["rate"] * matches, 0.0, 30.0))
        rmse = float(mean_squared_error(actual, np.concatenate(predictions)) ** 0.5)
        candidate = (rmse, weights)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], best[0]


def _select_goal_blend(
    folds: list[dict[str, Any]],
    goal_stage_weights: tuple[float, ...],
) -> tuple[tuple[float, ...], float]:
    actual = np.concatenate([fold["actual_goals"] for fold in folds])
    component_rows = []
    for fold in folds:
        stage = _fold_stage(fold, goal_stage_weights)
        component_rows.append(_goal_component_matrix(fold, stage))
    components = np.concatenate(component_rows)
    best: tuple[float, tuple[float, ...]] | None = None
    for weights in _simplex_grid(0.10, len(_GOAL_COMPONENTS)):
        prediction = np.clip(components @ np.asarray(weights), 0.0, 30.0)
        rmse = float(mean_squared_error(actual, prediction) ** 0.5)
        candidate = (rmse, weights)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], best[0]


def _goal_predictions(
    folds: list[dict[str, Any]],
    goal_stage_weights: tuple[float, ...],
    goal_weights: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.concatenate([fold["actual_goals"] for fold in folds])
    predictions = []
    for fold in folds:
        stage = _fold_stage(fold, goal_stage_weights)
        components = _goal_component_matrix(fold, stage)
        predictions.append(np.clip(components @ np.asarray(goal_weights), 0.0, 30.0))
    return actual, np.concatenate(predictions)


def _component_metrics(
    folds: list[dict[str, Any]],
    stage_weights: tuple[float, ...],
    goal_stage_weights: tuple[float, ...],
    goal_weights: tuple[float, ...],
) -> dict[str, Any]:
    stage = {}
    for index, (name, _) in enumerate(_STAGE_COMPONENTS):
        weights = tuple(float(position == index) for position in range(len(_STAGE_COMPONENTS)))
        actual, prediction = _stage_predictions(folds, weights)
        stage[name] = _stage_metrics(actual, prediction)
    actual_stage, blend_stage = _stage_predictions(folds, stage_weights)
    stage["selected blend"] = _stage_metrics(actual_stage, blend_stage)

    goals = {}
    for index, name in enumerate(_GOAL_COMPONENTS):
        weights = tuple(float(position == index) for position in range(len(_GOAL_COMPONENTS)))
        actual, prediction = _goal_predictions(folds, goal_stage_weights, weights)
        goals[name] = float(mean_squared_error(actual, prediction) ** 0.5)
    actual_goals, blend_goals = _goal_predictions(folds, goal_stage_weights, goal_weights)
    goals["selected blend"] = float(mean_squared_error(actual_goals, blend_goals) ** 0.5)
    return {"stage": stage, "goals_rmse": goals}


def _meta_cross_validation(folds: list[dict[str, Any]]) -> dict[str, Any]:
    actual_stages = []
    predicted_stages = []
    actual_goals = []
    predicted_goals = []
    selected_weights = []
    for held_out in range(len(folds)):
        fit_folds = [fold for index, fold in enumerate(folds) if index != held_out]
        valid_fold = folds[held_out]
        stage_weights, _ = _select_stage_blend(fit_folds)
        goal_stage_weights, _ = _select_goal_stage_blend(fit_folds)
        goal_weights, _ = _select_goal_blend(fit_folds, goal_stage_weights)
        stage = _fold_stage(valid_fold, stage_weights)
        goal_stage = _fold_stage(valid_fold, goal_stage_weights)
        components = _goal_component_matrix(valid_fold, goal_stage)
        goals = np.clip(components @ np.asarray(goal_weights), 0.0, 30.0)
        actual_stages.append(valid_fold["actual_stage"])
        predicted_stages.append(stage)
        actual_goals.append(valid_fold["actual_goals"])
        predicted_goals.append(goals)
        selected_weights.append(
            {
                "held_out_year": valid_fold["year"],
                "stage": stage_weights,
                "goal_stage": goal_stage_weights,
                "goals": goal_weights,
            }
        )
    actual_stage = np.concatenate(actual_stages)
    predicted_stage = np.concatenate(predicted_stages)
    return {
        "protocol": "leave-one-edition-out selection of ensemble weights",
        "stage": _stage_metrics(actual_stage, predicted_stage),
        "goals_rmse": float(
            mean_squared_error(np.concatenate(actual_goals), np.concatenate(predicted_goals)) ** 0.5
        ),
        "selected_weights": selected_weights,
    }


def _backtest(historical: pd.DataFrame, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folds = []
    for index, year in enumerate(BACKTEST_YEARS):
        print(f"backtesting {year}", flush=True)
        folds.append(_model_fold(historical, year, seed + 20 * index))
    stage_weights, stage_metrics = _select_stage_blend(folds)
    goal_stage_weights, goal_stage_rmse = _select_goal_stage_blend(folds)
    goal_weights, goal_rmse = _select_goal_blend(folds, goal_stage_weights)
    component_metrics = _component_metrics(
        folds,
        stage_weights,
        goal_stage_weights,
        goal_weights,
    )
    meta_validation = _meta_cross_validation(folds)
    per_year = []
    for fold in folds:
        stage = _fold_stage(fold, stage_weights)
        goal_stage = _fold_stage(fold, goal_stage_weights)
        components = _goal_component_matrix(fold, goal_stage)
        goals = np.clip(components @ np.asarray(goal_weights), 0.0, 30.0)
        per_year.append(
            {
                "year": fold["year"],
                "goals_rmse": float(mean_squared_error(fold["actual_goals"], goals) ** 0.5),
                "stage_macro_f1": float(_stage_metrics(fold["actual_stage"], stage)["macro_f1"]),
                "stage_weighted_f1": float(
                    f1_score(fold["actual_stage"], stage, average="weighted", zero_division=0)
                ),
            }
        )
    return folds, {
        "years": list(BACKTEST_YEARS),
        "stage_components": [name for name, _ in _STAGE_COMPONENTS],
        "stage_weights": stage_weights,
        "stage_max_component_weight": _MAX_STAGE_COMPONENT_WEIGHT,
        "stage_tie_break": "mean rank across all stage components",
        "stage": stage_metrics,
        "goal_stage_components": [name for name, _ in _STAGE_COMPONENTS],
        "goal_stage_weights": goal_stage_weights,
        "goal_stage_rate_rmse": goal_stage_rmse,
        "goal_components": list(_GOAL_COMPONENTS),
        "goal_weights": goal_weights,
        "goals_rmse": goal_rmse,
        "component_metrics": component_metrics,
        "meta_validation": meta_validation,
        "per_year": per_year,
    }


def _fit_final(
    historical: pd.DataFrame,
    test_features: pd.DataFrame,
    stage_weights: tuple[float, ...],
    goal_stage_weights: tuple[float, ...],
    goal_weights: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    columns = _feature_columns(historical)
    x_train = historical[columns]
    x_test = test_features[columns]
    targets = (
        ("stage", historical["progress"].to_numpy(float), 0, True),
        (
            "rate",
            historical["goals_per_match"].to_numpy(float),
            1,
            goal_weights[3] > 0.0 or goal_weights[4] > 0.0,
        ),
        ("total", historical[TARGET_GOALS].to_numpy(float), 2, goal_weights[0] > 0.0),
        ("log_total", historical["log_total_goals"].to_numpy(float), 3, goal_weights[1] > 0.0),
        ("sqrt_total", historical["sqrt_total_goals"].to_numpy(float), 4, goal_weights[2] > 0.0),
        ("matches", historical[TARGET_MATCHES].to_numpy(float), 5, goal_weights[4] > 0.0),
    )
    predictions = {}
    fit_seconds = {}
    skipped_heads = []
    for name, target, offset, active in targets:
        if not active:
            skipped_heads.append(name)
            continue
        started = time.perf_counter()
        model = _fit_regression(x_train, target, seed + offset)
        prediction = np.asarray(model.predict(x_test), dtype=float)
        if name == "log_total":
            prediction = np.expm1(prediction)
        elif name == "sqrt_total":
            prediction = np.square(np.clip(prediction, 0.0, None))
        predictions[name] = prediction
        fit_seconds[name] = time.perf_counter() - started

    stage_components = {
        "stage_model": predictions["stage"],
        "elo": test_features["elo"].to_numpy(float),
        "history": test_features["team_progress_ewm12"].fillna(0.0).to_numpy(float),
        "prior_best": test_features["prior_best_progress"].fillna(0.0).to_numpy(float),
        "pedigree": test_features["prior_semifinals"].to_numpy(float),
    }
    score = _stage_score(stage_components, stage_weights)
    stage = _assign_stages(score, len(test_features))
    goal_stage_score = _stage_score(stage_components, goal_stage_weights)
    goal_stage = _assign_stages(goal_stage_score, len(test_features))
    zeros = np.zeros(len(test_features), dtype=float)
    goal_components = {
        "team_count": len(test_features),
        "total": predictions.get("total", zeros),
        "log_total": predictions.get("log_total", zeros),
        "sqrt_total": predictions.get("sqrt_total", zeros),
        "rate": predictions.get("rate", zeros),
        "matches": predictions.get("matches", np.full(len(test_features), 3.0)),
    }
    components = _goal_component_matrix(goal_components, goal_stage)
    goals = np.clip(components @ np.asarray(goal_weights), 0.0, 30.0)
    return (
        goals,
        stage,
        {
            "fit_seconds": fit_seconds,
            "active_heads": list(predictions),
            "skipped_zero_weight_heads": skipped_heads,
            "predicted_total_goals": float(goals.sum()),
            "stage_counts": pd.Series(stage).value_counts().reindex(STAGE_ORDER, fill_value=0).to_dict(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="rerun architecture selection; default prediction uses the prevalidated weights",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    frames = _load_data(args.data_dir)
    records = _historical_records(frames)
    historical, test_features = _build_feature_table(records, frames["test"], frames["appearances"])
    if args.backtest:
        _, validation = _backtest(historical, args.seed)
        validation["selection_source"] = "fresh five-edition backtest"
    else:
        validation = _validated_selection()
    goals, stage, final_report = _fit_final(
        historical,
        test_features,
        tuple(validation["stage_weights"]),
        tuple(validation["goal_stage_weights"]),
        tuple(validation["goal_weights"]),
        args.seed,
    )

    submission = frames["sample"].copy()
    submission[TARGET_GOALS] = np.round(goals, 6)
    submission["Target"] = stage
    if not submission["ID"].equals(frames["test"]["ID"]):
        raise AssertionError("submission IDs changed order")
    if not set(submission["Target"]).issubset(STAGE_ORDER):
        raise AssertionError("submission contains an invalid stage")
    expected_stage_counts = _stage_quotas(len(submission))
    stage_counts = submission["Target"].value_counts().reindex(STAGE_ORDER, fill_value=0).to_dict()
    if stage_counts != expected_stage_counts:
        raise AssertionError(f"submission stage counts differ from the bracket: {stage_counts}")
    if not np.isfinite(submission[TARGET_GOALS]).all() or submission[TARGET_GOALS].lt(0).any():
        raise AssertionError("submission goals must be finite and non-negative")
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission, index=False)

    report = {
        "competition": "Zindi World Cup 2026 Goal Prediction Challenge",
        "closed_data": True,
        "model": "TabPVN causal tournament ensemble",
        "seed": args.seed,
        "data": {
            "historical_team_tournaments": len(historical),
            "historical_editions": int(records["year"].nunique()),
            "test_teams": len(test_features),
            "test_teams_without_world_cup_history": int(test_features["team_appearances"].eq(0).sum()),
            "features": len(_feature_columns(historical)),
        },
        "validation": validation,
        "submission": {
            "path": str(args.submission),
            **final_report,
        },
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(
        f"validation RMSE={validation['goals_rmse']:.4f} "
        f"macro-F1={validation['stage']['macro_f1']:.4f} "
        f"weighted-F1={validation['stage']['weighted_f1']:.4f}",
        flush=True,
    )
    print(f"wrote {args.submission}", flush=True)
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
