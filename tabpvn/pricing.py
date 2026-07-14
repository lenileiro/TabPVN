"""Validated fair-price decisions and calibration certificates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _finite_non_negative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _probability(value: float, name: str = "confidence") -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return result


def _payoff(reward: float, penalty: float, abstain_cost: float = 0.0) -> tuple[float, float, float]:
    reward = _finite_non_negative(reward, "reward")
    penalty = _finite_non_negative(penalty, "penalty")
    abstain_cost = _finite_non_negative(abstain_cost, "abstain_cost")
    if reward + penalty <= 0:
        raise ValueError("reward + penalty must be positive")
    return reward, penalty, abstain_cost


def fair_strike(
    reward: float = 1.0,
    penalty: float = 1.0,
    abstain_cost: float = 0.0,
) -> float:
    """Return the break-even confidence for answering instead of abstaining."""
    reward, penalty, abstain_cost = _payoff(reward, penalty, abstain_cost)
    return min(1.0, max(0.0, (penalty - abstain_cost) / (reward + penalty)))


def answer_ev(p: float, reward: float = 1.0, penalty: float = 1.0) -> float:
    """Return the expected value of answering at confidence p."""
    confidence = _probability(p)
    reward, penalty, _ = _payoff(reward, penalty)
    return confidence * reward - (1.0 - confidence) * penalty


def decide(
    confidences: Iterable[float],
    reward: float = 1.0,
    penalty: float = 1.0,
    abstain_cost: float = 0.0,
) -> tuple[list[bool], float, list[float]]:
    """Apply the fair strike to each confidence."""
    reward, penalty, abstain_cost = _payoff(reward, penalty, abstain_cost)
    values = [_probability(value) for value in confidences]
    strike = fair_strike(reward, penalty, abstain_cost)
    actions = [value >= strike for value in values]
    expected_values = [
        answer_ev(value, reward, penalty) if action else -abstain_cost
        for value, action in zip(values, actions, strict=True)
    ]
    return actions, strike, expected_values


def check_decision(
    actions: Sequence[bool],
    confidences: Sequence[float],
    reward: float,
    penalty: float,
    abstain_cost: float,
    tol: float = 1e-9,
) -> bool:
    """Fail-closed verifier for a claimed fair-strike action vector."""
    try:
        if len(actions) != len(confidences) or not math.isfinite(tol) or tol < 0:
            return False
        strike = fair_strike(reward, penalty, abstain_cost)
        return all(
            bool(action) == (_probability(confidence) >= strike - tol)
            for action, confidence in zip(actions, confidences, strict=True)
        )
    except (TypeError, ValueError):
        return False


def _equal_count_bins(
    pairs: Sequence[tuple[float, float]],
    n_bins: int,
) -> list[list[tuple[float, float]]]:
    """Split confidence/outcome pairs into deterministic equal-count bins."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    count = len(ordered)
    bins = []
    start = 0
    for bin_index in range(n_bins):
        stop = round((bin_index + 1) * count / n_bins)
        if stop > start:
            bins.append(ordered[start:stop])
            start = stop
    return bins


def no_arbitrage_report(
    confidences: Iterable[float],
    correct: Iterable[float],
    n_bins: int = 10,
    delta: float = 0.05,
) -> dict[str, Any]:
    """Bound the worst calibration-bin betting edge with Hoeffding slack."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    delta = float(delta)
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must be a finite probability strictly between 0 and 1")

    confidence_values = [_probability(value) for value in confidences]
    correctness_values = [_probability(value, "correct") for value in correct]
    if len(confidence_values) != len(correctness_values):
        raise ValueError("confidences and correct must have the same length")
    if not confidence_values:
        raise ValueError("no calibration data")
    pairs = list(zip(confidence_values, correctness_values, strict=True))

    bins_out = []
    empirical_edge = 0.0
    certified_edge = 0.0
    worst = None
    for calibration_bin in _equal_count_bins(pairs, n_bins):
        count = len(calibration_bin)
        mean_confidence = sum(confidence for confidence, _ in calibration_bin) / count
        accuracy = sum(outcome for _, outcome in calibration_bin) / count
        gap = accuracy - mean_confidence
        slack = math.sqrt(math.log(2.0 / delta) / (2.0 * count))
        record = {
            "n": count,
            "confidence": mean_confidence,
            "accuracy": accuracy,
            "gap": gap,
            "slack": slack,
        }
        bins_out.append(record)
        empirical_edge = max(empirical_edge, abs(gap))
        if abs(gap) + slack > certified_edge:
            certified_edge = abs(gap) + slack
            worst = record
    return {
        "bins": bins_out,
        "empirical_edge": empirical_edge,
        "certified_edge": certified_edge,
        "arbitrage_bin": worst,
        "delta": delta,
        "n_bins": n_bins,
    }


def check_no_arbitrage(report: Mapping[str, Any], epsilon: float) -> bool:
    """Fail-closed verifier for a no-arbitrage edge claim."""
    try:
        threshold = _finite_non_negative(epsilon, "epsilon")
        edge = _finite_non_negative(report["certified_edge"], "certified_edge")
        return edge <= threshold
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "answer_ev",
    "check_decision",
    "check_no_arbitrage",
    "decide",
    "fair_strike",
    "no_arbitrage_report",
]
