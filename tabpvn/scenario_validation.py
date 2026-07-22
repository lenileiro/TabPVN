"""Deterministic scenario stress tests for verifier candidate evidence.

TabPVN's candidate searches already reuse paired validation blocks.  This module
turns those finite block scores into a small distribution of plausible evidence
mixtures.  Latin-hypercube latent draws cover every marginal stratum, an
empirical shrunk correlation matrix preserves shared block movement, and anchor
scenarios retain each original block exactly.

The result is verifier evidence only: it allocates fit budget and produces an
audit report, but it never changes a proof-carrying prediction directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

_DEFAULT_SCENARIOS = 64
_LOWER_QUANTILE = 0.10
_CORRELATION_SHRINKAGE = 0.50
_SCENARIO_CONCENTRATION = 1.25
_REPORT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def latin_hypercube(samples: int, dimensions: int, *, seed: int = 0) -> np.ndarray:
    """Return deterministic midpoint Latin-hypercube draws in ``(0, 1)``."""

    samples, dimensions = int(samples), int(dimensions)
    if samples < 1 or dimensions < 1:
        raise ValueError("Latin-hypercube dimensions and samples must be positive")
    rng = np.random.default_rng(int(seed))
    draws = np.empty((samples, dimensions), dtype=float)
    for column in range(dimensions):
        draws[:, column] = (rng.permutation(samples) + 0.5) / samples
    return draws


def _validated_utility(utility: Any) -> np.ndarray:
    matrix = np.asarray(utility, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape, default=0) < 1:
        raise ValueError("scenario utility must be a non-empty candidate-by-block matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("scenario utility must contain only finite values")
    return matrix


def _shrunk_block_correlation(utility: np.ndarray) -> np.ndarray:
    """Estimate a positive-definite block correlation without unstable inversion."""

    block_count = utility.shape[1]
    if block_count == 1 or utility.shape[0] == 1:
        return np.eye(block_count, dtype=float)

    centered = utility - utility.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(utility.shape[0] - 1, 1)
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    valid = scale > np.finfo(float).eps
    correlation = np.eye(block_count, dtype=float)
    if np.any(valid):
        denominator = np.outer(scale, scale)
        indices = np.ix_(valid, valid)
        correlation[indices] = covariance[indices] / denominator[indices]
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)

    correlation = (1.0 - _CORRELATION_SHRINKAGE) * correlation + _CORRELATION_SHRINKAGE * np.eye(block_count)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1.0e-8)
    correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(correlation), 1.0e-12))
    correlation = correlation / np.outer(diagonal, diagonal)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _scenario_weights(
    utility: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    block_count = utility.shape[1]
    correlation = _shrunk_block_correlation(utility)
    if block_count == 1:
        return np.ones((1, 1), dtype=float), correlation

    uniform = latin_hypercube(samples, block_count, seed=seed)
    normal = NormalDist()
    latent = np.fromiter(
        (normal.inv_cdf(float(value)) for value in uniform.flat),
        dtype=float,
        count=uniform.size,
    ).reshape(uniform.shape)
    latent = latent @ np.linalg.cholesky(correlation).T
    logits = _SCENARIO_CONCENTRATION * latent
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)

    # The balanced scenario and exact block anchors make the finite support
    # explicit; the stochastic-looking portion remains fully deterministic.
    balanced = np.full((1, block_count), 1.0 / block_count, dtype=float)
    weights = np.vstack((balanced, weights, np.eye(block_count, dtype=float)))
    return weights, correlation


@dataclass(frozen=True)
class ScenarioCandidateSummary:
    """Distribution summary for one candidate across finite stress scenarios."""

    utility_percentiles: tuple[float, ...]
    relative_percentiles: tuple[float, ...]
    lower_relative_utility: float
    baseline_success_rate: float
    group_success_rate: float

    def asdict(self, candidate: int) -> dict[str, Any]:
        utility = dict(zip(("p05", "p25", "median", "p75", "p95"), self.utility_percentiles, strict=True))
        relative = dict(zip(("p05", "p25", "median", "p75", "p95"), self.relative_percentiles, strict=True))
        return {
            "candidate": int(candidate),
            "utility": utility,
            "group_relative_utility": relative,
            "lower_relative_utility": float(self.lower_relative_utility),
            "baseline_success_rate": float(self.baseline_success_rate),
            "group_success_rate": float(self.group_success_rate),
        }


@dataclass(frozen=True)
class ScenarioVerification:
    """Reusable candidate ordering evidence and its serializable audit data."""

    weights: np.ndarray
    correlation: np.ndarray
    lower_relative_utility: np.ndarray
    median_relative_utility: np.ndarray
    summaries: tuple[ScenarioCandidateSummary, ...]
    stratified_scenarios: int
    lower_quantile: float

    def report(
        self,
        candidates: list[int],
        baseline_position: int,
        *,
        role: str = "fit_budget_verifier",
    ) -> dict[str, Any]:
        if len(candidates) != len(self.summaries):
            raise ValueError("candidate identifiers must align with scenario summaries")
        return {
            "method": "correlated_latin_hypercube",
            "role": str(role),
            "scenario_count": int(len(self.weights)),
            "stratified_scenarios": int(self.stratified_scenarios),
            "anchor_scenarios": min(int(len(self.weights)), int(self.weights.shape[1] + 1)),
            "lower_quantile": float(self.lower_quantile),
            "baseline_candidate": int(candidates[baseline_position]),
            "block_correlation": self.correlation.tolist(),
            "candidate_summaries": [
                summary.asdict(candidate)
                for candidate, summary in zip(candidates, self.summaries, strict=True)
            ],
        }


def evaluate_candidate_scenarios(
    utility: Any,
    baseline_position: int,
    *,
    samples: int = _DEFAULT_SCENARIOS,
    lower_quantile: float = _LOWER_QUANTILE,
    seed: int = 0,
) -> ScenarioVerification:
    """Stress candidate utilities under correlated stratified block mixtures."""

    matrix = _validated_utility(utility)
    baseline_position = int(baseline_position)
    if not 0 <= baseline_position < matrix.shape[0]:
        raise ValueError("baseline position is outside the candidate matrix")
    samples = int(samples)
    if samples < 1:
        raise ValueError("scenario sample count must be positive")
    lower_quantile = float(lower_quantile)
    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower scenario quantile must be between zero and one half")

    weights, correlation = _scenario_weights(matrix, samples=samples, seed=int(seed))
    scenario_utility = matrix @ weights.T
    group_center = np.median(matrix, axis=0)
    relative_utility = (matrix - group_center) @ weights.T
    baseline_utility = scenario_utility[baseline_position]
    scale = max(1.0, float(np.max(np.abs(scenario_utility))))
    tolerance = 32.0 * float(np.finfo(float).eps) * scale

    lower = np.quantile(relative_utility, lower_quantile, axis=1)
    median = np.median(relative_utility, axis=1)
    utility_quantiles = np.quantile(scenario_utility, _REPORT_QUANTILES, axis=1).T
    relative_quantiles = np.quantile(relative_utility, _REPORT_QUANTILES, axis=1).T
    baseline_success = np.mean(scenario_utility >= baseline_utility - tolerance, axis=1)
    group_success = np.mean(relative_utility >= -tolerance, axis=1)

    summaries = tuple(
        ScenarioCandidateSummary(
            utility_percentiles=tuple(float(value) for value in utility_quantiles[position]),
            relative_percentiles=tuple(float(value) for value in relative_quantiles[position]),
            lower_relative_utility=float(lower[position]),
            baseline_success_rate=float(baseline_success[position]),
            group_success_rate=float(group_success[position]),
        )
        for position in range(matrix.shape[0])
    )
    return ScenarioVerification(
        weights=weights,
        correlation=correlation,
        lower_relative_utility=np.asarray(lower, dtype=float),
        median_relative_utility=np.asarray(median, dtype=float),
        summaries=summaries,
        stratified_scenarios=(samples if matrix.shape[1] > 1 else 0),
        lower_quantile=lower_quantile,
    )


__all__ = [
    "ScenarioCandidateSummary",
    "ScenarioVerification",
    "evaluate_candidate_scenarios",
    "latin_hypercube",
]
