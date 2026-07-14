"""Validated Bayesian updates used by TabPVN's opt-in decision layer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

BinaryTest = tuple[float, float, bool]


def _normalized_probability_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional probability vector")
    if expected_size is not None and vector.size != expected_size:
        raise ValueError(f"{name} has {vector.size} classes; expected {expected_size}")
    if not np.isfinite(vector).all() or np.any(vector < 0):
        raise ValueError(f"{name} must contain finite, non-negative probabilities")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass")
    return vector / total


def as_prior(prior: Mapping[Any, float] | ArrayLike, classes: Sequence[Any]) -> NDArray[np.float64]:
    """Normalize a class mapping or a class-aligned probability vector."""
    class_values = tuple(classes)
    if not class_values:
        raise ValueError("classes must not be empty")
    if isinstance(prior, Mapping):
        values = np.array([float(prior.get(label, 0.0)) for label in class_values], dtype=float)
    else:
        values = np.asarray(prior, dtype=float)
    return _normalized_probability_vector(values, name="prior", expected_size=len(class_values))


def prior_shift(
    proba: ArrayLike,
    prior_train: ArrayLike,
    prior_deploy: ArrayLike,
    eps: float = 1e-12,
    *,
    strength: float = 1.0,
) -> NDArray[np.float64]:
    """Correct calibrated posteriors when only the deployment class prior changes.

    ``strength=1`` is the exact prior-shift update. Intermediate strengths
    interpolate geometrically between the calibrated training posterior and
    that update; ``strength=0`` is the identity. The interpolation is useful
    only as a separately validated ranking projection, never as an exact
    deployment posterior.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number")
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be a finite number in [0, 1]")
    probability = np.asarray(proba, dtype=float)
    original_dimension = probability.ndim
    if original_dimension not in {1, 2}:
        raise ValueError("proba must be a one- or two-dimensional probability array")
    if probability.size == 0:
        raise ValueError("proba must not be empty")
    matrix = np.atleast_2d(probability)
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("proba must contain finite, non-negative probabilities")
    if np.any(matrix.sum(axis=1) <= 0):
        raise ValueError("every proba row must have positive total mass")

    class_count = matrix.shape[1]
    training_prior = _normalized_probability_vector(
        prior_train,
        name="prior_train",
        expected_size=class_count,
    )
    deployment_prior = _normalized_probability_vector(
        prior_deploy,
        name="prior_deploy",
        expected_size=class_count,
    )
    ratio = deployment_prior / np.maximum(training_prior, eps)
    corrected = matrix * np.power(ratio, strength)
    row_mass = corrected.sum(axis=1, keepdims=True)
    if np.any(row_mass <= eps):
        raise ValueError("prior shift assigns zero mass to at least one probability row")
    corrected /= row_mass
    return corrected[0] if original_dimension == 1 else corrected


def check_prior_shift(
    proba: ArrayLike,
    prior_train: ArrayLike,
    prior_deploy: ArrayLike,
    claimed: ArrayLike,
    tol: float = 1e-9,
) -> bool:
    """Fail-closed verifier for a claimed prior-shift result."""
    try:
        expected = prior_shift(proba, prior_train, prior_deploy)
        claim = np.asarray(claimed, dtype=float)
        return bool(
            tol >= 0
            and np.isfinite(tol)
            and claim.shape == expected.shape
            and np.isfinite(claim).all()
            and np.allclose(expected, claim, atol=tol, rtol=0.0)
        )
    except (TypeError, ValueError):
        return False


def _unit_probability(value: float, name: str) -> float:
    probability = float(value)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return probability


def test_posterior(
    prior: float,
    sensitivity: float,
    specificity: float,
    positive: bool = True,
) -> float:
    """Return the binary posterior after one positive or negative test."""
    prior = _unit_probability(prior, "prior")
    sensitivity = _unit_probability(sensitivity, "sensitivity")
    specificity = _unit_probability(specificity, "specificity")
    if positive:
        numerator = prior * sensitivity
        denominator = numerator + (1.0 - prior) * (1.0 - specificity)
    else:
        numerator = prior * (1.0 - sensitivity)
        denominator = numerator + (1.0 - prior) * specificity
    return numerator / denominator if denominator > 0 else 0.0


def sequential_test(prior: float, tests: Iterable[BinaryTest]) -> list[float]:
    """Fold independent binary test evidence into a running posterior."""
    posterior = _unit_probability(prior, "prior")
    trajectory = []
    for index, test in enumerate(tests):
        try:
            sensitivity, specificity, positive = test
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"tests[{index}] must be a (sensitivity, specificity, positive) triple"
            ) from error
        posterior = test_posterior(posterior, sensitivity, specificity, bool(positive))
        trajectory.append(posterior)
    return trajectory


def check_sequential_test(
    prior: float,
    tests: Iterable[BinaryTest],
    claimed: Sequence[float],
    tol: float = 1e-9,
) -> bool:
    """Fail-closed verifier for a claimed sequential posterior trajectory."""
    try:
        expected = sequential_test(prior, tests)
        claim = [float(value) for value in claimed]
        return bool(
            tol >= 0
            and np.isfinite(tol)
            and len(expected) == len(claim)
            and all(np.isfinite(value) for value in claim)
            and all(
                abs(actual - expected_value) <= tol
                for actual, expected_value in zip(claim, expected, strict=True)
            )
        )
    except (TypeError, ValueError):
        return False


__all__ = [
    "as_prior",
    "check_prior_shift",
    "check_sequential_test",
    "prior_shift",
    "sequential_test",
    "test_posterior",
]
