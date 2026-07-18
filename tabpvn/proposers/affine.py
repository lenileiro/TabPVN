"""Explicit global affine evidence for classification decisions and ranking.

The certified booster is intentionally local and region based.  This proposer
adds the complementary low-variance view: one strongly regularized affine
logit over the compiled numeric facts.  The fitted sklearn object is collapsed
to coefficients and intercepts, so serving is only explicit NumPy arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class AffineLogitRead:
    """Strongly regularized affine class-probability read.

    ``TabPVN`` normally admits this component as a class-preserving rank read.
    It may own the public class decision only after a separate paired OOF
    accuracy gate.  In that case :meth:`evidence` exposes the complete affine
    arithmetic needed to re-check the decision without fitted model state.
    """

    def __init__(
        self,
        *,
        inverse_regularization: float = 0.03,
        max_iter: int = 500,
        seed: int = 0,
    ) -> None:
        if inverse_regularization <= 0.0:
            raise ValueError("inverse_regularization must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.inverse_regularization = float(inverse_regularization)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        classes: ArrayLike | None = None,
    ) -> AffineLogitRead:
        """Fit standardized logistic evidence and collapse it to raw-space coefficients."""
        from sklearn.linear_model import LogisticRegression

        features = np.asarray(X, dtype=float)
        target = np.asarray(y)
        if features.ndim != 2 or not len(features) or features.shape[1] < 1:
            raise ValueError("affine logit evidence requires a non-empty two-dimensional table")
        if len(target) != len(features):
            raise ValueError("X and y must contain the same number of rows")
        if not np.isfinite(features).all():
            raise ValueError("affine logit evidence requires finite numeric features")

        model_classes = np.unique(target)
        if len(model_classes) < 2:
            raise ValueError("affine logit evidence requires at least two classes")
        requested_classes = model_classes if classes is None else np.asarray(classes)
        if set(requested_classes.tolist()) != set(model_classes.tolist()):
            raise ValueError("requested classes do not match the fitted target classes")

        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
        standardized = (features - mean) / scale
        estimator = LogisticRegression(
            C=self.inverse_regularization,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.seed,
        ).fit(standardized, target)

        standardized_coef = np.asarray(estimator.coef_, dtype=float)
        self.coef_ = np.ascontiguousarray(standardized_coef / scale[None, :])
        self.intercept_ = np.ascontiguousarray(
            np.asarray(estimator.intercept_, dtype=float) - standardized_coef @ (mean / scale)
        )
        self.model_classes_ = np.asarray(estimator.classes_)
        self.classes_ = requested_classes.copy()
        self.n_features_in_ = int(features.shape[1])
        self._class_order = np.asarray(
            [int(np.flatnonzero(self.model_classes_ == label)[0]) for label in self.classes_],
            dtype=np.int64,
        )
        return self

    def proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return probabilities from the collapsed affine coefficients."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("AffineLogitRead must be fitted before proba")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(f"expected a two-dimensional table with {self.n_features_in_} features")
        margins = features @ self.coef_.T + self.intercept_
        if len(self.model_classes_) == 2:
            margin = margins[:, 0]
            positive: NDArray[np.float64] = np.empty(len(margin), dtype=float)
            nonnegative = margin >= 0.0
            positive[nonnegative] = 1.0 / (1.0 + np.exp(-margin[nonnegative]))
            exp_margin = np.exp(margin[~nonnegative])
            positive[~nonnegative] = exp_margin / (1.0 + exp_margin)
            probability = np.column_stack((1.0 - positive, positive))
        else:
            shifted = margins - margins.max(axis=1, keepdims=True)
            probability = np.exp(shifted)
            probability /= probability.sum(axis=1, keepdims=True)
        return np.ascontiguousarray(probability[:, self._class_order], dtype=float)

    @staticmethod
    def combine(
        base_probability: ArrayLike,
        affine_probability: ArrayLike,
        weight: float,
        *,
        composition: str = "arithmetic",
        prior: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Compose base and affine probabilities with explicit bounded arithmetic."""
        base = np.asarray(base_probability, dtype=float)
        affine = np.asarray(affine_probability, dtype=float)
        weight = float(weight)
        if (
            base.ndim != 2
            or base.shape != affine.shape
            or base.shape[1] < 2
            or not np.isfinite(base).all()
            or not np.isfinite(affine).all()
            or (base < 0.0).any()
            or (affine < 0.0).any()
            or not np.allclose(base.sum(axis=1), 1.0, atol=1e-9)
            or not np.allclose(affine.sum(axis=1), 1.0, atol=1e-9)
            or not 0.0 < weight <= 1.0
        ):
            raise ValueError("finite probability rows and weight in (0, 1] are required")
        if composition == "arithmetic":
            combined = (1.0 - weight) * base + weight * affine
        elif composition == "prior_ratio":
            declared_prior = np.asarray(prior, dtype=float)
            if declared_prior.shape == (base.shape[1],):
                prior_rows = np.broadcast_to(declared_prior, base.shape)
            elif declared_prior.shape == base.shape:
                prior_rows = declared_prior
            else:
                raise ValueError("prior_ratio composition requires one aligned prior or one prior per row")
            if (
                not np.isfinite(prior_rows).all()
                or (prior_rows <= 0.0).any()
                or not np.allclose(prior_rows.sum(axis=1), 1.0, atol=1e-9)
            ):
                raise ValueError("prior_ratio composition requires positive normalized prior rows")
            tiny = np.finfo(float).tiny
            log_combined = np.log(np.clip(base, tiny, 1.0)) + weight * (
                np.log(np.clip(affine, tiny, 1.0)) - np.log(prior_rows)
            )
            log_combined -= log_combined.max(axis=1, keepdims=True)
            combined = np.exp(log_combined)
            combined /= combined.sum(axis=1, keepdims=True)
        else:
            raise ValueError("composition must be 'arithmetic' or 'prior_ratio'")
        return np.ascontiguousarray(combined, dtype=float)

    def evidence(
        self,
        X: ArrayLike,
        row: int,
        base_probability: ArrayLike,
        weight: float,
        *,
        base_proof: Any,
        composition: str = "arithmetic",
        prior: ArrayLike | None = None,
        verify_base: Callable[[Any], bool] | None = None,
    ) -> dict[str, Any]:
        """Return independently checkable arithmetic for one affine decision."""
        features = np.asarray(X, dtype=float)
        base = np.asarray(base_probability, dtype=float)
        row = int(row)
        if features.ndim != 2 or not 0 <= row < len(features):
            raise ValueError("row must identify one row in a two-dimensional table")
        if base.shape != (len(features), len(self.classes_)):
            raise ValueError("base_probability must contain one distribution per input row")
        if not np.isfinite(base).all() or (base < 0.0).any():
            raise ValueError("base_probability must be finite and non-negative")
        if not np.allclose(base.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("base_probability rows must sum to one")
        if not np.isfinite(weight) or not 0.0 < float(weight) <= 1.0:
            raise ValueError("weight must be in (0, 1]")

        member = self.proba(features[row : row + 1])[0]
        combined = self.combine(
            base[row : row + 1],
            member[None, :],
            weight,
            composition=composition,
            prior=prior,
        )[0]
        base_index = int(base[row].argmax())
        prediction_index = int(combined.argmax())
        record = {
            "kind": "affine_logit_decision",
            "classes": [_python_scalar(value) for value in self.classes_],
            "model_classes": [_python_scalar(value) for value in self.model_classes_],
            "class_order": self._class_order.tolist(),
            "input": features[row].tolist(),
            "coefficients": self.coef_.tolist(),
            "intercepts": self.intercept_.tolist(),
            "inverse_regularization": self.inverse_regularization,
            "weight": float(weight),
            "composition": composition,
            "prior": None if prior is None else np.asarray(prior, dtype=float).tolist(),
            "base_probability": base[row].tolist(),
            "affine_probability": member.tolist(),
            "combined_probability": combined.tolist(),
            "base_prediction": _python_scalar(self.classes_[base_index]),
            "prediction": _python_scalar(self.classes_[prediction_index]),
            "override": bool(base_index != prediction_index),
            "base_proof": base_proof,
        }
        record["verified"] = self.verify_evidence(record, verify_base=verify_base)
        return record

    @staticmethod
    def verify_evidence(
        record: Mapping[str, Any],
        *,
        verify_base: Callable[[Any], bool] | None = None,
        tol: float = 1e-9,
    ) -> bool:
        """Recompute an affine decision certificate without model state."""
        try:
            if record.get("kind") != "affine_logit_decision":
                return False
            classes = list(record["classes"])
            model_classes = list(record["model_classes"])
            class_order = np.asarray(record["class_order"], dtype=np.int64)
            features = np.asarray(record["input"], dtype=float)
            coefficients = np.asarray(record["coefficients"], dtype=float)
            intercepts = np.asarray(record["intercepts"], dtype=float)
            base = np.asarray(record["base_probability"], dtype=float)
            declared_member = np.asarray(record["affine_probability"], dtype=float)
            declared_combined = np.asarray(record["combined_probability"], dtype=float)
            weight = float(record["weight"])
            composition = str(record.get("composition", "arithmetic"))
            n_classes = len(classes)
            if (
                n_classes < 2
                or len(model_classes) != n_classes
                or class_order.shape != (n_classes,)
                or sorted(class_order.tolist()) != list(range(n_classes))
                or features.ndim != 1
                or base.shape != (n_classes,)
                or declared_member.shape != (n_classes,)
                or declared_combined.shape != (n_classes,)
                or intercepts.ndim != 1
                or not 0.0 < weight <= 1.0
            ):
                return False
            if any(
                not _labels_equal(classes[index], model_classes[int(class_order[index])])
                for index in range(n_classes)
            ):
                return False
            expected_rows = 1 if n_classes == 2 else n_classes
            if coefficients.shape != (expected_rows, len(features)) or intercepts.shape != (expected_rows,):
                return False
            arrays = (features, coefficients, intercepts, base, declared_member, declared_combined)
            if not all(np.isfinite(array).all() for array in arrays):
                return False
            if (base < 0.0).any() or not np.isclose(base.sum(), 1.0, atol=tol):
                return False

            margins = coefficients @ features + intercepts
            if n_classes == 2:
                margin = float(margins[0])
                positive = (
                    1.0 / (1.0 + np.exp(-margin))
                    if margin >= 0.0
                    else np.exp(margin) / (1.0 + np.exp(margin))
                )
                model_probability = np.asarray([1.0 - positive, positive])
            else:
                shifted = margins - margins.max()
                model_probability = np.exp(shifted)
                model_probability /= model_probability.sum()
            member = model_probability[class_order]
            combined = AffineLogitRead.combine(
                base[None, :],
                member[None, :],
                weight,
                composition=composition,
                prior=record.get("prior"),
            )[0]
            base_prediction = classes[int(base.argmax())]
            prediction = classes[int(combined.argmax())]
            base_proof = record["base_proof"]
            nested_prediction = _proof_prediction(base_proof)
            if nested_prediction is None or not _labels_equal(nested_prediction, base_prediction):
                return False
            if verify_base is not None and not verify_base(base_proof):
                return False
            return bool(
                np.allclose(member, declared_member, atol=tol, rtol=tol)
                and np.allclose(combined, declared_combined, atol=tol, rtol=tol)
                and _labels_equal(base_prediction, record["base_prediction"])
                and _labels_equal(prediction, record["prediction"])
                and bool(not _labels_equal(base_prediction, prediction)) == bool(record["override"])
            )
        except (KeyError, TypeError, ValueError, FloatingPointError, OverflowError):
            return False

    def report(self) -> dict[str, Any]:
        """Return compact, serializable evidence metadata."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("AffineLogitRead must be fitted before report")
        return {
            "kind": "global_affine_logit",
            "features": self.n_features_in_,
            "classes": int(len(self.classes_)),
            "coefficient_count": int(np.count_nonzero(self.coef_)),
            "coefficient_l2": float(np.linalg.norm(self.coef_)),
            "inverse_regularization": self.inverse_regularization,
            "serving": "explicit_affine_softmax",
        }


def _python_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _proof_prediction(proof: Any) -> Any:
    if not isinstance(proof, Mapping):
        return None
    return proof.get("class", proof.get("prediction"))


__all__ = ["AffineLogitRead"]
