"""Probe transparent nonlinear global reads on TabPVN's shared OOF stack.

The production affine read captures a global linear signal. This audit asks whether
two bounded extensions add transferable evidence without another booster fit:

* a strongly regularized logit over standardized values and their squares;
* a diagonal class-conditional Gaussian likelihood;
* a shrinkage-LDA likelihood with one explicit shared covariance geometry.

Both reads are cheap and explicit. This module is research-only until a candidate
passes the production OOF gate and an untouched official holdout.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark.affine_composition_probe import _evaluate
from benchmark.datasets import tabarena_suite
from tabpvn import TabPVN
from tabpvn.base import (
    _AFFINE_RANK_INVERSE_REGULARIZATION,
    _affine_rank_weight,
    _classification_rank_score,
    _preserve_certified_class,
)
from tabpvn.proposers.affine import AffineLogitRead

DEFAULT_CASES = (
    ("maternal_health_risk", 0),
    ("maternal_health_risk", 1),
    ("taiwanese_bankruptcy_prediction", 0),
    ("students_dropout_and_academic_success", 0),
    ("credit-g", 0),
    ("qsar-biodeg", 0),
)


class _StandardizedRead:
    """Shared finite standardization for research-only global reads."""

    def _fit_standardizer(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ <= 0.0)] = 1.0
        return (features - self.mean_) / self.scale_

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def _class_order(self, model_classes: np.ndarray, classes: np.ndarray) -> None:
        self.order_ = np.asarray(
            [int(np.flatnonzero(model_classes == label)[0]) for label in classes],
            dtype=np.int64,
        )


class _DiagonalQuadraticLogitRead(_StandardizedRead):
    """Strongly regularized discriminative read over z and z squared."""

    def _fit_map(self, standardized: np.ndarray) -> np.ndarray:
        self.quadratic_indices_ = np.asarray(
            [index for index in range(standardized.shape[1]) if len(np.unique(standardized[:, index])) > 2],
            dtype=np.int64,
        )
        if not len(self.quadratic_indices_):
            raise ValueError("quadratic read requires a non-binary feature")
        squared: np.ndarray = standardized[:, self.quadratic_indices_] ** 2
        self.square_mean_ = squared.mean(axis=0)
        self.square_scale_ = squared.std(axis=0)
        self.square_scale_[~np.isfinite(self.square_scale_) | (self.square_scale_ <= 0.0)] = 1.0
        return np.concatenate(
            (standardized, (squared - self.square_mean_) / self.square_scale_),
            axis=1,
        )

    def _map(self, standardized: np.ndarray) -> np.ndarray:
        squared: np.ndarray = standardized[:, self.quadratic_indices_] ** 2
        return np.concatenate(
            (standardized, (squared - self.square_mean_) / self.square_scale_),
            axis=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray):
        from sklearn.linear_model import LogisticRegression

        standardized = self._fit_standardizer(X)
        mapped = self._fit_map(standardized)
        self.estimator_ = LogisticRegression(
            C=_AFFINE_RANK_INVERSE_REGULARIZATION,
            solver="lbfgs",
            max_iter=500,
            random_state=0,
        ).fit(mapped, y)
        self._class_order(np.asarray(self.estimator_.classes_), classes)
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        standardized = self._standardize(X)
        mapped = self._map(standardized)
        return np.asarray(self.estimator_.predict_proba(mapped)[:, self.order_], dtype=float)


class _DiagonalGaussianRead(_StandardizedRead):
    """Diagonal class-conditional Gaussian likelihood with a fixed variance floor."""

    def fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray):
        from sklearn.naive_bayes import GaussianNB

        standardized = self._fit_standardizer(X)
        self.estimator_ = GaussianNB(var_smoothing=1e-3).fit(standardized, y)
        self._class_order(np.asarray(self.estimator_.classes_), classes)
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.estimator_.predict_proba(self._standardize(X))[:, self.order_],
            dtype=float,
        )


class _ShrinkageLDARead(_StandardizedRead):
    """Shared-covariance Gaussian discrimination with analytic shrinkage."""

    def fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        standardized = self._fit_standardizer(X)
        self.estimator_ = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        ).fit(standardized, y)
        self._class_order(np.asarray(self.estimator_.classes_), classes)
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.estimator_.predict_proba(self._standardize(X))[:, self.order_],
            dtype=float,
        )


class _GlobalShapeProbeTabPVN(TabPVN):
    """Capture the exact probability stack immediately before the affine gate."""

    def _global_affine_rank_gate(self, X, y, precomp):
        self.global_shape_probe_ = {
            "base": self._oof_probability_stack(precomp).copy(),
            "splits": tuple(precomp["splits"]),
            "evidence_rows": np.asarray(precomp["evidence_rows"], dtype=np.int64),
        }
        weight = super()._global_affine_rank_gate(X, y, precomp)
        self.global_shape_probe_["post_affine_base"] = self._oof_probability_stack(precomp).copy()
        return weight


def _fold_priors(target: np.ndarray, splits, n_classes: int) -> np.ndarray:
    counts = np.bincount(target, minlength=n_classes).astype(float)
    priors = np.tile(counts / counts.sum(), (len(target), 1))
    for train, valid in splits:
        fold_counts = np.bincount(target[train], minlength=n_classes).astype(float)
        priors[valid] = fold_counts / fold_counts.sum()
    return priors


def _oof_read(read_type, X, y, classes, splits, base):
    probability = base.copy()
    for train, valid in splits:
        member = read_type().fit(X[train], y[train], classes=classes)
        probability[valid] = member.proba(X[valid])
    return probability


def _compose(base, member, prior, weight, composition):
    return AffineLogitRead.combine(
        base,
        member,
        weight,
        composition=composition,
        prior=prior,
    )


def audit(dataset, fold_index: int) -> list[dict[str, object]]:
    train, test = dataset.splits[fold_index]
    model = _GlobalShapeProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], dataset.y[train])
    probe = model.global_shape_probe_
    classes = np.asarray(model.classes_)
    class_index = {value: index for index, value in enumerate(classes)}
    encoded_train = np.asarray([class_index[value] for value in dataset.y[train]], dtype=np.int32)
    encoded_test = np.asarray([class_index[value] for value in dataset.y[test]], dtype=np.int32)
    X_train = model._X(dataset.X.iloc[train])
    X_test = model._X(dataset.X.iloc[test])
    splits = probe["splits"]
    rows = probe["evidence_rows"]
    base = probe["base"]
    post_affine_base = probe["post_affine_base"]
    weight = _affine_rank_weight(len(train))
    oof_prior = _fold_priors(encoded_train, splits, len(classes))
    deployment_prior = np.bincount(encoded_train, minlength=len(classes)).astype(float)
    deployment_prior /= deployment_prior.sum()

    heldout_base = model._blended_proba(
        X_test,
        include_prior_rank=False,
        include_affine_rank=False,
    )
    heldout_post_affine = model._blended_proba(
        X_test,
        include_prior_rank=False,
    )
    heldout_base_rank = _classification_rank_score(encoded_test, heldout_base)
    heldout_base_accuracy = float(np.mean(heldout_base.argmax(1) == encoded_test))
    current = model.predict_proba(dataset.X.iloc[test])
    current_rank = _classification_rank_score(encoded_test, current)
    current_accuracy = float(np.mean(model.predict(dataset.X.iloc[test]) == dataset.y[test]))
    affine_report = model.affine_rank_report_[-1]
    decision_eligible = getattr(model, "_numeric_interval", None) is None

    read_probabilities = {
        "affine": _oof_read(
            AffineLogitRead,
            X_train,
            np.asarray(dataset.y[train]),
            classes,
            splits,
            base,
        ),
        "quadratic": _oof_read(
            _DiagonalQuadraticLogitRead,
            X_train,
            np.asarray(dataset.y[train]),
            classes,
            splits,
            base,
        ),
        "gaussian": _oof_read(
            _DiagonalGaussianRead,
            X_train,
            np.asarray(dataset.y[train]),
            classes,
            splits,
            base,
        ),
        "shrinkage_lda": _oof_read(
            _ShrinkageLDARead,
            X_train,
            np.asarray(dataset.y[train]),
            classes,
            splits,
            base,
        ),
    }
    read_probabilities["affine_quadratic"] = 0.5 * (
        read_probabilities["affine"] + read_probabilities["quadratic"]
    )
    specifications = (
        ("affine_arithmetic", "affine", "arithmetic", "pre_affine"),
        ("affine_prior_ratio", "affine", "prior_ratio", "pre_affine"),
        ("quadratic_arithmetic", "quadratic", "arithmetic", "pre_affine"),
        ("quadratic_prior_ratio", "quadratic", "prior_ratio", "pre_affine"),
        ("gaussian_prior_ratio", "gaussian", "prior_ratio", "pre_affine"),
        ("shrinkage_lda_arithmetic", "shrinkage_lda", "arithmetic", "pre_affine"),
        ("shrinkage_lda_prior_ratio", "shrinkage_lda", "prior_ratio", "pre_affine"),
        (
            "affine_quadratic_arithmetic",
            "affine_quadratic",
            "arithmetic",
            "pre_affine",
        ),
        (
            "affine_quadratic_prior_ratio",
            "affine_quadratic",
            "prior_ratio",
            "pre_affine",
        ),
        ("sequential_quadratic_arithmetic", "quadratic", "arithmetic", "post_affine"),
        ("sequential_quadratic_prior_ratio", "quadratic", "prior_ratio", "post_affine"),
        (
            "sequential_shrinkage_lda_arithmetic",
            "shrinkage_lda",
            "arithmetic",
            "post_affine",
        ),
        (
            "sequential_shrinkage_lda_prior_ratio",
            "shrinkage_lda",
            "prior_ratio",
            "post_affine",
        ),
    )
    full_reads = {
        "affine": AffineLogitRead(
            inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
            seed=0,
        ).fit(X_train, np.asarray(dataset.y[train]), classes=classes),
        "quadratic": _DiagonalQuadraticLogitRead().fit(X_train, np.asarray(dataset.y[train]), classes),
        "gaussian": _DiagonalGaussianRead().fit(X_train, np.asarray(dataset.y[train]), classes),
        "shrinkage_lda": _ShrinkageLDARead().fit(X_train, np.asarray(dataset.y[train]), classes),
    }
    heldout_members = {name: read.proba(X_test) for name, read in full_reads.items()}
    heldout_members["affine_quadratic"] = 0.5 * (heldout_members["affine"] + heldout_members["quadratic"])

    records = []
    for name, read_name, composition, stack in specifications:
        candidate_base = base if stack == "pre_affine" else post_affine_base
        heldout_candidate_base = heldout_base if stack == "pre_affine" else heldout_post_affine
        candidate_base_rank = _classification_rank_score(encoded_test, heldout_candidate_base)
        candidate_base_accuracy = float(np.mean(heldout_candidate_base.argmax(1) == encoded_test))
        candidate = _compose(
            candidate_base,
            read_probabilities[read_name],
            oof_prior,
            weight,
            composition,
        )
        evaluation = _evaluate(encoded_train, candidate_base, candidate, splits, rows)
        candidate_decision_eligible = decision_eligible and not (
            stack == "post_affine"
            and getattr(model, "_affine_rank_permission", None) in {"decision_only", "decision_and_rank"}
        )
        if not candidate_decision_eligible:
            evaluation["decision_selected"] = False
        projected = _preserve_certified_class(candidate_base, candidate)
        projected_evaluation = _evaluate(encoded_train, candidate_base, projected, splits, rows)

        heldout_member = heldout_members[read_name]
        heldout = _compose(
            heldout_candidate_base,
            heldout_member,
            deployment_prior,
            weight,
            composition,
        )
        heldout_projected = _preserve_certified_class(heldout_candidate_base, heldout)
        records.append(
            {
                "dataset": dataset.name,
                "fold": fold_index,
                "candidate": name,
                "stack": stack,
                "rows": len(train),
                "features": X_train.shape[1],
                "classes": len(classes),
                "weight": weight,
                "decision_eligible": candidate_decision_eligible,
                **evaluation,
                "projected_oof_rank_auc_delta": projected_evaluation["oof_rank_auc_delta"],
                "projected_fold_rank_auc_delta": projected_evaluation["fold_rank_auc_delta"],
                "projected_rank_selected": projected_evaluation["rank_selected"],
                "current_affine_permission": affine_report.get("permission"),
                "current_affine_composition": affine_report.get("composition"),
                "current_oof_rank_auc": affine_report.get("mean_score"),
                "current_heldout_rank_auc_delta": current_rank - heldout_base_rank,
                "current_heldout_accuracy_delta": current_accuracy - heldout_base_accuracy,
                "heldout_rank_auc_delta": (
                    _classification_rank_score(encoded_test, heldout) - candidate_base_rank
                ),
                "heldout_accuracy_delta": (
                    float(np.mean(heldout.argmax(1) == encoded_test)) - candidate_base_accuracy
                ),
                "projected_heldout_rank_auc_delta": (
                    _classification_rank_score(encoded_test, heldout_projected) - candidate_base_rank
                ),
            }
        )
    return records


def _parse_cases(value: str | None) -> tuple[tuple[str, int], ...]:
    if not value:
        return DEFAULT_CASES
    cases = []
    for item in value.split(","):
        name, separator, fold = item.rpartition(":")
        if not separator or not name:
            raise ValueError("cases must use dataset:fold entries")
        cases.append((name, int(fold)))
    return tuple(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", help="Comma-separated dataset:fold entries")
    parser.add_argument("--out", default="results/tabpvn_global_shape_probe.csv")
    args = parser.parse_args()
    cases = _parse_cases(args.cases)
    datasets = {
        dataset.name: dataset
        for dataset in tabarena_suite(
            size="all",
            dataset_names=[name for name, _fold in cases],
        )
    }
    records = [record for name, fold in cases for record in audit(datasets[name], fold)]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(
            f"{record['dataset']} fold {record['fold']} {record['candidate']}: "
            f"decision={record['decision_selected']} rank={record['rank_selected']} "
            f"projected_rank={record['projected_rank_selected']} "
            f"OOF rank={record['oof_rank_auc_delta']:+.6f} "
            f"heldout rank={record['heldout_rank_auc_delta']:+.6f} "
            f"heldout accuracy={record['heldout_accuracy_delta']:+.6f}"
        )


if __name__ == "__main__":
    main()
