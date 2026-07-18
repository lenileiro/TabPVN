"""Probe a fold-local ordinal affine read on compact three-class tables.

The candidate derives an unordered class path from standardized class-centroid
geometry: the farthest pair are endpoints and the remaining class is the
middle. Two strongly regularized cumulative affine logits then estimate the
probability of lying above each boundary. Serving is explicit matrix arithmetic;
this module remains research-only until OOF and official holdout evidence agree.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark.affine_composition_probe import _evaluate
from benchmark.datasets import tabarena_suite
from benchmark.global_shape_probe import _fold_priors, _GlobalShapeProbeTabPVN
from tabpvn.base import (
    _AFFINE_RANK_INVERSE_REGULARIZATION,
    _affine_rank_weight,
    _classification_rank_score,
    _preserve_certified_class,
)
from tabpvn.proposers.affine import AffineLogitRead

DEFAULT_CASES = tuple(("maternal_health_risk", fold) for fold in range(3))


class _CentroidOrdinalAffineRead:
    """Explicit cumulative logits with a fold-local geometric class order."""

    def fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray):
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        self.classes_ = np.asarray(classes)
        if features.ndim != 2 or len(features) != len(labels):
            raise ValueError("ordinal affine read requires aligned two-dimensional features")
        if len(self.classes_) != 3:
            raise ValueError("ordinal affine read is bounded to exactly three classes")

        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
        standardized = (features - mean) / scale
        centroids = np.vstack([standardized[labels == label].mean(axis=0) for label in self.classes_])
        if not np.isfinite(centroids).all():
            raise ValueError("every ordinal class must have finite centroid evidence")

        pairs = ((0, 1), (0, 2), (1, 2))
        distances = np.asarray([np.linalg.norm(centroids[left] - centroids[right]) for left, right in pairs])
        endpoint_pair = pairs[int(np.argmax(distances))]
        middle = next(index for index in range(3) if index not in endpoint_pair)
        left, right = sorted(endpoint_pair)
        self.order_indices_ = np.asarray([left, middle, right], dtype=np.int64)
        self.ordered_classes_ = self.classes_[self.order_indices_]
        self.centroid_distances_ = distances

        ranks: np.ndarray = np.empty(len(labels), dtype=np.int8)
        for rank, label in enumerate(self.ordered_classes_):
            ranks[labels == label] = rank
        self.boundaries_ = [
            AffineLogitRead(
                inverse_regularization=_AFFINE_RANK_INVERSE_REGULARIZATION,
                seed=0,
            ).fit(features, (ranks > boundary).astype(np.int8), classes=np.array([0, 1]))
            for boundary in (0, 1)
        ]
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        cumulative = np.column_stack([boundary.proba(features)[:, 1] for boundary in self.boundaries_])
        crossing = cumulative[:, 0] < cumulative[:, 1]
        if np.any(crossing):
            pooled = cumulative[crossing].mean(axis=1)
            cumulative[crossing, 0] = pooled
            cumulative[crossing, 1] = pooled

        ordered = np.column_stack(
            (
                1.0 - cumulative[:, 0],
                cumulative[:, 0] - cumulative[:, 1],
                cumulative[:, 1],
            )
        )
        probability = np.empty_like(ordered)
        probability[:, self.order_indices_] = ordered
        probability /= probability.sum(axis=1, keepdims=True)
        return probability

    def report(self) -> dict[str, object]:
        return {
            "order": self.ordered_classes_.tolist(),
            "centroid_distances": self.centroid_distances_.tolist(),
            "serving": "two_explicit_cumulative_affine_logits",
        }


def _canonical_order(order: list[object]) -> tuple[object, ...]:
    forward = tuple(order)
    reverse = tuple(reversed(order))
    return min(forward, reverse, key=lambda values: tuple(map(str, values)))


def _ordinal_transform(
    probability: np.ndarray,
    order_indices: np.ndarray,
    kind: str,
    strength: float,
) -> np.ndarray:
    ordered = np.asarray(probability, dtype=float)[:, order_indices].copy()
    epsilon = 1e-12
    if kind == "cumulative_temperature":
        tails = np.column_stack((ordered[:, 1] + ordered[:, 2], ordered[:, 2]))
        logits = np.log(np.clip(tails, epsilon, 1.0 - epsilon)) - np.log(
            np.clip(1.0 - tails, epsilon, 1.0 - epsilon)
        )
        adjusted = 1.0 / (1.0 + np.exp(-logits / strength))
        ordered = np.column_stack((1.0 - adjusted[:, 0], adjusted[:, 0] - adjusted[:, 1], adjusted[:, 1]))
    elif kind == "middle_contrast":
        bridge = np.sqrt(np.clip(ordered[:, 0] * ordered[:, 2], epsilon, None))
        ratio = np.clip(ordered[:, 1], epsilon, None) / bridge
        ordered[:, 1] *= np.power(ratio, strength)
    elif kind == "middle_bridge":
        bridge = np.sqrt(np.clip(ordered[:, 0] * ordered[:, 2], 0.0, None))
        ordered[:, 1] += strength * bridge
    elif kind == "moment_gaussian":
        positions: np.ndarray = np.arange(3, dtype=float)
        center = ordered @ positions
        variance = np.sum(ordered * (positions[None, :] - center[:, None]) ** 2, axis=1)
        scale = np.maximum(variance + strength, epsilon)
        logits = -((positions[None, :] - center[:, None]) ** 2) / (2.0 * scale[:, None])
        logits -= logits.max(axis=1, keepdims=True)
        ordered = np.exp(logits)
    else:
        raise ValueError(f"unknown ordinal transform: {kind}")
    ordered = np.clip(ordered, 0.0, None)
    ordered /= ordered.sum(axis=1, keepdims=True)
    transformed = np.empty_like(ordered)
    transformed[:, order_indices] = ordered
    return transformed


def audit(dataset, fold_index: int) -> list[dict[str, object]]:
    train, test = dataset.splits[fold_index]
    model = _GlobalShapeProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], dataset.y[train])
    probe = model.global_shape_probe_
    classes = np.asarray(model.classes_)
    if len(classes) != 3:
        raise ValueError("ordinal shape probe requires exactly three classes")

    class_index = {value: index for index, value in enumerate(classes)}
    train_target = np.asarray(dataset.y[train])
    test_target = np.asarray(dataset.y[test])
    encoded_train = np.asarray([class_index[value] for value in train_target], dtype=np.int32)
    encoded_test = np.asarray([class_index[value] for value in test_target], dtype=np.int32)
    X_train = model._X(dataset.X.iloc[train])
    X_test = model._X(dataset.X.iloc[test])
    splits = tuple(probe["splits"])
    evidence_rows = np.asarray(probe["evidence_rows"], dtype=np.int64)
    base = np.asarray(probe["post_affine_base"], dtype=float)
    member_probability = base.copy()
    fold_reports = []
    covered: np.ndarray = np.zeros(len(train), dtype=bool)
    for fold_train, fold_valid in splits:
        member = _CentroidOrdinalAffineRead().fit(
            X_train[fold_train],
            train_target[fold_train],
            classes,
        )
        member_probability[fold_valid] = member.proba(X_train[fold_valid])
        covered[fold_valid] = True
        fold_reports.append(member.report())
    if not np.all(covered[evidence_rows]):
        raise ValueError("ordinal evidence rows require fold-local predictions")

    full_member = _CentroidOrdinalAffineRead().fit(X_train, train_target, classes)
    heldout_member = full_member.proba(X_test)
    canonical_orders = [_canonical_order(report["order"]) for report in fold_reports]
    full_order = _canonical_order(full_member.report()["order"])
    order_consistent = len(set(canonical_orders + [full_order])) == 1
    weight = _affine_rank_weight(len(train))
    fold_prior = _fold_priors(encoded_train, splits, len(classes))
    deployment_prior = np.bincount(encoded_train, minlength=len(classes)).astype(float)
    deployment_prior /= deployment_prior.sum()
    heldout_base = model._blended_proba(X_test, include_prior_rank=False)
    heldout_base_rank = _classification_rank_score(encoded_test, heldout_base)
    heldout_base_accuracy = float(np.mean(heldout_base.argmax(axis=1) == encoded_test))

    records = []
    for composition in ("arithmetic", "prior_ratio"):
        candidate = AffineLogitRead.combine(
            base,
            member_probability,
            weight,
            composition=composition,
            prior=fold_prior,
        )
        evaluation = _evaluate(encoded_train, base, candidate, splits, evidence_rows)
        projected = _preserve_certified_class(base, candidate)
        projected_evaluation = _evaluate(
            encoded_train,
            base,
            projected,
            splits,
            evidence_rows,
        )
        heldout = AffineLogitRead.combine(
            heldout_base,
            heldout_member,
            weight,
            composition=composition,
            prior=deployment_prior,
        )
        heldout_projected = _preserve_certified_class(heldout_base, heldout)
        records.append(
            {
                "dataset": dataset.name,
                "fold": fold_index,
                "candidate": f"ordinal_{composition}",
                "rows": len(train),
                "features": X_train.shape[1],
                "weight": weight,
                "order_consistent": order_consistent,
                "fold_orders": [report["order"] for report in fold_reports],
                "full_order": full_member.report()["order"],
                **evaluation,
                "projected_oof_rank_auc_delta": projected_evaluation["oof_rank_auc_delta"],
                "projected_fold_rank_auc_delta": projected_evaluation["fold_rank_auc_delta"],
                "projected_rank_selected": projected_evaluation["rank_selected"],
                "heldout_rank_auc_delta": (
                    _classification_rank_score(encoded_test, heldout) - heldout_base_rank
                ),
                "projected_heldout_rank_auc_delta": (
                    _classification_rank_score(encoded_test, heldout_projected) - heldout_base_rank
                ),
                "heldout_accuracy_delta": (
                    float(np.mean(heldout.argmax(axis=1) == encoded_test)) - heldout_base_accuracy
                ),
            }
        )

    transform_specs = (
        ("cumulative_temperature", 0.75),
        ("cumulative_temperature", 1.25),
        ("middle_contrast", 0.25),
        ("middle_contrast", 0.5),
        ("middle_bridge", 0.25),
        ("moment_gaussian", 0.25),
    )
    for kind, strength in transform_specs:
        transformed_oof = base.copy()
        for (_fold_train, fold_valid), report in zip(splits, fold_reports, strict=True):
            order_indices = np.asarray(
                [class_index[label] for label in report["order"]],
                dtype=np.int64,
            )
            transformed_oof[fold_valid] = _ordinal_transform(
                base[fold_valid],
                order_indices,
                kind,
                strength,
            )
        full_order_indices = np.asarray(
            [class_index[label] for label in full_member.report()["order"]],
            dtype=np.int64,
        )
        transformed_heldout = _ordinal_transform(
            heldout_base,
            full_order_indices,
            kind,
            strength,
        )
        for blend_weight in (0.5, 1.0):
            candidate = (1.0 - blend_weight) * base + blend_weight * transformed_oof
            evaluation = _evaluate(encoded_train, base, candidate, splits, evidence_rows)
            projected = _preserve_certified_class(base, candidate)
            projected_evaluation = _evaluate(
                encoded_train,
                base,
                projected,
                splits,
                evidence_rows,
            )
            heldout = (1.0 - blend_weight) * heldout_base + blend_weight * transformed_heldout
            heldout_projected = _preserve_certified_class(heldout_base, heldout)
            records.append(
                {
                    "dataset": dataset.name,
                    "fold": fold_index,
                    "candidate": f"{kind}_{strength:g}_blend_{blend_weight:g}",
                    "rows": len(train),
                    "features": X_train.shape[1],
                    "weight": blend_weight,
                    "order_consistent": order_consistent,
                    "fold_orders": [report["order"] for report in fold_reports],
                    "full_order": full_member.report()["order"],
                    **evaluation,
                    "projected_oof_rank_auc_delta": projected_evaluation["oof_rank_auc_delta"],
                    "projected_fold_rank_auc_delta": projected_evaluation["fold_rank_auc_delta"],
                    "projected_rank_selected": projected_evaluation["rank_selected"],
                    "heldout_rank_auc_delta": (
                        _classification_rank_score(encoded_test, heldout) - heldout_base_rank
                    ),
                    "projected_heldout_rank_auc_delta": (
                        _classification_rank_score(encoded_test, heldout_projected) - heldout_base_rank
                    ),
                    "heldout_accuracy_delta": (
                        float(np.mean(heldout.argmax(axis=1) == encoded_test)) - heldout_base_accuracy
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
    parser.add_argument("--out", default="results/tabpvn_ordinal_shape_probe.csv")
    args = parser.parse_args()
    cases = _parse_cases(args.cases)
    datasets = {
        dataset.name: dataset
        for dataset in tabarena_suite(
            size="le10k",
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
            f"order={record['full_order']} consistent={record['order_consistent']} "
            f"projected_OOF={record['projected_oof_rank_auc_delta']:+.6f} "
            f"projected_heldout={record['projected_heldout_rank_auc_delta']:+.6f} "
            f"accuracy={record['heldout_accuracy_delta']:+.6f}"
        )


if __name__ == "__main__":
    main()
