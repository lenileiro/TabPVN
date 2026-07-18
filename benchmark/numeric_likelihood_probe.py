"""Probe a finite class-conditional interval likelihood read.

The production numeric interval challenger selects one strongest conjunction for
decision accuracy.  This research probe asks a different question: can every
supported one-dimensional interval contribute one likelihood ratio to ranking?
Each factor is a finite Dirichlet count table, each source feature is used once,
and the incumbent predicted class is preserved.  No additional booster is fit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from benchmark.affine_composition_probe import _evaluate
from benchmark.datasets import tabarena_suite
from benchmark.global_shape_probe import _GlobalShapeProbeTabPVN
from tabpvn.base import _classification_rank_score, _preserve_certified_class
from tabpvn.proposers.posterior import NumericIntervalPosteriorChallenger

DEFAULT_CASES = tuple(("maternal_health_risk", fold) for fold in range(3))
WEIGHTS = (0.1, 0.25, 0.5)


def _all_single_posterior(challenger, X):
    """Multiply one supported interval posterior/prior ratio per feature."""
    codes = challenger._interval_codes(X)
    delegate = challenger._delegate
    prior = np.asarray(delegate.prior, dtype=float)
    log_prior = np.log(np.clip(prior, 1e-300, 1.0))
    log_probability = np.tile(log_prior, (len(codes), 1))
    factor_counts: NDArray[np.int16] = np.zeros(len(codes), dtype=np.int16)

    for group, table in sorted(delegate.single_tables.items()):
        for row, level in enumerate(codes[:, group]):
            counts = table.counts.get((int(level),))
            if counts is None:
                continue
            support = int(counts.sum())
            if support < delegate.minimum_support:
                continue
            posterior = (counts + delegate.prior_strength * prior) / (support + delegate.prior_strength)
            log_probability[row] += np.log(np.clip(posterior, 1e-300, 1.0)) - log_prior
            factor_counts[row] += 1

    log_probability -= log_probability.max(axis=1, keepdims=True)
    probability = np.exp(log_probability)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, factor_counts


def _combine(base, posterior, prior, weight):
    return NumericIntervalPosteriorChallenger.combine_from_posterior(
        base,
        posterior,
        prior,
        weight,
    )


def _exact_profile_posterior(X, y, classes, query):
    """Finite Dirichlet posterior for recurring complete numeric profiles."""
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    query = np.asarray(query, dtype=float)
    classes = np.asarray(classes)
    class_index = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([class_index[label] for label in labels], dtype=np.int32)
    prior_counts = np.bincount(encoded, minlength=len(classes)).astype(float)
    prior = prior_counts / prior_counts.sum()
    unique, inverse = np.unique(features, axis=0, return_inverse=True)
    flat = inverse * len(classes) + encoded
    counts = np.bincount(
        flat,
        minlength=len(unique) * len(classes),
    ).reshape(len(unique), len(classes))
    lookup = {row.tobytes(): counts[index] for index, row in enumerate(unique)}
    probability = np.tile(prior, (len(query), 1))
    support: NDArray[np.int32] = np.zeros(len(query), dtype=np.int32)
    prior_strength = float(len(classes))
    minimum_support = len(classes)
    for row, values in enumerate(query):
        profile_counts = lookup.get(values.tobytes())
        if profile_counts is None or int(profile_counts.sum()) < minimum_support:
            continue
        support[row] = int(profile_counts.sum())
        probability[row] = (profile_counts + prior_strength * prior) / (support[row] + prior_strength)
    return probability, support, prior


def audit(dataset, fold_index):
    train, test = dataset.splits[fold_index]
    model = _GlobalShapeProbeTabPVN(seed=0, task="classification")
    model.fit(dataset.X.iloc[train], dataset.y[train])

    probe = model.global_shape_probe_
    X_train = model._X(dataset.X.iloc[train])
    X_test = model._X(dataset.X.iloc[test])
    labels = np.asarray(dataset.y[train])
    classes = np.asarray(model.classes_)
    class_index = {label: index for index, label in enumerate(classes)}
    encoded_train = np.asarray([class_index[label] for label in labels], dtype=np.int32)
    encoded_test = np.asarray(
        [class_index[label] for label in np.asarray(dataset.y[test])],
        dtype=np.int32,
    )
    splits = tuple(probe["splits"])
    evidence_rows = np.asarray(probe["evidence_rows"], dtype=np.int64)
    base = np.asarray(probe["base"], dtype=float)
    columns, names = model._numeric_interval_columns(X_train)

    posterior = np.tile(np.bincount(encoded_train, minlength=len(classes)), (len(labels), 1))
    posterior = posterior / posterior.sum(axis=1, keepdims=True)
    factor_counts: NDArray[np.int16] = np.zeros(len(labels), dtype=np.int16)
    exact_posterior = posterior.copy()
    exact_support: NDArray[np.int32] = np.zeros(len(labels), dtype=np.int32)
    covered: NDArray[np.bool_] = np.zeros(len(labels), dtype=bool)
    fold_priors: NDArray[np.float64] = np.zeros((len(labels), len(classes)), dtype=float)
    for fold_train, fold_valid in splits:
        challenger = NumericIntervalPosteriorChallenger(
            X_train[fold_train],
            labels[fold_train],
            classes,
            columns,
            names=names,
            smoothing="hierarchical",
        )
        fold_posterior, fold_factors = _all_single_posterior(
            challenger,
            X_train[fold_valid],
        )
        posterior[fold_valid] = fold_posterior
        factor_counts[fold_valid] = fold_factors
        fold_exact, fold_support, _fold_exact_prior = _exact_profile_posterior(
            X_train[fold_train],
            labels[fold_train],
            classes,
            X_train[fold_valid],
        )
        exact_posterior[fold_valid] = fold_exact
        exact_support[fold_valid] = fold_support
        fold_priors[fold_valid] = challenger.prior
        covered[fold_valid] = True
    if not np.all(covered[evidence_rows]):
        raise ValueError("likelihood probe requires fold-local evidence for every scored row")

    full = NumericIntervalPosteriorChallenger(
        X_train,
        labels,
        classes,
        columns,
        names=names,
        smoothing="hierarchical",
    )
    heldout_posterior, heldout_factors = _all_single_posterior(full, X_test)
    heldout_exact, heldout_exact_support, exact_prior = _exact_profile_posterior(
        X_train,
        labels,
        classes,
        X_test,
    )
    heldout_base = model._blended_proba(
        X_test,
        include_prior_rank=False,
        include_affine_rank=False,
    )
    heldout_base_score = _classification_rank_score(encoded_test, heldout_base)

    records = []
    candidates = (
        (
            "all_single_interval_likelihood",
            posterior,
            heldout_posterior,
            full.prior,
            factor_counts,
            heldout_factors,
        ),
        (
            "exact_profile_likelihood",
            exact_posterior,
            heldout_exact,
            exact_prior,
            (exact_support > 0).astype(np.int16),
            (heldout_exact_support > 0).astype(np.int16),
        ),
    )
    for name, oof_posterior, test_posterior, deployment_prior, oof_factors, test_factors in candidates:
        for weight in WEIGHTS:
            candidate = base.copy()
            for _fold_train, fold_valid in splits:
                candidate[fold_valid] = _combine(
                    base[fold_valid],
                    oof_posterior[fold_valid],
                    fold_priors[fold_valid][0],
                    weight,
                )
            projected = _preserve_certified_class(base, candidate)
            evaluation = _evaluate(
                encoded_train,
                base,
                projected,
                splits,
                evidence_rows,
            )

            heldout = _combine(
                heldout_base,
                test_posterior,
                deployment_prior,
                weight,
            )
            heldout_projected = _preserve_certified_class(heldout_base, heldout)
            heldout_score = _classification_rank_score(encoded_test, heldout_projected)
            records.append(
                {
                    "dataset": dataset.name,
                    "fold": int(fold_index),
                    "candidate": name,
                    "weight": float(weight),
                    "rows": int(len(train)),
                    "features": int(len(full.features)),
                    "mean_oof_factors": float(oof_factors[evidence_rows].mean()),
                    "mean_heldout_factors": float(test_factors.mean()),
                    **evaluation,
                    "heldout_base_rank_auc": heldout_base_score,
                    "heldout_rank_auc": heldout_score,
                    "heldout_rank_auc_delta": heldout_score - heldout_base_score,
                }
            )
    return records


def _parse_cases(value):
    if not value:
        return DEFAULT_CASES
    cases = []
    for item in value.split(","):
        name, separator, fold = item.rpartition(":")
        if not separator or not name:
            raise ValueError("cases must use dataset:fold entries")
        cases.append((name, int(fold)))
    return tuple(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", help="Comma-separated dataset:fold entries")
    parser.add_argument(
        "--out",
        default="results/tabpvn_numeric_likelihood_probe.csv",
    )
    args = parser.parse_args()
    cases = _parse_cases(args.cases)
    datasets = {
        dataset.name: dataset
        for dataset in tabarena_suite(
            size="all",
            dataset_names=sorted({name for name, _fold in cases}),
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
            f"{record['dataset']} fold={record['fold']} {record['candidate']} "
            f"weight={record['weight']:.2f} "
            f"oof={record['oof_rank_auc_delta']:+.6f} "
            f"heldout={record['heldout_rank_auc_delta']:+.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
