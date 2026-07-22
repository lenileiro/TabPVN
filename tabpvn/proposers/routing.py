"""Finite Bayesian routing for proof-compatible probability experts.

The router is deliberately not a learned black box.  A row enters one of a
small number of explicit contexts defined by probability margins and class
agreement.  Fit-time Beta evidence decides which contexts may use an expert;
cross-fitting keeps every routed OOF row disjoint from the labels that enabled
its context.  The caller remains responsible for the final all-fold metric
gate and for preserving any certified class boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class BayesianExpertRouter:
    """Route a bounded probability expert through explicit confidence cells."""

    MARGIN_BINS = (0.25, 0.5, 0.75)
    PRIOR_MASS = 0.5
    MIN_SUPPORT = 8
    MIN_NET_WINS = 2
    POSTERIOR_Z = 1.0

    def __init__(self) -> None:
        self.enabled_contexts_: tuple[int, ...] = ()
        self.context_report_: tuple[dict[str, object], ...] = ()

    @staticmethod
    def _validate_probabilities(base, expert, candidate=None):
        base = np.asarray(base, dtype=float)
        expert = np.asarray(expert, dtype=float)
        if base.ndim != 2 or base.shape != expert.shape or base.shape[1] < 2:
            raise ValueError("base and expert probabilities must share a multiclass matrix shape")
        if not np.isfinite(base).all() or not np.isfinite(expert).all():
            raise ValueError("expert routing requires finite probabilities")
        if candidate is None:
            return base, expert, None
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != base.shape or not np.isfinite(candidate).all():
            raise ValueError("candidate probabilities must align with the routing inputs")
        return base, expert, candidate

    @staticmethod
    def _margin(probability):
        partitioned = np.partition(np.asarray(probability, dtype=float), -2, axis=1)
        return np.clip(partitioned[:, -1] - partitioned[:, -2], 0.0, 1.0)

    @classmethod
    def contexts(cls, base, expert):
        """Encode the finite routing program for every row."""
        base, expert, _candidate = cls._validate_probabilities(base, expert)
        base_margin = cls._margin(base)
        expert_margin = cls._margin(expert)
        margin_bin = np.digitize(base_margin, cls.MARGIN_BINS, right=False).astype(np.int8)
        expert_advantage = expert_margin > base_margin
        class_agreement = expert.argmax(1) == base.argmax(1)
        return (
            margin_bin
            + 4 * expert_advantage.astype(np.int8)
            + 8 * class_agreement.astype(np.int8)
        )

    @staticmethod
    def _context_record(context: int) -> dict[str, object]:
        context = int(context)
        margin_bin = context % 4
        return {
            "context": context,
            "base_margin_bin": margin_bin,
            "base_margin_lower": 0.0 if margin_bin == 0 else BayesianExpertRouter.MARGIN_BINS[margin_bin - 1],
            "base_margin_upper": (
                1.0 if margin_bin == len(BayesianExpertRouter.MARGIN_BINS) else BayesianExpertRouter.MARGIN_BINS[margin_bin]
            ),
            "expert_more_confident": bool((context // 4) % 2),
            "class_agreement": bool((context // 8) % 2),
        }

    def fit(self, base, expert, candidate, target_index, rows=None):
        """Enable contexts whose candidate-vs-incumbent wins have posterior support."""
        base, expert, candidate = self._validate_probabilities(base, expert, candidate)
        target = np.asarray(target_index, dtype=int)
        if target.ndim != 1 or len(target) != len(base):
            raise ValueError("target_index must have one value per routing row")
        if np.any(target < 0) or np.any(target >= base.shape[1]):
            raise ValueError("target_index contains an invalid class column")
        rows = np.arange(len(base), dtype=int) if rows is None else np.asarray(rows, dtype=int)
        if rows.ndim != 1 or not len(rows) or np.any(rows < 0) or np.any(rows >= len(base)):
            raise ValueError("router rows must be a non-empty in-range vector")

        context = self.contexts(base, expert)
        row_probability = np.arange(len(base))
        incumbent_true = np.clip(base[row_probability, target], 1e-15, 1.0)
        candidate_true = np.clip(candidate[row_probability, target], 1e-15, 1.0)
        improvement = np.log(candidate_true) - np.log(incumbent_true)
        reports = []
        enabled = []
        for cell in range(16):
            selected_rows = rows[context[rows] == cell]
            strict = selected_rows[np.abs(improvement[selected_rows]) > 1e-12]
            wins = int(np.sum(improvement[strict] > 0.0))
            losses = int(np.sum(improvement[strict] < 0.0))
            support = wins + losses
            alpha = wins + self.PRIOR_MASS
            beta = losses + self.PRIOR_MASS
            posterior_mean = float(alpha / (alpha + beta))
            posterior_variance = float(
                alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
            )
            posterior_lower = float(
                np.clip(
                    posterior_mean - self.POSTERIOR_Z * np.sqrt(posterior_variance),
                    0.0,
                    1.0,
                )
            )
            selected = bool(
                support >= self.MIN_SUPPORT
                and wins - losses >= self.MIN_NET_WINS
                and posterior_lower > 0.5
            )
            if selected:
                enabled.append(cell)
            if len(selected_rows):
                reports.append(
                    {
                        **self._context_record(cell),
                        "rows": int(len(selected_rows)),
                        "support": support,
                        "wins": wins,
                        "losses": losses,
                        "posterior_mean": posterior_mean,
                        "posterior_lower": posterior_lower,
                        "selected": selected,
                    }
                )
        self.enabled_contexts_ = tuple(enabled)
        self.context_report_ = tuple(reports)
        return self

    def route_mask(self, base, expert):
        base, expert, _candidate = self._validate_probabilities(base, expert)
        if not self.enabled_contexts_:
            return np.zeros(len(base), dtype=bool)
        return np.isin(self.contexts(base, expert), np.asarray(self.enabled_contexts_, dtype=np.int8))

    def apply(self, base, expert, candidate):
        """Choose the already-verified candidate only in enabled contexts."""
        base, expert, candidate = self._validate_probabilities(base, expert, candidate)
        routed = np.asarray(base, dtype=float).copy()
        selected = self.route_mask(base, expert)
        routed[selected] = candidate[selected]
        return routed

    @classmethod
    def cross_fit(
        cls,
        base,
        expert,
        candidate,
        target_index,
        splits: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        evidence_rows=None,
    ):
        """Return leak-safe routed OOF probabilities and the final replay router."""
        base, expert, candidate = cls._validate_probabilities(base, expert, candidate)
        target = np.asarray(target_index, dtype=int)
        evidence = (
            np.arange(len(base), dtype=int)
            if evidence_rows is None
            else np.asarray(evidence_rows, dtype=int)
        )
        routed = base.copy()
        covered = np.zeros(len(base), dtype=bool)
        fold_reports = []
        for train, valid in splits:
            train = np.intersect1d(np.asarray(train, dtype=int), evidence, assume_unique=False)
            valid = np.asarray(valid, dtype=int)
            if not len(train) or not len(valid):
                raise ValueError("router cross-fit folds require fit and validation rows")
            router = cls().fit(base, expert, candidate, target, rows=train)
            routed[valid] = router.apply(base[valid], expert[valid], candidate[valid])
            covered[valid] = True
            fold_reports.append(
                {
                    "fit_rows": int(len(train)),
                    "validation_rows": int(len(valid)),
                    "enabled_contexts": list(router.enabled_contexts_),
                    "routed_rows": int(router.route_mask(base[valid], expert[valid]).sum()),
                }
            )
        if not np.all(covered[evidence]):
            raise ValueError("router cross-fit folds must cover every evidence row")
        final = cls().fit(base, expert, candidate, target, rows=evidence)
        report = {
            "method": "cross_fitted_beta_context_table",
            "contexts": 16,
            "enabled_contexts": list(final.enabled_contexts_),
            "routed_rows": int(final.route_mask(base[evidence], expert[evidence]).sum()),
            "evidence_rows": int(len(evidence)),
            "folds": fold_reports,
            "cells": list(final.context_report_),
        }
        return routed, final, report
