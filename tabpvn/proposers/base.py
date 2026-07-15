"""Candidate proposer contracts for TabPVN's zero-knob fit pipeline.

The deployed estimator follows a repeated pattern: a candidate component proposes
extra capacity, an out-of-fold or held-out gate decides whether it earned
deployment, and the final certified predictor remains the source of class
soundness. This module makes that pattern explicit without forcing every legacy
gate out of ``TabPVN`` in one refactor.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProposerSpec:
    """Static metadata for one optional capacity component."""

    name: str
    stage: str
    proof_policy: str
    description: str


@dataclass(frozen=True)
class GateReport:
    """Serializable result of evaluating a candidate proposer."""

    name: str
    selected: bool
    stage: str = "candidate"
    metric: str | None = None
    mean_score: float | None = None
    fold_auc_delta: tuple[float, ...] = field(default_factory=tuple)
    reason: str | None = None

    def asdict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "selected": bool(self.selected),
            "stage": self.stage,
        }
        if self.metric is not None:
            out["metric"] = self.metric
        if self.mean_score is not None:
            out["mean_score"] = float(self.mean_score)
        if self.fold_auc_delta:
            out["fold_auc_delta"] = [float(v) for v in self.fold_auc_delta]
        if self.reason is not None:
            out["reason"] = self.reason
        return out


class Proposer(Protocol):
    """Minimal interface for future extracted proposer implementations."""

    spec: ProposerSpec

    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        boost: dict[str, Any],
    ) -> GateReport:
        """Return the gate decision for this proposer."""


class CandidateRegistry:
    """Names the optional proposer set used by the fit pipeline."""

    def __init__(self, specs: Iterable[ProposerSpec]) -> None:
        self._specs = tuple(specs)
        names = [spec.name for spec in self._specs]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate proposer names: {names}")

    @property
    def specs(self) -> tuple[ProposerSpec, ...]:
        return self._specs

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    def describe(self) -> list[dict[str, str]]:
        return [
            {
                "name": spec.name,
                "stage": spec.stage,
                "proof_policy": spec.proof_policy,
                "description": spec.description,
            }
            for spec in self._specs
        ]


def gate_report(name: str, selected: bool, **kwargs: Any) -> dict[str, Any]:
    """Build the public dict shape existing users/tests already consume."""

    known = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in {"stage", "metric", "mean_score", "fold_auc_delta", "reason"}
    }
    out = GateReport(name=name, selected=selected, **known).asdict()
    out.update(kwargs)
    return out


def default_registry() -> CandidateRegistry:
    """The current zero-knob proposer set, in the order the pipeline considers it."""

    return CandidateRegistry(
        (
            ProposerSpec(
                "automatic_event_schema",
                "schema",
                "unlabeled bounded discovery followed by a power-aware future-window gate",
                "Entity, timestamp, and marked-value roles promoted only after causal history wins.",
            ),
            ProposerSpec(
                "target_encoding",
                "schema",
                "raw-data holdout gate; retained features are replayable",
                "Smoothed high-cardinality category statistics selected only after a held-out win.",
            ),
            ProposerSpec(
                "compression_evidence",
                "schema",
                "cross-fit quantized phrase evidence; final class remains booster-certified",
                "Class-balanced byte-phrase codelength features selected only after consistent OOF rank signal.",
            ),
            ProposerSpec(
                "temporal_laplace_evidence",
                "schema",
                "strictly causal future-window gate; final class remains booster-certified",
                "Bounded multiscale entity-history facts selected only after stable forward wins.",
            ),
            ProposerSpec(
                "auto_boost",
                "predictor",
                "certified additive booster",
                "Successive-halving search over the certified booster's structural config.",
            ),
            ProposerSpec(
                "shallow_certified_boost",
                "predictor",
                "certified additive booster",
                "Depth-three readable booster challenger for small and medium binary tables.",
            ),
            ProposerSpec(
                "class_weight",
                "predictor",
                "probability-ranking gate; class proof still comes from booster",
                "Balanced class weighting selected only when OOF ROC-AUC improves.",
            ),
            ProposerSpec(
                "rare_rank_checkpoint",
                "predictor",
                "certified additive booster",
                "Average-precision prefix selected from a shared multi-metric booster trace.",
            ),
            ProposerSpec(
                "multiclass_rank_checkpoint",
                "predictor",
                "certified additive booster",
                "Macro OVO-AUC prefix selected from one coupled softmax training trace.",
            ),
            ProposerSpec(
                "multiclass_residual_stump_head",
                "predictor",
                "certified additive predicate stumps",
                "Class-owned Newton stump corrections selected by shared-fold macro OVO-AUC.",
            ),
            ProposerSpec(
                "linear_leaf",
                "predictor",
                "affine certified leaf intervals",
                "Path-constrained linear/logit-linear leaves selected by held-out or OOF loss.",
            ),
            ProposerSpec(
                "symbolic_predicate_boost",
                "schema",
                "deterministic replayable Boolean program",
                "Finite pair/parity/cardinality predicates selected by two-fold OOF ROC-AUC.",
            ),
            ProposerSpec(
                "threshold_predicate_boost",
                "schema",
                "deterministic replayable threshold program",
                "Bounded pair/triple ordinal clauses selected by layered two-fold OOF ROC-AUC.",
            ),
            ProposerSpec(
                "rare_symbolic_predicate_boost",
                "schema",
                "deterministic replayable rare-event program",
                "Residual-guided tail, interval, conjunction, and cardinality rules selected by shared-fold AP.",
            ),
            ProposerSpec(
                "multiclass_residual_predicate_boost",
                "schema",
                "deterministic replayable multiclass program",
                "Class-balanced one-vs-rest residual predicates selected by shared-fold macro OVO-AUC.",
            ),
            ProposerSpec(
                "monotone_constraints",
                "predictor",
                "certified monotone additive predictor",
                "Discovered monotone invariants retained only if verification does not hurt RMSE.",
            ),
            ProposerSpec(
                "multiclass_prior_fallback",
                "auxiliary_proba",
                "conjunctive shared-OOF no-signal screen; final class remains booster-certified",
                "Dominant multiclass probabilities revert to the training prior only when rank, accuracy, and log-loss lack supported signal.",
            ),
            ProposerSpec(
                "smooth_knn",
                "auxiliary_proba",
                "projected to preserve certified class",
                "Fold-local fixed/adaptive neighborhood with binary distance-concentration screening.",
            ),
            ProposerSpec(
                "global_affine_rank",
                "auxiliary_proba",
                "cross-fit rank gate plus stricter paired top-1 permission",
                "Strongly regularized explicit affine logits complement local boosted regions; class changes require material non-losing OOF accuracy evidence.",
            ),
            ProposerSpec(
                "categorical_posterior",
                "posterior_challenger",
                "cross-fit rank-only or class-change permission over a Dirichlet count update",
                "OOF-selected global/hierarchical shrinkage and strongest/pooled facts require paired evidence to correct classes.",
            ),
            ProposerSpec(
                "numeric_interval_decision",
                "decision_challenger",
                "cross-fit decision permission plus an independent finite-interval rank gate",
                "OOF-supported interval counts may change predict labels; only the preselected accuracy winner may separately challenge public ranking.",
            ),
            ProposerSpec(
                "sdm_attention",
                "auxiliary_proba",
                "projected or conformal-calibrated blend",
                "Sparse distributed-memory text member selected by OOF gain.",
            ),
        )
    )
