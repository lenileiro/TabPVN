"""Fit pipeline facade for TabPVN.

``FitPipeline`` is the stable orchestration boundary between public fit routing
and the shared certified predictor. Ordinary and event-aware fits establish
their validation geometry before entering this same predictor lifecycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tabpvn.proposers import CandidateRegistry, default_registry


class _PipelineModel(Protocol):
    fit_pipeline_: dict[str, object]
    fit_stages_: object
    proposer_registry_: tuple[str, ...]

    def _fit_predictor(self, data: Any, y: Any = None, target: Any = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class PipelineStage:
    name: str
    component: str
    description: str

    def asdict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "component": self.component,
            "description": self.description,
        }


DEFAULT_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        "schema",
        "SchemaCompiler",
        "Compile numeric, datetime, missingness, category, text, and gated compression-evidence facts.",
    ),
    PipelineStage(
        "candidate_gates",
        "CandidateRegistry",
        "Evaluate optional proposers with held-out or out-of-fold gates.",
    ),
    PipelineStage(
        "certified_predictor",
        "AdditiveCertifiedPredictor",
        "Fit the selected proof-carrying booster path.",
    ),
    PipelineStage(
        "confidence",
        "CertifiedConfidence",
        "Calibrate conformal or selective precision guarantees on leak-safe predictions.",
    ),
    PipelineStage(
        "reports",
        "FitReport",
        "Expose selected config, candidate decisions, and pipeline metadata.",
    ),
)


class FitPipeline:
    """Orchestrates fitting and records the architectural stages used."""

    def __init__(
        self,
        candidate_registry: CandidateRegistry | None = None,
        stages: Sequence[PipelineStage] = DEFAULT_STAGES,
    ) -> None:
        self.candidate_registry = default_registry() if candidate_registry is None else candidate_registry
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("FitPipeline requires at least one stage")

    def describe(self) -> dict[str, object]:
        return {
            "stages": [stage.asdict() for stage in self.stages],
            "proposers": self.candidate_registry.describe(),
        }

    def fit(
        self,
        model: _PipelineModel,
        data: Any,
        y: Any = None,
        target: Any = None,
    ) -> Any:
        """Record the pipeline contract and delegate to the current implementation."""

        model.fit_pipeline_ = self.describe()
        model.fit_stages_ = model.fit_pipeline_["stages"]
        model.proposer_registry_ = self.candidate_registry.names()
        return model._fit_predictor(data, y=y, target=target)


__all__ = ["DEFAULT_STAGES", "FitPipeline", "PipelineStage"]
