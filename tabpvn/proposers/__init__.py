"""Proposer registry for TabPVN's fit-time candidate gates."""

from tabpvn.proposers.affine import AffineLogitRead
from tabpvn.proposers.base import (
    CandidateRegistry,
    GateReport,
    Proposer,
    ProposerSpec,
    default_registry,
    gate_report,
)
from tabpvn.proposers.evidence import ClassificationEvidence, ClassificationEvidenceWorkspace
from tabpvn.proposers.posterior import (
    CategoricalPosteriorChallenger,
    NumericIntervalPosteriorChallenger,
)
from tabpvn.proposers.temporal import TemporalEvidenceChallenger

__all__ = [
    "CandidateRegistry",
    "AffineLogitRead",
    "GateReport",
    "Proposer",
    "ProposerSpec",
    "default_registry",
    "gate_report",
    "ClassificationEvidence",
    "ClassificationEvidenceWorkspace",
    "CategoricalPosteriorChallenger",
    "NumericIntervalPosteriorChallenger",
    "TemporalEvidenceChallenger",
]
