"""Public API for the TabPVN production runtime.

Imports are resolved lazily so metadata checks and process startup do not load
NumPy, pandas, scikit-learn, or the fitting stack until a public object is used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from tabpvn._version import __version__ as __version__

_EXPORTS: dict[str, tuple[str, str]] = {
    "TabPVN": ("tabpvn.base", "TabPVN"),
    "TabPVNMultiOutput": ("tabpvn.adapters", "TabPVNMultiOutput"),
    "TabPVNOrdinal": ("tabpvn.adapters", "TabPVNOrdinal"),
    "derive_features": ("tabpvn.relational", "derive_features"),
    "target_encode": ("tabpvn.preprocessing", "target_encode"),
    "TemporalLaplaceMap": ("tabpvn.temporal", "TemporalLaplaceMap"),
    "DecisionResponse": ("tabpvn.api", "DecisionResponse"),
    "TabularDecisionClient": ("tabpvn.api", "TabularDecisionClient"),
    "load_model": ("tabpvn.model_io", "load_model"),
    "save_model": ("tabpvn.model_io", "save_model"),
    # Backward-compatible decision helpers. The predictor does not import these
    # unless the explicit Bayes/fair-price decision API is used.
    "prior_shift": ("tabpvn.bayes", "prior_shift"),
    "sequential_test": ("tabpvn.bayes", "sequential_test"),
    "test_posterior": ("tabpvn.bayes", "test_posterior"),
    "fair_strike": ("tabpvn.pricing", "fair_strike"),
    "no_arbitrage_report": ("tabpvn.pricing", "no_arbitrage_report"),
    "PROOF_SCHEMA": ("tabpvn.proofs", "PROOF_SCHEMA"),
    "PROOF_ARTIFACT_SCHEMA": ("tabpvn.proofs", "PROOF_ARTIFACT_SCHEMA"),
    "TargetAttestation": ("tabpvn.attestations", "TargetAttestation"),
    "SignedTargetAttestation": ("tabpvn.attestations", "SignedTargetAttestation"),
    "generate_attestation_keypair": ("tabpvn.attestations", "generate_attestation_keypair"),
}

__all__: list[str] = [
    "__version__",
    "TabPVN",
    "TabPVNMultiOutput",
    "TabPVNOrdinal",
    "TabularDecisionClient",
    "DecisionResponse",
    "save_model",
    "load_model",
    "derive_features",
    "target_encode",
    "TemporalLaplaceMap",
    "PROOF_SCHEMA",
    "PROOF_ARTIFACT_SCHEMA",
    "SignedTargetAttestation",
    "TargetAttestation",
    "generate_attestation_keypair",
    "fair_strike",
    "no_arbitrage_report",
    "prior_shift",
    "test_posterior",
    "sequential_test",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
