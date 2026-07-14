"""In-process, proof-carrying decision API for fitted TabPVN classifiers."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tabpvn.base import TabPVN


@dataclass
class DecisionResponse:
    """One selective tabular decision and its independently verifiable bundle."""

    id: str
    created: int
    model: str
    results: list[dict[str, Any]]
    strike: float
    payoff: dict[str, float]
    prior: Any
    no_arbitrage: dict[str, Any] | None
    verified: bool
    bundle: dict[str, Any]
    object: str = field(default="tabular.decision", init=False)


class _Decisions:
    def __init__(self, client: TabularDecisionClient):
        self._client = client

    def create(
        self,
        model: str,
        rows: Any,
        reward: float = 1.0,
        penalty: float = 1.0,
        abstain_cost: float = 0.0,
        prior: Any = None,
        epsilon: float | None = None,
        delta: float = 0.05,
        n_bins: int | None = None,
    ) -> DecisionResponse:
        """Create a fair-price decision and re-checkable certificate bundle."""
        from tabpvn.base import TabPVN

        if model not in self._client._models:
            raise ValueError(f"model {model!r} not found; available: {list(self._client._models)}")
        classifier = self._client._models[model]
        if not isinstance(classifier, TabPVN):
            raise ValueError(f"model {model!r} is not a TabPVN classifier; decisions.create needs one")

        bundle = classifier.certified_decision(
            rows,
            reward=reward,
            penalty=penalty,
            abstain_cost=abstain_cost,
            prior=prior,
            epsilon=epsilon,
            delta=delta,
            n_bins=n_bins,
        )
        decision = bundle["decision"]
        results = [
            {
                "index": index,
                "prediction": decision["prediction"][index],
                "answered": bool(decision["answer"][index]),
                "confidence": float(decision["confidence"][index]),
                "expected_value": float(decision["expected_value"][index]),
            }
            for index in range(len(decision["answer"]))
        ]
        no_arbitrage = bundle["no_arbitrage"]
        no_arbitrage_summary = (
            None
            if no_arbitrage is None
            else {
                "holds": no_arbitrage["holds"],
                "certified_edge": no_arbitrage["certified_edge"],
                "empirical_edge": no_arbitrage["empirical_edge"],
                "epsilon": no_arbitrage["epsilon"],
            }
        )
        return DecisionResponse(
            id="tabdec-" + uuid.uuid4().hex[:24],
            created=int(time.time()),
            model=model,
            results=results,
            strike=float(decision["strike"]),
            payoff=decision["payoff"],
            prior=bundle["prior_deploy"],
            no_arbitrage=no_arbitrage_summary,
            verified=bool(bundle["verified"]),
            bundle=bundle,
        )


class TabularDecisionClient:
    """In-process client over named, fitted TabPVN classifiers."""

    def __init__(self, models: Mapping[str, TabPVN]):
        if not models:
            raise ValueError("TabularDecisionClient requires at least one fitted model: {name: TabPVN}")
        from tabpvn.base import TabPVN

        for name, model in models.items():
            if not isinstance(name, str) or not name:
                raise ValueError("model names must be non-empty strings")
            if not isinstance(model, TabPVN):
                raise TypeError(f"model {name!r} is not a TabPVN estimator")
            if not model.__sklearn_is_fitted__():
                raise RuntimeError(f"model {name!r} is not fitted")
            if model.mode != "classification":
                raise ValueError(f"model {name!r} is not a classifier")
        self._models = dict(models)
        self._created = int(time.time())
        self.decisions = _Decisions(self)

    def models_list(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": self._created,
                    "owned_by": "tabpvn",
                }
                for name in self._models
            ],
        }

    @staticmethod
    def verify_decision(response: DecisionResponse | None) -> bool:
        """Re-check a response bundle without using the fitted model."""
        from tabpvn.base import TabPVN

        return bool(response is not None and TabPVN.verify_decision(response.bundle))


__all__ = ["DecisionResponse", "TabularDecisionClient"]
