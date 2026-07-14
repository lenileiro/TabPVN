"""Tabular decision API — a fitted TabPVN classifier served via decisions.create, one certified decision per
request (Bayes posterior + fair strike + no-arbitrage), re-checkable with verify_decision and no model."""

import pytest

from tabpvn import TabPVN
from tabpvn.api import TabularDecisionClient


def test_api_tabular_decisions_endpoint():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=4000, n_features=16, n_informative=7, random_state=1)
    clf = TabPVN().fit(X[:3200], y[:3200])
    client = TabularDecisionClient(models={"risk": clf})
    resp = client.decisions.create(
        model="risk",
        rows=X[3200:3210],
        reward=1.0,
        penalty=9.0,
        abstain_cost=0.1,
        prior={0: 0.9, 1: 0.1},
        epsilon=0.11,
        n_bins=5,
    )
    assert resp.object == "tabular.decision" and resp.verified and TabularDecisionClient.verify_decision(resp)
    assert len(resp.results) == 10 and resp.no_arbitrage["holds"]
    # a row below the strike abstains (prediction None); above it answers
    assert any(r["prediction"] is None and not r["answered"] for r in resp.results) or all(
        r["confidence"] >= resp.strike for r in resp.results
    )
    for r in resp.results:
        assert (r["prediction"] is None) == (not r["answered"])  # answer <=> a prediction
    # wrong model registries and endpoints are rejected, not faked
    with pytest.raises(TypeError, match="not a TabPVN"):
        TabularDecisionClient(models={"notclf": object()})
    with pytest.raises(ValueError):
        client.decisions.create(model="nope", rows=X[:5])
