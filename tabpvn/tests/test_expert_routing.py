"""Focused coverage for the finite Bayesian mixture-of-experts router."""

import numpy as np

from tabpvn import TabPVN
from tabpvn.base import _preserve_certified_class
from tabpvn.proposers import BayesianExpertRouter


def test_cross_fitted_router_keeps_only_the_reliable_expert_context():
    rows = np.arange(480)
    target = rows % 2
    reliable = rows % 4 < 2
    base_positive = np.where(target == 1, 0.55, 0.45)
    expert_positive = np.where(
        reliable,
        np.where(target == 1, 0.9, 0.1),
        np.where(target == 1, 0.1, 0.9),
    )
    base = np.column_stack((1.0 - base_positive, base_positive))
    expert = np.column_stack((1.0 - expert_positive, expert_positive))
    candidate = 0.5 * base + 0.5 * expert
    splits = tuple(
        (np.setdiff1d(rows, rows[fold::3]), rows[fold::3])
        for fold in range(3)
    )

    routed, router, report = BayesianExpertRouter.cross_fit(
        base,
        expert,
        candidate,
        target,
        splits,
    )

    np.testing.assert_allclose(routed[reliable], candidate[reliable])
    np.testing.assert_allclose(routed[~reliable], base[~reliable])
    assert len(router.enabled_contexts_) == 1
    assert router.route_mask(base, expert).tolist() == reliable.tolist()
    assert report["method"] == "cross_fitted_beta_context_table"
    assert report["routed_rows"] == int(reliable.sum())
    selected = [cell for cell in report["cells"] if cell["selected"]]
    assert selected[0]["wins"] == int(reliable.sum())
    assert selected[0]["losses"] == 0


def test_router_rejects_contexts_without_posterior_support():
    rows = np.arange(60)
    target = rows % 2
    base = np.tile([0.55, 0.45], (len(rows), 1))
    expert = np.tile([0.45, 0.55], (len(rows), 1))
    candidate = 0.5 * base + 0.5 * expert

    router = BayesianExpertRouter().fit(base, expert, candidate, target)

    assert router.enabled_contexts_ == ()
    np.testing.assert_allclose(router.apply(base, expert, candidate), base)


def test_selected_category_router_is_replayed_by_probability_inference():
    rows = np.arange(240)
    target = rows % 2
    reliable = rows % 4 < 2
    base_positive = np.where(target == 1, 0.55, 0.45)
    expert_positive = np.where(
        reliable,
        np.where(target == 1, 0.9, 0.1),
        np.where(target == 1, 0.1, 0.9),
    )
    base = np.column_stack((1.0 - base_positive, base_positive))
    expert = np.column_stack((1.0 - expert_positive, expert_positive))
    candidate = _preserve_certified_class(base, 0.5 * base + 0.5 * expert)
    router = BayesianExpertRouter().fit(base, expert, candidate, target)
    X = np.column_stack((base, expert))

    class Predictor:
        def _scores(self, query):
            return np.log(np.asarray(query)[:, :2])

    class Memory:
        def proba(self, query):
            return np.asarray(query)[:, 2:4]

    model = TabPVN(seed=0)
    model._pred = Predictor()
    model._temp = 1.0
    model._category_memory = Memory()
    model._category_memory_w = 0.5
    model._category_memory_router = router

    probability = model._blended_proba(X)

    np.testing.assert_allclose(probability, router.apply(base, expert, candidate))
    np.testing.assert_array_equal(probability.argmax(1), base.argmax(1))
