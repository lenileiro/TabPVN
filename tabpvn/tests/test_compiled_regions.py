"""Exactness tests for the compiled certified-region and evidence indexes."""

import itertools

import numpy as np

from tabpvn.base import _CategoricalEvidenceMemory
from tabpvn.certified_boost import AdditiveCertifiedClassifier, _CompiledRegionGraph


def _numeric_model():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(160, 4))
    y = (X[:, 0] * X[:, 1] + 0.4 * X[:, 2] > 0.0).astype(int)
    model = AdditiveCertifiedClassifier(
        rounds=50, lr=0.1, depth=3, leaf=8, patience=10, seed=3, refit=False
    ).fit(X, y)
    return model, X


def _grid_representatives(model, lo, hi):
    reps = []
    for j in range(len(lo)):
        thresholds = model._thresholds().get(j, np.empty(0))
        inside = thresholds[(thresholds > lo[j]) & (thresholds < hi[j])]
        if len(inside):
            lower = inside[0] - 1.0 if not np.isfinite(lo[j]) else lo[j]
            upper = inside[-1] + 1.0 if not np.isfinite(hi[j]) else hi[j]
            bounds = np.concatenate(([lower], inside, [upper]))
            reps.append((bounds[:-1] + bounds[1:]) / 2.0)
        elif np.isfinite(lo[j]) and np.isfinite(hi[j]):
            reps.append(np.array([(lo[j] + hi[j]) / 2.0]))
        elif np.isfinite(hi[j]):
            reps.append(np.array([hi[j] - 1.0]))
        elif np.isfinite(lo[j]):
            reps.append(np.array([lo[j] + 1.0]))
        else:
            reps.append(np.array([0.0]))
    return np.array(list(itertools.product(*reps)))


def test_compiled_region_graph_matches_exhaustive_pair_margins_and_falls_closed():
    model, X = _numeric_model()
    report = model.compiled_region_report()
    graph = model._region_graph

    assert graph is not None
    assert report["active"] is True
    assert report["factors"] == len(model.trees_)
    assert report["atomic_predicates"] > 0

    lo, hi = np.full(X.shape[1], -np.inf), np.full(X.shape[1], np.inf)
    grid = _grid_representatives(model, lo, hi)
    scores = model._scores(grid)
    smin, smax = model._score_interval(lo, hi)

    for winner, challenger in ((0, 1), (1, 0)):
        expected = float((scores[:, winner] - scores[:, challenger]).min())
        actual = graph.margin_lower(lo, hi, winner, challenger, cap=100_000)
        independent = float(smin[winner] - smax[challenger])

        assert actual is not None
        assert np.isclose(actual, np.nextafter(expected, -np.inf), atol=1e-10)
        assert actual >= independent - 1e-10

    # An inadequate contraction budget never produces an optimistic answer.
    assert graph.margin_lower(lo, hi, 0, 1, cap=1) is None
    fallback = model._margin_lower_bound(lo, hi, 0, cap=1)
    ibp_min, ibp_max = model._score_interval(lo, hi, cap=1)
    assert np.isclose(fallback, ibp_min[0] - ibp_max[1])
    assert model.kernel_certify(X[:16], n_trees=5)["scores_reproduced"] == 1.0


def test_compiled_region_graph_skips_wide_feature_graphs_without_querying_them():
    # The bounded compiler must never turn a one-hot-wide default fit into an
    # expensive graph-ordering exercise.
    trees, flats = [], []
    for feature in range(129):
        trees.append((feature % 2, ("leaf", 0.0)))
        flats.append(
            (
                np.array([feature, -1, -1]),
                np.array([0.0, 0.0, 0.0]),
                np.array([1, -1, -1]),
                np.array([2, -1, -1]),
                np.array([0.0, -1.0, 1.0]),
            )
        )
    graph = _CompiledRegionGraph(trees, flats, 0.1, np.array([0.0, 0.0]), 129)

    assert graph.report()["active"] is False
    assert graph.report()["skip_reason"] == "feature_budget"
    assert graph.margin_lower(np.full(129, -np.inf), np.full(129, np.inf), 0, 1) is None


def test_compiled_margin_certifies_correlated_class_trees_that_ibp_cannot():
    # Both class scores move together with feature 0, so class 0 has a fixed
    # +0.5 margin. Independent per-class intervals lose that correlation.
    tree0 = ("node", 0, 0.0, ("leaf", 0.0), ("leaf", 1.0))
    tree1 = ("node", 0, 0.0, ("leaf", -0.5), ("leaf", 0.5))
    flat0 = (
        np.array([0, -1, -1]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1, -1, -1]),
        np.array([2, -1, -1]),
        np.array([0.0, 0.0, 1.0]),
    )
    flat1 = (
        np.array([0, -1, -1]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1, -1, -1]),
        np.array([2, -1, -1]),
        np.array([0.0, -0.5, 0.5]),
    )
    model = AdditiveCertifiedClassifier()
    model.base_ = np.array([0.0, 0.0])
    model.lr_ = 1.0
    model.trees_ = [(0, tree0), (1, tree1)]
    model.classes_ = [0, 1]
    model._flat_cache = [flat0, flat1]
    model.linear_ = False
    model.scale_ = np.array([1.0])
    model._region_graph = _CompiledRegionGraph(model.trees_, model._flat_cache, 1.0, model.base_, 1)
    X = np.array([[-0.25]])

    compiled = model.certified_robustness(X, 0, delta=10.0)
    model._region_graph = None
    fallback = model.certified_robustness(X, 0, delta=10.0)

    assert compiled["certified_stable"] is True
    assert compiled["margin"] == 0.5
    assert fallback["certified_stable"] is False
    assert fallback["margin"] == -0.5


def _dense_evidence_proba(memory, X):
    """Reference definition with the same explicit score/index tie ordering."""
    codes = memory._codes(X)
    out = np.empty((len(codes), memory.C))
    rows = np.arange(memory.n)
    for row, code in enumerate(codes):
        score = memory._similarity(code[None, :])[0]
        indices = np.lexsort((rows, -score))[: memory.k]
        local = score[indices]
        weight = np.exp((local - local.max()) / memory.temp)
        vote = np.bincount(memory.yidx[indices], weights=weight, minlength=memory.C).astype(float)
        vote /= np.maximum(memory.prior, 1e-12)
        out[row] = vote / np.maximum(vote.sum(), 1e-12)
    return out


def test_postings_category_evidence_matches_dense_atomic_fact_reference():
    rng = np.random.default_rng(5)
    widths = (3, 4, 5, 2)
    n = 180
    codes = [rng.integers(0, width, size=n) for width in widths]
    blocks = []
    groups = []
    start = 0
    for code, width in zip(codes, widths, strict=False):
        block = np.zeros((n, width), float)
        block[np.arange(n), code] = 1.0
        blocks.append(block)
        groups.append(tuple(range(start, start + width)))
        start += width
    X = np.concatenate(blocks, axis=1)
    y = (codes[0] + 2 * codes[1] + codes[2]) % 3
    memory = _CategoricalEvidenceMemory(X, y, [0, 1, 2], groups, seed=9)

    query = X[:24].copy()
    query[0] = 0.0  # exercise the deterministic zero-overlap fill path
    actual = memory.proba(query)
    indexed = memory._indexed_proba(memory._codes(query))
    expected = _dense_evidence_proba(memory, query)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(indexed, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(actual.sum(1), 1.0)
    assert memory.index_report()["postings"] == n * len(widths)
    assert memory.index_report()["read_backend"] == "dense"
