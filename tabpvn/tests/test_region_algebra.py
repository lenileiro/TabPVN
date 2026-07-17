"""Exact simplification tests for proof-region literals."""

import numpy as np

from core.kernel_fol import FOLKernel
from tabpvn.certified_boost import _region_facts, _region_rule
from tabpvn.region_algebra import (
    CategoryDomain,
    NumericInterval,
    NumericIntervalUnion,
    canonicalize_conjunction,
)


def _fired_rows(X, predicates):
    head, body, inputs = _region_rule("test", predicates)
    facts = _region_facts(X, range(len(X)), inputs)
    fired, _ = FOLKernel([(head, body)]).closure(facts)
    return {fact[1] for fact in fired if fact[0] == head[0]}, body


def test_numeric_interval_operations_preserve_open_closed_boundaries():
    outer = NumericInterval(-3.0, 8.0)
    inner = NumericInterval(1.0, 5.0)

    assert outer.intersect(inner) == inner
    assert inner.is_subset_of(outer)
    assert not outer.is_disjoint(inner)
    assert outer.difference(inner) == (
        NumericInterval(-3.0, 1.0),
        NumericInterval(5.0, 8.0),
    )
    assert NumericInterval(-3.0, 1.0).is_disjoint(NumericInterval(1.0, 5.0))


def test_numeric_interval_union_canonicalizes_without_distributing_branches():
    union = NumericIntervalUnion.canonical(
        (
            NumericInterval(8.0, 9.0),
            NumericInterval(0.0, 2.0),
            NumericInterval(-2.0, 1.0),
            NumericInterval(1.0, 3.0),
            NumericInterval(9.0, 10.0),
        )
    )

    assert union.intervals == (
        NumericInterval(-2.0, 3.0),
        NumericInterval(8.0, 10.0),
    )
    assert union.union(NumericIntervalUnion((NumericInterval(12.0, 13.0),))).intervals == (
        NumericInterval(-2.0, 3.0),
        NumericInterval(8.0, 10.0),
        NumericInterval(12.0, 13.0),
    )


def test_category_operations_include_the_unseen_level():
    north_or_unseen = CategoryDomain.from_literal(3, "not in", (1, 2))
    seen = CategoryDomain.from_literal(3, "in", (0, 1, 2))

    assert north_or_unseen.allowed == frozenset({-1, 0})
    assert north_or_unseen.intersect(seen).allowed == frozenset({0})
    assert north_or_unseen.difference(seen).allowed == frozenset({-1})
    assert CategoryDomain.from_literal(3, "in", (1,)).is_disjoint(north_or_unseen)


def test_conjunction_keeps_only_strongest_numeric_and_category_literals():
    predicates = [
        (0, ">", -2.0),
        (0, "<=", 9.0),
        (0, ">", 1.0),
        (0, "<=", 4.0),
        ("cat", (1, 2, 3), "in", (0, 1, 2)),
        ("cat", (1, 2, 3), "not in", (1,)),
        ("cat", (1, 2, 3), "in", (1, 2)),
    ]

    assert canonicalize_conjunction(predicates) == (
        (0, ">", 1.0),
        (0, "<=", 4.0),
        ("cat", (1, 2, 3), "in", (2,)),
    )


def test_canonical_horn_clause_matches_original_path_and_reuses_facts():
    X = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
            [4.5, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0],
        ]
    )
    predicates = [
        (0, ">", -2.0),
        (0, "<=", 9.0),
        (0, ">", 1.0),
        (0, "<=", 4.0),
        ("cat", (1, 2, 3), "in", (0, 1, 2)),
        ("cat", (1, 2, 3), "not in", (1,)),
        ("cat", (1, 2, 3), "in", (1, 2)),
    ]

    rows, body = _fired_rows(X, predicates)
    expected = {
        row
        for row, values in enumerate(X)
        if values[0] > -2.0 and values[0] <= 9.0 and values[0] > 1.0 and values[0] <= 4.0 and values[3] > 0.5
    }

    assert rows == expected == {2, 3}
    assert body == [
        ("feat", "R", 0, "V0"),
        ("cmp", ">", "V0", 1.0),
        ("cmp", "<=", "V0", 4.0),
        ("cat", "R", (1, 2, 3), "V1"),
        ("cmp", "in", "V1", (2,)),
    ]


def test_negated_category_canonicalization_preserves_unseen_rows():
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    predicates = [
        ("cat", (0, 1, 2), "not in", (0,)),
        ("cat", (0, 1, 2), "not in", (2,)),
    ]

    rows, body = _fired_rows(X, predicates)

    assert rows == {1, 3}
    assert body[-1] == ("cmp", "not in", "V0", (0, 2))


def test_empty_conjunction_compiles_to_a_non_firing_safe_clause():
    X = np.array([[0.0], [1.0], [2.0]])
    predicates = [(0, ">", 2.0), (0, "<=", 2.0)]

    rows, body = _fired_rows(X, predicates)

    assert canonicalize_conjunction(predicates) is None
    assert rows == set()
    assert body == [("feat", "R", 0, "V0"), ("cmp", "==", 0, 1)]
