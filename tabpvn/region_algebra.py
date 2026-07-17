"""Exact, bounded algebra for proof-region literals.

Numeric tree paths use intervals of the form ``(lower, upper]`` because right
branches are strict and left branches are inclusive. Categorical paths operate
over a finite one-hot level universe plus ``-1`` for an unseen all-zero block.
Keeping those semantics explicit lets proof clauses simplify eagerly without
changing the model's learned regions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf, isnan
from typing import Literal, TypeAlias

NumericOperator: TypeAlias = Literal[">", "<="]
CategoryOperator: TypeAlias = Literal["in", "not in"]
NumericPredicate: TypeAlias = tuple[int, NumericOperator, float]
CategoryPredicate: TypeAlias = tuple[Literal["cat"], tuple[int, ...], CategoryOperator, tuple[int, ...]]
RegionPredicate: TypeAlias = NumericPredicate | CategoryPredicate


@dataclass(frozen=True, slots=True)
class NumericInterval:
    """A possibly empty numeric interval with exact tree-path boundaries."""

    lower: float = -inf
    upper: float = inf

    def __post_init__(self) -> None:
        if isnan(self.lower) or isnan(self.upper):
            raise ValueError("numeric interval bounds cannot be NaN")

    @classmethod
    def from_literal(cls, operator: str, threshold: float) -> NumericInterval:
        """Create the domain selected by one numeric tree-path literal."""
        threshold = float(threshold)
        if operator == ">":
            return cls(lower=threshold)
        if operator == "<=":
            return cls(upper=threshold)
        raise ValueError(f"unsupported numeric operator: {operator!r}")

    @property
    def is_empty(self) -> bool:
        return self.lower >= self.upper

    @property
    def is_unconstrained(self) -> bool:
        return self.lower == -inf and self.upper == inf

    def intersect(self, other: NumericInterval) -> NumericInterval:
        return NumericInterval(max(self.lower, other.lower), min(self.upper, other.upper))

    def is_disjoint(self, other: NumericInterval) -> bool:
        return self.intersect(other).is_empty

    def is_subset_of(self, other: NumericInterval) -> bool:
        if self.is_empty:
            return True
        if other.is_empty:
            return False
        return self.lower >= other.lower and self.upper <= other.upper

    def difference(self, other: NumericInterval) -> tuple[NumericInterval, ...]:
        """Return ``self \\ other`` as at most two exact intervals."""
        if self.is_empty:
            return ()
        if self.is_disjoint(other):
            return (self,)
        if self.is_subset_of(other):
            return ()

        pieces = (
            NumericInterval(self.lower, min(self.upper, other.lower)),
            NumericInterval(max(self.lower, other.upper), self.upper),
        )
        return tuple(piece for piece in pieces if not piece.is_empty)


@dataclass(frozen=True, slots=True)
class NumericIntervalUnion:
    """A canonical finite union of disjoint ``(lower, upper]`` intervals."""

    intervals: tuple[NumericInterval, ...]

    def __post_init__(self) -> None:
        previous: NumericInterval | None = None
        for interval in self.intervals:
            if interval.is_empty:
                raise ValueError("numeric interval unions cannot contain empty branches")
            if previous is not None:
                if (interval.lower, interval.upper) < (previous.lower, previous.upper):
                    raise ValueError("numeric interval union branches must be sorted")
                if interval.lower <= previous.upper:
                    raise ValueError("overlapping or touching interval branches must be merged")
            previous = interval

    @classmethod
    def canonical(cls, intervals: Sequence[NumericInterval]) -> NumericIntervalUnion:
        """Sort branches, absorb subsets, and merge touching intervals exactly."""
        ordered = sorted(
            (interval for interval in intervals if not interval.is_empty),
            key=lambda interval: (interval.lower, interval.upper),
        )
        merged: list[NumericInterval] = []
        for interval in ordered:
            if not merged or interval.lower > merged[-1].upper:
                merged.append(interval)
                continue
            previous = merged[-1]
            merged[-1] = NumericInterval(previous.lower, max(previous.upper, interval.upper))
        return cls(tuple(merged))

    @property
    def is_empty(self) -> bool:
        return not self.intervals

    def union(self, other: NumericIntervalUnion) -> NumericIntervalUnion:
        return NumericIntervalUnion.canonical(self.intervals + other.intervals)


@dataclass(frozen=True, slots=True)
class CategoryDomain:
    """Allowed levels in one finite one-hot block, including unseen level -1."""

    universe: frozenset[int]
    allowed: frozenset[int]

    def __post_init__(self) -> None:
        if not self.allowed <= self.universe:
            raise ValueError("allowed category levels must belong to the universe")

    @classmethod
    def from_literal(
        cls,
        width: int,
        operator: str,
        levels: Sequence[int],
    ) -> CategoryDomain:
        """Create the domain selected by a categorical tree-path literal."""
        if width < 1:
            raise ValueError("categorical blocks must contain at least one column")
        universe = frozenset({-1, *range(width)})
        selected = frozenset(int(level) for level in levels) & universe
        if operator == "in":
            return cls(universe, selected)
        if operator == "not in":
            return cls(universe, universe - selected)
        raise ValueError(f"unsupported category operator: {operator!r}")

    @property
    def is_empty(self) -> bool:
        return not self.allowed

    @property
    def is_unconstrained(self) -> bool:
        return self.allowed == self.universe

    def _require_same_universe(self, other: CategoryDomain) -> None:
        if self.universe != other.universe:
            raise ValueError("category domains must have the same universe")

    def intersect(self, other: CategoryDomain) -> CategoryDomain:
        self._require_same_universe(other)
        return CategoryDomain(self.universe, self.allowed & other.allowed)

    def is_disjoint(self, other: CategoryDomain) -> bool:
        self._require_same_universe(other)
        return self.allowed.isdisjoint(other.allowed)

    def is_subset_of(self, other: CategoryDomain) -> bool:
        self._require_same_universe(other)
        return self.allowed <= other.allowed

    def difference(self, other: CategoryDomain) -> CategoryDomain:
        self._require_same_universe(other)
        return CategoryDomain(self.universe, self.allowed - other.allowed)

    def to_literal(self, columns: tuple[int, ...]) -> CategoryPredicate | None:
        """Emit one exact predicate, choosing a form that preserves unseen rows."""
        if len(columns) + 1 != len(self.universe):
            raise ValueError("category columns do not match the domain universe")
        if self.is_empty:
            raise ValueError("an empty category domain has no satisfiable literal")
        if self.is_unconstrained:
            return None
        if -1 in self.allowed:
            excluded = tuple(sorted(self.universe - self.allowed))
            return ("cat", columns, "not in", excluded)
        return ("cat", columns, "in", tuple(sorted(self.allowed)))


def canonicalize_conjunction(
    predicates: Sequence[RegionPredicate],
) -> tuple[RegionPredicate, ...] | None:
    """Intersect repeated path literals and return an exact canonical conjunction.

    ``None`` denotes an empty region. Keys retain first-occurrence order while
    numeric bounds and categorical level sets receive a deterministic form.
    """
    numeric: dict[int, NumericInterval] = {}
    categorical: dict[tuple[int, ...], CategoryDomain] = {}
    order: list[tuple[str, int | tuple[int, ...]]] = []

    for predicate in predicates:
        if len(predicate) < 3:
            raise ValueError(f"invalid region predicate: {predicate!r}")
        if predicate[0] == "cat":
            if len(predicate) != 4:
                raise ValueError(f"invalid categorical predicate: {predicate!r}")
            columns = tuple(int(column) for column in predicate[1])
            operator = str(predicate[2])
            levels = tuple(int(level) for level in predicate[3])
            category_domain = CategoryDomain.from_literal(len(columns), operator, levels)
            if columns in categorical:
                category_domain = categorical[columns].intersect(category_domain)
            else:
                order.append(("cat", columns))
            if category_domain.is_empty:
                return None
            categorical[columns] = category_domain
            continue

        if len(predicate) != 3:
            raise ValueError(f"invalid numeric predicate: {predicate!r}")
        feature = int(predicate[0])
        operator = str(predicate[1])
        numeric_domain = NumericInterval.from_literal(operator, float(predicate[2]))
        if feature in numeric:
            numeric_domain = numeric[feature].intersect(numeric_domain)
        else:
            order.append(("num", feature))
        if numeric_domain.is_empty:
            return None
        numeric[feature] = numeric_domain

    result: list[RegionPredicate] = []
    for kind, key in order:
        if kind == "cat":
            if not isinstance(key, tuple):
                raise AssertionError("invalid categorical region key")
            literal = categorical[key].to_literal(key)
            if literal is not None:
                result.append(literal)
            continue

        if not isinstance(key, int):
            raise AssertionError("invalid numeric region key")
        numeric_domain = numeric[key]
        if numeric_domain.lower != -inf:
            result.append((key, ">", numeric_domain.lower))
        if numeric_domain.upper != inf:
            result.append((key, "<=", numeric_domain.upper))
    return tuple(result)
