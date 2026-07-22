"""Deterministic, replayable symbolic predicates for proof-carrying tables.

The compiler operates on binary facts already emitted by TabPVN's schema
encoder: one-hot category levels, missingness flags, native binary fields, and
bounded threshold facts over numeric columns. It proposes only finite programs
that can be replayed exactly at prediction time. Statistical selection belongs
to ``TabPVN._auto_interactions``; this module never trains a neural model or
produces a prediction by itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import lgamma, log, log2
from typing import Any

import numpy as np

from tabpvn.region_algebra import NumericInterval, NumericIntervalUnion


@dataclass(frozen=True)
class Predicate:
    """A replayable Boolean program over finite input facts."""

    kind: str  # binary projections or bounded threshold programs
    columns: tuple[int, ...]
    value: int
    thresholds: tuple[float, ...] = ()
    # Threshold programs: True means <= and False means >. Binary programs:
    # True requires the fact and False requires its negation. For
    # ``binary_dnf``, value is the first branch's literal count. Recursive DNF
    # programs store every branch width explicitly and use value as the branch
    # count so replay never has to infer tree structure. Exception programs use
    # value as the number of clauses subtracted from the first (base) branch.
    directions: tuple[bool, ...] = ()
    branch_widths: tuple[int, ...] = ()


BinaryLiteral = tuple[int, bool]
BinaryProgram = tuple[BinaryLiteral, ...]
MdlBranchState = tuple[float, float, float, BinaryProgram, np.ndarray]
MdlPredicateState = tuple[float, float, float, Predicate]
MdlClauseState = tuple[float, BinaryProgram, np.ndarray]


class SymbolicPredicateMap:
    """Compile a bounded predicate hierarchy from binary tabular facts.

    Pair mutual information finds interactions that univariate screening misses
    (notably XOR). The compiler then materializes informative pair states,
    Boolean projections, and exact-cardinality programs. It is deliberately
    bounded: the booster is the general learner, while this map supplies only
    concepts that axis-aligned splits represent inefficiently.
    """

    MIN_ROWS = 400
    # Two-fold candidate selection compiles on one half of an eligible table.
    # The outer gate supplies the additional robustness, so 200 rows is enough
    # to propose a bounded program inside an otherwise 400-row deployment fit.
    MIN_FIT_ROWS = 200
    MAX_ROWS = 10_000
    MAX_BINARY_COLUMNS = 512
    MAX_PAIR_OPERATIONS = 75_000_000
    PAIR_POOL = 16
    MAX_PREDICATES = 16
    MAX_CARDINALITY_GROUP = 6
    MIN_SUPPORT = 8
    MIN_ASSOCIATION = 0.02
    THRESHOLD_QUANTILES = (0.2, 0.4, 0.6, 0.8)
    MAX_EXACT_THRESHOLDS = 16
    THRESHOLD_LITERAL_POOL = 12
    MAX_THRESHOLD_PREDICATES = 8
    MAX_THRESHOLD_TRIPLES = 4
    MIN_COMPOSITION_GAIN = 0.002
    AUTO_MIN_ORDINAL_COLUMNS = 24
    RARE_QUANTILES = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995)
    RARE_CLASS_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    RARE_LITERAL_POOL = 24
    RARE_LITERALS_PER_COLUMN = 2
    RARE_THRESHOLD_MAX = 24
    MAX_RARE_INTERVALS = 6
    MAX_RARE_INTERVAL_UNIONS = 2
    # Only the strongest finite intervals reach exact residual subtraction.
    # Two disjoint regions per field are enough to represent common bimodal
    # risk without widening the global six-interval feature budget.
    RARE_INTERVAL_POOL_PER_COLUMN = 16
    RARE_INTERVALS_PER_COLUMN = 2
    MAX_RARE_PAIRS = 12
    MAX_RARE_TRIPLES = 4
    RARE_MIN_EVENTS = 4
    RARE_MIN_ASSOCIATION = 0.005
    RESIDUAL_MIN_GAIN = 1e-5
    RESIDUAL_COMPOSITION_GAIN = 1e-5
    MAX_RESIDUAL_PREDICATES = 20
    RESIDUAL_NUMERIC_RESERVE = 6
    RESIDUAL_INTERVAL_TRIGGER = 5
    MDL_ATOM_POOL = 12
    MDL_BEAM_WIDTH = 24
    MDL_MAX_LITERALS = 5
    MDL_MAX_PREDICATES = 6
    MDL_DNF_BRANCH_POOL = 12
    MDL_DNF_MAX_TOTAL_LITERALS = 6
    MDL_DNF_MAX_PREDICATES = 3
    MDL_RECURSIVE_DNF_PAIR_BEAM = 48
    MDL_RECURSIVE_DNF_MAX_TOTAL_LITERALS = 9
    MDL_RECURSIVE_DNF_MAX_PREDICATES = 2
    MDL_EXCEPTION_BASE_POOL = 6
    MDL_EXCEPTION_ATOM_BASES = 4
    MDL_EXCEPTION_BRANCH_POOL = 6
    MDL_EXCEPTION_CLAUSE_BEAM = 8
    MDL_EXCEPTION_MAX_CLAUSE_LITERALS = 3
    MDL_EXCEPTION_PAIR_BEAM = 12
    MDL_EXCEPTION_MAX_TOTAL_LITERALS = 9
    MDL_EXCEPTION_MAX_PREDICATES = 2
    MDL_MAX_EVIDENCE_ROWS = 2_048
    MDL_MIN_NET_BITS = 2.0
    MDL_PROOF_BITS_PER_LITERAL = 1.0
    MDL_NEWTON_PRIOR = 1.0

    def __init__(
        self,
        seed=0,
        exclusive_groups=(),
        numeric_rules=False,
        rare_rules=False,
        rare_class=None,
        mdl_beam=True,
        mdl_dnf=True,
        mdl_recursive_dnf=True,
        mdl_exception=True,
    ):
        self.seed = seed
        self.rare_rules = bool(rare_rules)
        self.rare_class = rare_class
        self.numeric_rules = bool(numeric_rules or rare_rules)
        self.mdl_beam = bool(mdl_beam)
        self.mdl_dnf = bool(self.mdl_beam and mdl_dnf)
        self.mdl_recursive_dnf = bool(self.mdl_dnf and mdl_recursive_dnf)
        self.mdl_exception = bool(self.mdl_beam and mdl_exception)
        self.exclusive_groups = tuple(frozenset(group) for group in exclusive_groups)
        self._exclusive_group_for_column = {
            int(column): group_id for group_id, group in enumerate(self.exclusive_groups) for column in group
        }
        self.predicates: list[Predicate] = []
        self.mdl_atom_columns_: tuple[int, ...] = ()

    def _same_exclusive_group(self, columns):
        """Whether two facts are distinct levels of the same categorical field."""
        if len(columns) != 2:
            return False
        left = self._exclusive_group_for_column.get(int(columns[0]))
        right = self._exclusive_group_for_column.get(int(columns[1]))
        return left is not None and left == right

    @staticmethod
    def _binary_columns(X):
        return np.array(
            [j for j in range(X.shape[1]) if np.all((X[:, j] == 0.0) | (X[:, j] == 1.0))],
            dtype=int,
        )

    @classmethod
    def _binary_screen_limit(cls, rows):
        pair_limit = int(np.sqrt(cls.MAX_PAIR_OPERATIONS / max(int(rows), 1)))
        return max(0, min(cls.MAX_BINARY_COLUMNS, pair_limit))

    def _screen_binary_columns(
        self,
        X,
        y,
        columns,
        sample_weight=None,
        residual=None,
        hessian=None,
    ):
        """Bound pairwise work using fit-side evidence plus deterministic exploration."""
        cls = type(self)
        columns = np.asarray(columns, dtype=int)
        limit = cls._binary_screen_limit(len(X))
        self.source_binary_columns_ = int(len(columns))
        if len(columns) <= limit:
            self.screened_binary_columns_ = int(len(columns))
            return columns
        y01 = np.asarray(y) == np.unique(y)[1]
        scores = np.array(
            [
                cls._score(
                    X[:, int(column)].astype(bool),
                    y01,
                    sample_weight,
                    residual,
                    hessian,
                )
                for column in columns
            ],
            dtype=float,
        )
        signal_count = max(1, 3 * limit // 4)
        ranked = np.lexsort((columns, -scores))
        signal = ranked[:signal_count]
        remaining = ranked[signal_count:]
        explore_count = limit - len(signal)
        if explore_count:
            rng = np.random.default_rng(self.seed + 811)
            exploration = rng.permutation(remaining)[:explore_count]
            selected = np.r_[signal, exploration]
        else:
            selected = signal
        screened = np.sort(columns[selected]).astype(int, copy=False)
        self.screened_binary_columns_ = int(len(screened))
        return screened

    @staticmethod
    def _numeric_columns(X):
        return np.array(
            [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) >= 3],
            dtype=int,
        )

    @classmethod
    def auto_numeric_applicable(cls, X):
        """Whether a table is wide enough in bounded ordinal facts to pay the gate cost."""
        X = np.asarray(X, float)
        ordinal = sum(
            3 <= len(np.unique(X[:, column])) <= cls.MAX_EXACT_THRESHOLDS for column in range(X.shape[1])
        )
        return ordinal >= cls.AUTO_MIN_ORDINAL_COLUMNS

    @classmethod
    def applicable(cls, X, y, numeric_rules=False):
        X, y = np.asarray(X, float), np.asarray(y)
        if not (cls.MIN_ROWS <= len(y) <= cls.MAX_ROWS and np.isfinite(X).all()):
            return False
        if len(np.unique(y)) != 2:
            return False
        n_binary = len(cls._binary_columns(X))
        binary_ok = n_binary >= 4 and cls._binary_screen_limit(len(y)) >= 4
        n_numeric = len(cls._numeric_columns(X))
        numeric_ok = (
            bool(numeric_rules)
            and 2 <= n_numeric <= cls.MAX_BINARY_COLUMNS
            and len(y) * n_numeric <= cls.MAX_PAIR_OPERATIONS
        )
        return binary_ok or numeric_ok

    @classmethod
    def _fittable(cls, X, y, numeric_rules=False):
        X, y = np.asarray(X, float), np.asarray(y)
        if not (cls.MIN_FIT_ROWS <= len(y) <= cls.MAX_ROWS and np.isfinite(X).all()):
            return False
        if len(np.unique(y)) != 2:
            return False
        n_binary = len(cls._binary_columns(X))
        binary_ok = n_binary >= 4 and cls._binary_screen_limit(len(y)) >= 4
        n_numeric = len(cls._numeric_columns(X))
        numeric_ok = (
            bool(numeric_rules)
            and 2 <= n_numeric <= cls.MAX_BINARY_COLUMNS
            and len(y) * n_numeric <= cls.MAX_PAIR_OPERATIONS
        )
        return binary_ok or numeric_ok

    @staticmethod
    def _pair_information(bits, y01):
        """Mutual information I((bit_i, bit_j); target) for every pair."""
        B = np.asarray(bits, float)
        pos = B[y01]
        neg = B[~y01]
        n_pos, n_neg = len(pos), len(neg)
        n = n_pos + n_neg

        def states(M):
            one = M.sum(0)
            both = M.T @ M
            ten = one[:, None] - both
            zero_one = one[None, :] - both
            zero_zero = len(M) - both - ten - zero_one
            return (zero_zero, zero_one, ten, both)

        p_states, n_states = states(pos), states(neg)
        py = (n_neg / n, n_pos / n)
        info = np.zeros((B.shape[1], B.shape[1]), float)
        for c0, c1 in zip(n_states, p_states, strict=False):
            total = c0 + c1
            p_state = total / n
            for count, prior in ((c0, py[0]), (c1, py[1])):
                joint = count / n
                valid = joint > 0
                info[valid] += joint[valid] * np.log(joint[valid] / (p_state[valid] * prior))
        np.fill_diagonal(info, -np.inf)
        return info

    @staticmethod
    def _association(mask, y01, sample_weight=None):
        """AUC-equivalent association for a binary predicate, without sklearn."""
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if len(weights) != len(mask):
                raise ValueError("sample_weight must have one value per predicate row")
            mass1 = float(weights[mask].sum())
            mass0 = float(weights[~mask].sum())
            if mass0 <= 0.0 or mass1 <= 0.0:
                return 0.0
            p1 = float(np.dot(weights[mask], y01[mask]) / mass1)
            p0 = float(np.dot(weights[~mask], y01[~mask]) / mass0)
            return abs(p1 - p0) * 0.5
        n1 = int(mask.sum())
        n0 = len(mask) - n1
        if n0 == 0 or n1 == 0:
            return 0.0
        p1 = float(y01[mask].mean())
        p0 = float(y01[~mask].mean())
        # For a binary score, AUC - 0.5 is half the class-rate separation.
        return abs(p1 - p0) * 0.5

    @staticmethod
    def _residual_gain(mask, residual, hessian, sample_weight=None):
        """Normalized Newton gain of splitting a signed residual by ``mask``."""
        mask = np.asarray(mask, dtype=bool)
        residual = np.asarray(residual, dtype=float)
        hessian = np.asarray(hessian, dtype=float)
        weights = (
            np.ones(len(mask), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if not (len(mask) == len(residual) == len(hessian) == len(weights)):
            raise ValueError("residual evidence must align with predicate rows")
        if mask.sum() == 0 or mask.sum() == len(mask):
            return 0.0
        weighted_gradient = weights * residual
        weighted_hessian = weights * np.clip(hessian, 1e-9, None)
        gradient_left = float(weighted_gradient[mask].sum())
        hessian_left = float(weighted_hessian[mask].sum())
        gradient_right = float(weighted_gradient[~mask].sum())
        hessian_right = float(weighted_hessian[~mask].sum())
        gradient_total = gradient_left + gradient_right
        hessian_total = hessian_left + hessian_right
        gain = (
            gradient_left**2 / (hessian_left + 1.0)
            + gradient_right**2 / (hessian_right + 1.0)
            - gradient_total**2 / (hessian_total + 1.0)
        )
        return max(0.0, 0.5 * gain / max(float(weights.sum()), 1e-12))

    @classmethod
    def _score(cls, mask, y01, sample_weight=None, residual=None, hessian=None):
        if residual is None:
            return cls._association(mask, y01, sample_weight)
        return cls._residual_gain(mask, residual, hessian, sample_weight)

    @staticmethod
    def _predicate_values(X, predicate: Predicate) -> np.ndarray:
        """Replay one finite predicate without depending on fitted mapper state."""
        X = np.asarray(X, float)
        if predicate.kind == "threshold_union":
            if not (len(predicate.columns) == len(predicate.thresholds) == len(predicate.directions)):
                raise ValueError("threshold union literals must be aligned")
            if len(predicate.thresholds) < 2 or len(predicate.thresholds) % 2:
                raise ValueError("threshold unions require complete interval branches")
            combined: np.ndarray = np.zeros(len(X), dtype=bool)
            for index in range(0, len(predicate.thresholds), 2):
                lower_column, upper_column = predicate.columns[index : index + 2]
                lower, upper = predicate.thresholds[index : index + 2]
                lower_direction, upper_direction = predicate.directions[index : index + 2]
                lower_mask = X[:, lower_column] <= lower if lower_direction else X[:, lower_column] > lower
                upper_mask = X[:, upper_column] <= upper if upper_direction else X[:, upper_column] > upper
                combined |= lower_mask & upper_mask
            return combined

        values = X[:, predicate.columns]
        if predicate.kind == "state":
            state = 2 * values[:, 0].astype(np.int8) + values[:, 1].astype(np.int8)
            return state == predicate.value
        if predicate.kind == "or":
            return values.max(1) > 0.5
        if predicate.kind == "xor":
            return values[:, 0] != values[:, 1]
        if predicate.kind == "binary_and":
            if len(predicate.columns) != len(predicate.directions) or len(predicate.columns) < 2:
                raise ValueError("binary conjunction literals must be aligned")
            literals = [
                values[:, index] > 0.5 if positive else values[:, index] <= 0.5
                for index, positive in enumerate(predicate.directions)
            ]
            return np.logical_and.reduce(literals)
        if predicate.kind in {"binary_dnf", "binary_recursive_dnf"}:
            branch_widths = SymbolicPredicateMap._binary_dnf_widths(predicate)
            literals = [
                values[:, index] > 0.5 if positive else values[:, index] <= 0.5
                for index, positive in enumerate(predicate.directions)
            ]
            combined = np.zeros(len(X), dtype=bool)
            start = 0
            for width in branch_widths:
                combined |= np.logical_and.reduce(literals[start : start + width])
                start += width
            return combined
        if predicate.kind == "binary_exception":
            branch_widths = SymbolicPredicateMap._binary_exception_widths(predicate)
            literals = [
                values[:, index] > 0.5 if positive else values[:, index] <= 0.5
                for index, positive in enumerate(predicate.directions)
            ]
            branches = []
            start = 0
            for width in branch_widths:
                branches.append(np.logical_and.reduce(literals[start : start + width]))
                start += width
            return branches[0] & ~np.logical_or.reduce(branches[1:])
        if predicate.kind in {"threshold_and", "threshold_or", "threshold_interval"}:
            literals = [
                values[:, index] <= threshold if direction else values[:, index] > threshold
                for index, (threshold, direction) in enumerate(
                    zip(predicate.thresholds, predicate.directions, strict=False)
                )
            ]
            return (
                np.logical_and.reduce(literals)
                if predicate.kind in {"threshold_and", "threshold_interval"}
                else np.logical_or.reduce(literals)
            )
        return values.sum(1) == predicate.value

    @staticmethod
    def _binary_dnf_widths(predicate: Predicate) -> tuple[int, ...]:
        """Validate and return explicit branch widths for a binary OR tree."""
        literal_count = len(predicate.columns)
        if literal_count != len(predicate.directions):
            raise ValueError("binary DNF branches and literals must be aligned")
        if predicate.kind == "binary_dnf":
            split = int(predicate.value)
            widths: tuple[int, ...] = (split, literal_count - split)
        elif predicate.kind == "binary_recursive_dnf":
            widths = tuple(int(width) for width in predicate.branch_widths)
            if len(widths) != 3 or int(predicate.value) != len(widths):
                raise ValueError("recursive binary DNF requires exactly three branches")
        else:
            raise ValueError(f"{predicate.kind!r} is not a binary DNF program")
        if any(width < 1 for width in widths) or sum(widths) != literal_count:
            raise ValueError("binary DNF branches and literals must be aligned")
        return widths

    @staticmethod
    def _binary_exception_widths(predicate: Predicate) -> tuple[int, ...]:
        """Validate and return the base and exception clause widths."""
        literal_count = len(predicate.columns)
        widths = tuple(int(width) for width in predicate.branch_widths)
        if literal_count != len(predicate.directions):
            raise ValueError("binary exception branches and literals must be aligned")
        if len(widths) not in {2, 3} or int(predicate.value) != len(widths) - 1:
            raise ValueError("binary exception programs require one or two exception branches")
        if any(width < 1 for width in widths) or sum(widths) != literal_count:
            raise ValueError("binary exception branches and literals must be aligned")
        return widths

    @staticmethod
    def _bernoulli_code_length(positive_mass: float, negative_mass: float) -> float:
        """Jeffreys-mixture Bernoulli codelength in bits."""
        if positive_mass < 0.0 or negative_mass < 0.0:
            raise ValueError("Bernoulli masses must be non-negative")
        log_probability = (
            lgamma(positive_mass + 0.5)
            + lgamma(negative_mass + 0.5)
            - lgamma(positive_mass + negative_mass + 1.0)
            - 2.0 * lgamma(0.5)
        )
        return float(-log_probability / log(2.0))

    @classmethod
    def _mdl_description_bits(cls, literal_count: int, source_columns: int) -> float:
        """Prefix cost for an AND program plus its directly checkable literals."""
        column_bits = log2(max(2, int(source_columns)))
        width_bits = log2(max(2, int(cls.MDL_MAX_LITERALS)))
        literal_bits = column_bits + 1.0 + float(cls.MDL_PROOF_BITS_PER_LITERAL)
        return float(1.0 + width_bits + literal_count * literal_bits)

    @classmethod
    def _mdl_dnf_description_bits(
        cls,
        branch_widths: tuple[int, int],
        source_columns: int,
    ) -> float:
        """Prefix cost for a two-branch OR-of-AND program."""
        if min(branch_widths) < 1:
            raise ValueError("DNF branches must each contain at least one literal")
        literal_count = int(sum(branch_widths))
        if literal_count > cls.MDL_DNF_MAX_TOTAL_LITERALS:
            raise ValueError("DNF program exceeds the bounded literal budget")
        column_bits = log2(max(2, int(source_columns)))
        literal_bits = column_bits + 1.0 + float(cls.MDL_PROOF_BITS_PER_LITERAL)
        width_bits = log2(max(2, int(cls.MDL_DNF_MAX_TOTAL_LITERALS)))
        branch_bits = log2(max(2, literal_count - 1))
        return float(2.0 + width_bits + branch_bits + literal_count * literal_bits)

    @classmethod
    def _mdl_recursive_dnf_description_bits(
        cls,
        branch_widths: tuple[int, int, int],
        source_columns: int,
    ) -> float:
        """Prefix cost for a depth-three OR tree over conjunction branches."""
        if len(branch_widths) != 3 or min(branch_widths) < 1:
            raise ValueError("recursive DNF requires three non-empty branches")
        literal_count = int(sum(branch_widths))
        if literal_count > cls.MDL_RECURSIVE_DNF_MAX_TOTAL_LITERALS:
            raise ValueError("recursive DNF program exceeds the bounded literal budget")
        column_bits = log2(max(2, int(source_columns)))
        literal_bits = column_bits + 1.0 + float(cls.MDL_PROOF_BITS_PER_LITERAL)
        width_bits = log2(max(2, int(cls.MDL_RECURSIVE_DNF_MAX_TOTAL_LITERALS)))
        # Choosing two separators among literal_count - 1 positions uniquely
        # encodes the three branch widths.
        partitions = max(2, (literal_count - 1) * (literal_count - 2) // 2)
        branch_bits = log2(partitions)
        return float(3.0 + width_bits + branch_bits + literal_count * literal_bits)

    @classmethod
    def _mdl_exception_description_bits(
        cls,
        branch_widths: tuple[int, ...],
        source_columns: int,
    ) -> float:
        """Prefix cost for a base conjunction minus one or two clauses."""
        if len(branch_widths) not in {2, 3} or min(branch_widths) < 1:
            raise ValueError("exception programs require a base and one or two clauses")
        literal_count = int(sum(branch_widths))
        if literal_count > cls.MDL_EXCEPTION_MAX_TOTAL_LITERALS:
            raise ValueError("exception program exceeds the bounded literal budget")
        column_bits = log2(max(2, int(source_columns)))
        literal_bits = column_bits + 1.0 + float(cls.MDL_PROOF_BITS_PER_LITERAL)
        width_bits = log2(max(2, int(cls.MDL_EXCEPTION_MAX_TOTAL_LITERALS)))
        if len(branch_widths) == 2:
            partitions = max(2, literal_count - 1)
        else:
            partitions = max(2, (literal_count - 1) * (literal_count - 2) // 2)
        return float(3.0 + width_bits + log2(partitions) + literal_count * literal_bits)

    @staticmethod
    def _canonical_partition_signature(mask: np.ndarray) -> bytes:
        """A predicate and its complement define the same binary partition."""
        packed = np.packbits(np.asarray(mask, dtype=np.uint8)).tobytes()
        complement = np.packbits(np.asarray(~mask, dtype=np.uint8)).tobytes()
        return min(packed, complement)

    @classmethod
    def _mdl_codelength_scorer(cls, y01, weights, baseline_probability=None):
        """Return the search objective, parameter cost, and partition scorer."""
        if baseline_probability is None:
            positive = weights * y01
            negative = weights * ~y01
            null_code = cls._bernoulli_code_length(float(positive.sum()), float(negative.sum()))

            def target_saved_bits(mask):
                left_positive = float(positive[mask].sum())
                left_negative = float(negative[mask].sum())
                right_positive = float(positive.sum() - left_positive)
                right_negative = float(negative.sum() - left_negative)
                split_code = cls._bernoulli_code_length(
                    left_positive,
                    left_negative,
                ) + cls._bernoulli_code_length(right_positive, right_negative)
                return float(null_code - split_code)

            return "jeffreys_target_codelength", 0.0, target_saved_bits

        baseline_probability = np.asarray(baseline_probability, dtype=float)
        if baseline_probability.ndim != 1 or len(baseline_probability) != len(y01):
            raise ValueError("baseline_probability must have one value per predicate row")
        baseline_probability = np.clip(baseline_probability, 1e-9, 1.0 - 1e-9)
        gradient = weights * (np.asarray(y01, dtype=float) - baseline_probability)
        hessian = weights * baseline_probability * (1.0 - baseline_probability)
        gradient_total = float(gradient.sum())
        hessian_total = float(hessian.sum())
        prior = float(cls.MDL_NEWTON_PRIOR)
        global_gain = gradient_total**2 / (hessian_total + prior)

        def residual_saved_bits(mask):
            gradient_left = float(gradient[mask].sum())
            hessian_left = float(hessian[mask].sum())
            gradient_right = gradient_total - gradient_left
            hessian_right = hessian_total - hessian_left
            split_gain = (
                gradient_left**2 / (hessian_left + prior)
                + gradient_right**2 / (hessian_right + prior)
                - global_gain
            )
            return float(max(0.0, 0.5 * split_gain) / log(2.0))

        # The split adds one calibrated degree of freedom beyond a global
        # intercept correction. BIC is the Laplace approximation to its code.
        parameter_bits = 0.5 * log2(max(2, len(y01)))
        return "certified_booster_residual_laplace_codelength", parameter_bits, residual_saved_bits

    def _bounded_mdl_evidence_rows(self, y, evidence_rows=None):
        """Bound search work while retaining deterministic minority evidence."""
        cls = type(self)
        rows = np.arange(len(y), dtype=int) if evidence_rows is None else np.asarray(evidence_rows, dtype=int)
        self.mdl_source_evidence_rows_ = int(len(rows))
        if len(rows) <= cls.MDL_MAX_EVIDENCE_ROWS:
            self.mdl_evidence_capped_ = False
            return rows

        labels = np.asarray(y)[rows]
        _classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        expected = cls.MDL_MAX_EVIDENCE_ROWS * counts.astype(float) / len(rows)
        quotas = np.minimum(counts, np.floor(expected).astype(int))
        minimum = np.minimum(counts, cls.MIN_SUPPORT)
        quotas = np.maximum(quotas, minimum)
        while int(quotas.sum()) > cls.MDL_MAX_EVIDENCE_ROWS:
            reducible = quotas - minimum
            index = int(np.argmax(reducible))
            if reducible[index] <= 0:
                break
            quotas[index] -= 1
        while int(quotas.sum()) < cls.MDL_MAX_EVIDENCE_ROWS:
            available = counts - quotas
            if not np.any(available > 0):
                break
            deficit = expected - quotas
            deficit[available <= 0] = -np.inf
            quotas[int(np.argmax(deficit))] += 1

        rng = np.random.default_rng(self.seed + 1871)
        selected = [
            rng.permutation(rows[inverse == class_index])[: int(quota)]
            for class_index, quota in enumerate(quotas)
        ]
        self.mdl_evidence_capped_ = True
        return np.sort(np.concatenate(selected).astype(int, copy=False))

    def _mdl_dnf_candidates(
        self,
        X,
        branch_states,
        output_signatures,
        selected_columns,
        source_binary_columns,
        residual_parameter_bits,
        saved_bits,
    ):
        """Compose a bounded pair of conjunction branches under MDL cost."""
        cls = type(self)
        if not self.mdl_dnf or not branch_states:
            return []
        simpler_signatures = set(output_signatures)
        simpler_signatures.update(self._canonical_partition_signature(state[4]) for state in branch_states)
        simpler_signatures.update(
            self._canonical_partition_signature(X[:, column] > 0.5) for column in selected_columns
        )
        branches_by_mask: dict[bytes, MdlBranchState] = {}
        for state in branch_states:
            signature = np.packbits(state[4].astype(np.uint8)).tobytes()
            branch_incumbent = branches_by_mask.get(signature)
            if branch_incumbent is None or (state[0], tuple(state[3])) > (
                branch_incumbent[0],
                tuple(branch_incumbent[3]),
            ):
                branches_by_mask[signature] = state
        branch_pool = sorted(
            branches_by_mask.values(),
            key=lambda item: (-item[0], len(item[3]), item[3]),
        )[: cls.MDL_DNF_BRANCH_POOL]
        candidates: dict[bytes, MdlPredicateState] = {}
        for left, right in combinations(branch_pool, 2):
            total_literals = len(left[3]) + len(right[3])
            if total_literals > cls.MDL_DNF_MAX_TOTAL_LITERALS:
                continue
            mask = left[4] | right[4]
            support = int(mask.sum())
            if (
                support < cls.MIN_SUPPORT
                or len(mask) - support < cls.MIN_SUPPORT
                or np.array_equal(mask, left[4])
                or np.array_equal(mask, right[4])
            ):
                continue
            self.mdl_dnf_candidates_evaluated_ += 1
            savings = saved_bits(mask)
            branches = tuple(sorted((tuple(left[3]), tuple(right[3]))))
            description = (
                cls._mdl_dnf_description_bits(
                    (len(branches[0]), len(branches[1])),
                    source_binary_columns,
                )
                + residual_parameter_bits
            )
            net_bits = savings - description
            if net_bits < cls.MDL_MIN_NET_BITS:
                continue
            predicate = Predicate(
                "binary_dnf",
                tuple(column for branch in branches for column, _positive in branch),
                len(branches[0]),
                directions=tuple(positive for branch in branches for _column, positive in branch),
            )
            signature = self._canonical_partition_signature(mask)
            if signature in simpler_signatures:
                continue
            state = (net_bits, savings, description, predicate)
            candidate_incumbent = candidates.get(signature)
            if candidate_incumbent is None or (
                state[0],
                -len(state[3].columns),
                state[3].columns,
                state[3].directions,
            ) > (
                candidate_incumbent[0],
                -len(candidate_incumbent[3].columns),
                candidate_incumbent[3].columns,
                candidate_incumbent[3].directions,
            ):
                candidates[signature] = state
        self.mdl_candidates_evaluated_ += self.mdl_dnf_candidates_evaluated_
        return sorted(
            candidates.values(),
            key=lambda item: (
                -item[0],
                len(item[3].columns),
                item[3].columns,
                item[3].directions,
            ),
        )[: cls.MDL_DNF_MAX_PREDICATES]

    def _mdl_recursive_dnf_candidates(
        self,
        X,
        branch_states,
        output_signatures,
        selected_columns,
        source_binary_columns,
        residual_parameter_bits,
        saved_bits,
    ):
        """Grow promising two-branch programs by one verified OR branch.

        The search is recursive but finite: at most twelve canonical branch
        masks form a bounded pair beam, and only one additional branch may be
        attached. A three-branch program must encode a partition unavailable to
        every atom, conjunction, or pair before it can pay the MDL threshold.
        """
        cls = type(self)
        if not self.mdl_recursive_dnf or len(branch_states) < 3:
            return []

        simpler_signatures = set(output_signatures)
        simpler_signatures.update(self._canonical_partition_signature(state[4]) for state in branch_states)
        simpler_signatures.update(
            self._canonical_partition_signature(X[:, column] > 0.5) for column in selected_columns
        )
        branches_by_mask: dict[bytes, MdlBranchState] = {}
        for state in branch_states:
            signature = np.packbits(state[4].astype(np.uint8)).tobytes()
            branch_incumbent = branches_by_mask.get(signature)
            if branch_incumbent is None or (state[0], tuple(state[3])) > (
                branch_incumbent[0],
                tuple(branch_incumbent[3]),
            ):
                branches_by_mask[signature] = state
        branch_pool = sorted(
            branches_by_mask.values(),
            key=lambda item: (-item[0], len(item[3]), item[3]),
        )[: cls.MDL_DNF_BRANCH_POOL]
        if len(branch_pool) < 3:
            return []
        branch_masks_by_literals = {tuple(state[3]): state[4] for state in branch_pool}

        pair_states = []
        for left_index, right_index in combinations(range(len(branch_pool)), 2):
            left, right = branch_pool[left_index], branch_pool[right_index]
            mask = left[4] | right[4]
            support = int(mask.sum())
            if (
                support < cls.MIN_SUPPORT
                or len(mask) - support < cls.MIN_SUPPORT
                or np.array_equal(mask, left[4])
                or np.array_equal(mask, right[4])
            ):
                continue
            simpler_signatures.add(self._canonical_partition_signature(mask))
            branches = tuple(sorted((tuple(left[3]), tuple(right[3]))))
            if sum(len(literals) for literals in branches) + 1 > cls.MDL_RECURSIVE_DNF_MAX_TOTAL_LITERALS:
                continue
            description = (
                cls._mdl_recursive_dnf_description_bits(
                    (len(branches[0]), len(branches[1]), 1),
                    source_binary_columns,
                )
                + residual_parameter_bits
            )
            pair_states.append(
                (
                    saved_bits(mask) - description,
                    branches,
                    (left_index, right_index),
                    mask,
                )
            )
        pair_states.sort(key=lambda item: (-item[0], item[1]))
        pair_states = pair_states[: cls.MDL_RECURSIVE_DNF_PAIR_BEAM]

        candidates: dict[bytes, MdlPredicateState] = {}
        visited_programs = set()
        for _pair_net, pair_branches, pair_indices, pair_mask in pair_states:
            for branch_index, branch in enumerate(branch_pool):
                if branch_index in pair_indices:
                    continue
                branches = tuple(sorted((*pair_branches, tuple(branch[3]))))
                if branches in visited_programs:
                    continue
                visited_programs.add(branches)
                branch_widths = (
                    len(branches[0]),
                    len(branches[1]),
                    len(branches[2]),
                )
                if sum(branch_widths) > cls.MDL_RECURSIVE_DNF_MAX_TOTAL_LITERALS:
                    continue
                mask = pair_mask | branch[4]
                support = int(mask.sum())
                if (
                    support < cls.MIN_SUPPORT
                    or len(mask) - support < cls.MIN_SUPPORT
                    or np.array_equal(mask, pair_mask)
                ):
                    continue
                # Every branch must own evidence that the other two do not;
                # otherwise this is only a more expensive encoding of a pair.
                branch_masks = [branch_masks_by_literals[literals] for literals in branches]
                if any(
                    np.array_equal(
                        mask,
                        np.logical_or.reduce(
                            [other for other_index, other in enumerate(branch_masks) if other_index != index]
                        ),
                    )
                    for index in range(3)
                ):
                    continue

                self.mdl_recursive_dnf_candidates_evaluated_ += 1
                savings = saved_bits(mask)
                description = (
                    cls._mdl_recursive_dnf_description_bits(
                        branch_widths,
                        source_binary_columns,
                    )
                    + residual_parameter_bits
                )
                net_bits = savings - description
                if net_bits < cls.MDL_MIN_NET_BITS:
                    continue
                signature = self._canonical_partition_signature(mask)
                if signature in simpler_signatures:
                    continue
                predicate = Predicate(
                    "binary_recursive_dnf",
                    tuple(column for literals in branches for column, _positive in literals),
                    len(branches),
                    directions=tuple(positive for literals in branches for _column, positive in literals),
                    branch_widths=branch_widths,
                )
                state = (net_bits, savings, description, predicate)
                candidate_incumbent = candidates.get(signature)
                if candidate_incumbent is None or (
                    state[0],
                    -len(state[3].columns),
                    state[3].columns,
                    state[3].directions,
                ) > (
                    candidate_incumbent[0],
                    -len(candidate_incumbent[3].columns),
                    candidate_incumbent[3].columns,
                    candidate_incumbent[3].directions,
                ):
                    candidates[signature] = state

        self.mdl_candidates_evaluated_ += self.mdl_recursive_dnf_candidates_evaluated_
        return sorted(
            candidates.values(),
            key=lambda item: (
                -item[0],
                len(item[3].columns),
                item[3].columns,
                item[3].directions,
            ),
        )[: cls.MDL_RECURSIVE_DNF_MAX_PREDICATES]

    @staticmethod
    def _mdl_unique_state_pool(states, limit):
        """Keep the strongest canonical program for each exact mask."""
        by_mask: dict[bytes, MdlBranchState] = {}
        for state in states:
            signature = np.packbits(state[4].astype(np.uint8)).tobytes()
            incumbent = by_mask.get(signature)
            if incumbent is None or (state[0], tuple(state[3])) > (
                incumbent[0],
                tuple(incumbent[3]),
            ):
                by_mask[signature] = state
        return sorted(
            by_mask.values(),
            key=lambda item: (-item[0], len(item[3]), item[3]),
        )[:limit]

    def _mdl_local_exception_pool(
        self,
        base_literals,
        base_mask,
        atomic_conditions,
        source_binary_columns,
        residual_parameter_bits,
        saved_bits,
    ):
        """Compile exception clauses against counterexamples inside one base."""
        cls = type(self)
        beam = []
        for literals, condition in atomic_conditions.items():
            removed = base_mask & condition
            mask = base_mask & ~condition
            support = int(mask.sum())
            if (
                int(removed.sum()) < cls.MIN_SUPPORT
                or support < cls.MIN_SUPPORT
                or len(mask) - support < cls.MIN_SUPPORT
            ):
                continue
            branch_widths = (len(base_literals), 1)
            if sum(branch_widths) > cls.MDL_EXCEPTION_MAX_TOTAL_LITERALS:
                continue
            description = (
                cls._mdl_exception_description_bits(
                    branch_widths,
                    source_binary_columns,
                )
                + residual_parameter_bits
            )
            self.mdl_exception_clause_candidates_evaluated_ += 1
            beam.append((saved_bits(mask) - description, literals, condition))
        beam.sort(key=lambda item: (-item[0], item[1]))
        beam = beam[: cls.MDL_EXCEPTION_CLAUSE_BEAM]

        clauses = []
        for width in range(2, cls.MDL_EXCEPTION_MAX_CLAUSE_LITERALS + 1):
            by_removed_mask: dict[bytes, MdlClauseState] = {}
            for _parent_net, literals, parent_mask in beam:
                used_columns = {column for column, _positive in literals}
                for atomic_literals, condition in atomic_conditions.items():
                    column, _positive = atomic_literals[0]
                    if column in used_columns:
                        continue
                    clause_mask = parent_mask & condition
                    removed = base_mask & clause_mask
                    mask = base_mask & ~clause_mask
                    support = int(mask.sum())
                    if (
                        int(removed.sum()) < cls.MIN_SUPPORT
                        or support < cls.MIN_SUPPORT
                        or len(mask) - support < cls.MIN_SUPPORT
                    ):
                        continue
                    next_literals = tuple(sorted((*literals, atomic_literals[0])))
                    branch_widths = (len(base_literals), width)
                    if sum(branch_widths) > cls.MDL_EXCEPTION_MAX_TOTAL_LITERALS:
                        continue
                    description = (
                        cls._mdl_exception_description_bits(
                            branch_widths,
                            source_binary_columns,
                        )
                        + residual_parameter_bits
                    )
                    self.mdl_exception_clause_candidates_evaluated_ += 1
                    state = (
                        saved_bits(mask) - description,
                        next_literals,
                        clause_mask,
                    )
                    signature = np.packbits(removed.astype(np.uint8)).tobytes()
                    incumbent = by_removed_mask.get(signature)
                    if incumbent is None or (state[0], state[1]) > (
                        incumbent[0],
                        incumbent[1],
                    ):
                        by_removed_mask[signature] = state
            beam = sorted(
                by_removed_mask.values(),
                key=lambda item: (-item[0], item[1]),
            )[: cls.MDL_EXCEPTION_CLAUSE_BEAM]
            clauses.extend(beam)
        return sorted(
            clauses,
            key=lambda item: (-item[0], len(item[1]), item[1]),
        )[: cls.MDL_EXCEPTION_BRANCH_POOL]

    def _mdl_exception_candidates(
        self,
        atom_states,
        branch_states,
        output_signatures,
        source_binary_columns,
        residual_parameter_bits,
        saved_bits,
    ):
        """Search a bounded base clause with one or two subtracted clauses."""
        cls = type(self)
        if not self.mdl_exception or not atom_states or not branch_states:
            return []

        simpler_signatures = set(output_signatures)

        atom_bases = self._mdl_unique_state_pool(
            atom_states,
            cls.MDL_EXCEPTION_ATOM_BASES,
        )
        compound_bases = self._mdl_unique_state_pool(
            branch_states,
            cls.MDL_EXCEPTION_BASE_POOL - len(atom_bases),
        )
        base_pool = [*atom_bases, *compound_bases]
        simpler_signatures.update(
            self._canonical_partition_signature(state[4]) for state in base_pool
        )
        atomic_conditions = {
            tuple(state[3]): state[4] for state in atom_states if len(state[3]) == 1
        }
        candidates: dict[bytes, MdlPredicateState] = {}
        one_exception_states = []
        one_exception_signatures = set()

        def retain_candidate(signature, state):
            incumbent = candidates.get(signature)
            if incumbent is None or (
                state[0],
                -len(state[3].columns),
                state[3].columns,
                state[3].directions,
            ) > (
                incumbent[0],
                -len(incumbent[3].columns),
                incumbent[3].columns,
                incumbent[3].directions,
            ):
                candidates[signature] = state

        for base in base_pool:
            base_literals = tuple(base[3])
            base_mask = base[4]
            exception_pool = self._mdl_local_exception_pool(
                base_literals,
                base_mask,
                atomic_conditions,
                source_binary_columns,
                residual_parameter_bits,
                saved_bits,
            )
            for _exception_net, exception_literals, exception_mask in exception_pool:
                branch_widths: tuple[int, ...] = (
                    len(base_literals),
                    len(exception_literals),
                )
                mask = base_mask & ~exception_mask
                self.mdl_exception_candidates_evaluated_ += 1
                savings = saved_bits(mask)
                description = (
                    cls._mdl_exception_description_bits(
                        branch_widths,
                        source_binary_columns,
                    )
                    + residual_parameter_bits
                )
                net_bits = savings - description
                signature = self._canonical_partition_signature(mask)
                one_exception_signatures.add(signature)
                one_exception_states.append(
                    (
                        net_bits,
                        base_literals,
                        exception_literals,
                        base_mask,
                        mask,
                        exception_mask,
                        exception_pool,
                    )
                )
                if net_bits < cls.MDL_MIN_NET_BITS or signature in simpler_signatures:
                    continue
                branches = (base_literals, exception_literals)
                predicate = Predicate(
                    "binary_exception",
                    tuple(column for branch in branches for column, _positive in branch),
                    1,
                    directions=tuple(
                        positive for branch in branches for _column, positive in branch
                    ),
                    branch_widths=branch_widths,
                )
                retain_candidate(signature, (net_bits, savings, description, predicate))

        one_exception_states.sort(key=lambda item: (-item[0], item[1], item[2]))
        simpler_signatures.update(one_exception_signatures)
        visited_programs = set()
        for (
            _net,
            base_literals,
            first_exception,
            base_mask,
            current_mask,
            first_mask,
            exception_pool,
        ) in one_exception_states[: cls.MDL_EXCEPTION_PAIR_BEAM]:
            for _second_net, second_exception, second_mask in exception_pool:
                if second_exception == first_exception:
                    continue
                exceptions = tuple(sorted((first_exception, second_exception)))
                program = (base_literals, exceptions)
                if program in visited_programs:
                    continue
                visited_programs.add(program)
                branch_widths = (
                    len(base_literals),
                    len(exceptions[0]),
                    len(exceptions[1]),
                )
                if sum(branch_widths) > cls.MDL_EXCEPTION_MAX_TOTAL_LITERALS:
                    continue
                first_unique = base_mask & first_mask & ~second_mask
                second_unique = base_mask & second_mask & ~first_mask
                mask = current_mask & ~second_mask
                support = int(mask.sum())
                if (
                    int(first_unique.sum()) < cls.MIN_SUPPORT
                    or int(second_unique.sum()) < cls.MIN_SUPPORT
                    or support < cls.MIN_SUPPORT
                    or len(mask) - support < cls.MIN_SUPPORT
                ):
                    continue
                self.mdl_exception_candidates_evaluated_ += 1
                savings = saved_bits(mask)
                description = (
                    cls._mdl_exception_description_bits(
                        branch_widths,
                        source_binary_columns,
                    )
                    + residual_parameter_bits
                )
                net_bits = savings - description
                if net_bits < cls.MDL_MIN_NET_BITS:
                    continue
                signature = self._canonical_partition_signature(mask)
                if signature in simpler_signatures:
                    continue
                branches = (base_literals, *exceptions)
                predicate = Predicate(
                    "binary_exception",
                    tuple(column for branch in branches for column, _positive in branch),
                    2,
                    directions=tuple(
                        positive for branch in branches for _column, positive in branch
                    ),
                    branch_widths=branch_widths,
                )
                retain_candidate(signature, (net_bits, savings, description, predicate))

        self.mdl_candidates_evaluated_ += (
            self.mdl_exception_clause_candidates_evaluated_
            + self.mdl_exception_candidates_evaluated_
        )
        return sorted(
            candidates.values(),
            key=lambda item: (
                -item[0],
                len(item[3].columns),
                item[3].columns,
                item[3].directions,
            ),
        )[: cls.MDL_EXCEPTION_MAX_PREDICATES]

    def _mdl_beam_candidates(
        self,
        X,
        y,
        existing_predicates,
        sample_weight=None,
        baseline_probability=None,
    ):
        """Search signed Boolean conjunctions by target codelength reduction.

        Search sees only the mapper's fit rows. The outer TabPVN gate remains
        responsible for admission on untouched folds.
        """
        cls = type(self)
        self.mdl_candidates_evaluated_ = 0
        self.mdl_predicates_selected_ = 0
        self.mdl_dnf_candidates_evaluated_ = 0
        self.mdl_dnf_predicates_selected_ = 0
        self.mdl_recursive_dnf_candidates_evaluated_ = 0
        self.mdl_recursive_dnf_predicates_selected_ = 0
        self.mdl_exception_candidates_evaluated_ = 0
        self.mdl_exception_clause_candidates_evaluated_ = 0
        self.mdl_exception_predicates_selected_ = 0
        self.mdl_atom_columns_ = ()
        self.mdl_program_report_ = []
        binary = np.asarray(getattr(self, "_screened_binary_columns_", ()), dtype=int)
        if not self.mdl_beam or self.rare_rules or len(binary) < 3:
            return []

        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            return []
        y01 = y == classes[1]
        weights = (
            np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        if len(weights) != len(y):
            raise ValueError("sample_weight must have one value per predicate row")
        objective, residual_parameter_bits, saved_bits = cls._mdl_codelength_scorer(
            y01,
            weights,
            baseline_probability,
        )
        self.mdl_search_objective_ = objective

        atomic_scores = sorted(
            ((saved_bits(X[:, column] > 0.5), int(column)) for column in binary),
            key=lambda item: (-item[0], item[1]),
        )
        pool_size = min(int(cls.MDL_ATOM_POOL), len(atomic_scores))
        signal_count = min(pool_size, max(1, 3 * pool_size // 4))
        selected_columns = [column for _score, column in atomic_scores[:signal_count]]
        remaining = np.asarray([column for _score, column in atomic_scores[signal_count:]], dtype=int)
        if len(selected_columns) < pool_size and len(remaining):
            exploration = np.random.default_rng(self.seed + 1709).permutation(remaining)
            selected_columns.extend(
                int(column) for column in exploration[: pool_size - len(selected_columns)]
            )
        selected_columns = sorted(selected_columns)
        self.mdl_atom_columns_ = tuple(selected_columns)
        source_binary_columns = int(getattr(self, "source_binary_columns_", len(binary)))

        existing_signatures = {
            self._canonical_partition_signature(self._predicate_values(X, predicate))
            for predicate in existing_predicates
        }
        output_signatures = set(existing_signatures)
        outputs: list[MdlPredicateState] = []
        branch_states: list[MdlBranchState] = []
        beam: list[MdlBranchState] = []
        for column in selected_columns:
            for positive_literal in (False, True):
                mask = X[:, column] > 0.5 if positive_literal else X[:, column] <= 0.5
                if mask.sum() < cls.MIN_SUPPORT or len(mask) - mask.sum() < cls.MIN_SUPPORT:
                    continue
                savings = saved_bits(mask)
                description = cls._mdl_description_bits(1, source_binary_columns) + residual_parameter_bits
                beam.append(
                    (savings - description, savings, description, ((column, positive_literal),), mask)
                )
        self.mdl_candidates_evaluated_ += len(beam)
        beam.sort(key=lambda item: (-item[0], item[3]))
        beam = beam[: cls.MDL_BEAM_WIDTH]
        atom_states = list(beam)

        for width in range(2, int(cls.MDL_MAX_LITERALS) + 1):
            candidates_by_mask: dict[bytes, MdlBranchState] = {}
            for _parent_net, _parent_saved, _parent_description, literals, parent_mask in beam:
                used_columns = {column for column, _positive in literals}
                for column in selected_columns:
                    if column in used_columns:
                        continue
                    for positive_literal in (False, True):
                        condition = X[:, column] > 0.5 if positive_literal else X[:, column] <= 0.5
                        mask = parent_mask & condition
                        support = int(mask.sum())
                        if support < cls.MIN_SUPPORT or len(mask) - support < cls.MIN_SUPPORT:
                            continue
                        next_literals = tuple(sorted(literals + ((column, positive_literal),)))
                        savings = saved_bits(mask)
                        description = (
                            cls._mdl_description_bits(width, source_binary_columns) + residual_parameter_bits
                        )
                        state = (
                            savings - description,
                            savings,
                            description,
                            next_literals,
                            mask,
                        )
                        signature = np.packbits(mask.astype(np.uint8)).tobytes()
                        incumbent = candidates_by_mask.get(signature)
                        if incumbent is None or (state[0], tuple(state[3])) > (
                            incumbent[0],
                            tuple(incumbent[3]),
                        ):
                            candidates_by_mask[signature] = state
            candidates = list(candidates_by_mask.values())
            self.mdl_candidates_evaluated_ += len(candidates)
            candidates.sort(key=lambda item: (-item[0], item[3]))
            beam = candidates[: cls.MDL_BEAM_WIDTH]
            branch_states.extend(beam)
            for net_bits, savings, description, literals, mask in beam:
                if net_bits < cls.MDL_MIN_NET_BITS:
                    continue
                signature = self._canonical_partition_signature(mask)
                if signature in output_signatures:
                    continue
                output_signatures.add(signature)
                predicate = Predicate(
                    "binary_and",
                    tuple(column for column, _positive in literals),
                    1,
                    directions=tuple(positive for _column, positive in literals),
                )
                outputs.append((net_bits, savings, description, predicate))

        dnf_outputs = self._mdl_dnf_candidates(
            X,
            branch_states,
            output_signatures,
            selected_columns,
            source_binary_columns,
            residual_parameter_bits,
            saved_bits,
        )
        recursive_signatures = set(output_signatures)
        recursive_signatures.update(
            self._canonical_partition_signature(self._predicate_values(X, state[3]))
            for state in dnf_outputs
        )
        recursive_dnf_outputs = self._mdl_recursive_dnf_candidates(
            X,
            branch_states,
            recursive_signatures,
            selected_columns,
            source_binary_columns,
            residual_parameter_bits,
            saved_bits,
        )
        exception_signatures = set(recursive_signatures)
        exception_signatures.update(
            self._canonical_partition_signature(self._predicate_values(X, state[3]))
            for state in recursive_dnf_outputs
        )
        exception_outputs = self._mdl_exception_candidates(
            atom_states,
            branch_states,
            exception_signatures,
            source_binary_columns,
            residual_parameter_bits,
            saved_bits,
        )

        outputs.sort(key=lambda item: (-item[0], len(item[3].columns), item[3].columns, item[3].directions))
        # Composed programs have separate augmentation budgets so they cannot
        # displace simpler conjunctions before the untouched-fold gates.
        selected = (
            outputs[: cls.MDL_MAX_PREDICATES]
            + dnf_outputs
            + recursive_dnf_outputs
            + exception_outputs
        )
        self.mdl_predicates_selected_ = len(selected)
        self.mdl_dnf_predicates_selected_ = sum(
            predicate.kind == "binary_dnf" for _net, _saved, _description, predicate in selected
        )
        self.mdl_recursive_dnf_predicates_selected_ = sum(
            predicate.kind == "binary_recursive_dnf"
            for _net, _saved, _description, predicate in selected
        )
        self.mdl_exception_predicates_selected_ = sum(
            predicate.kind == "binary_exception"
            for _net, _saved, _description, predicate in selected
        )
        self.mdl_program_report_ = [
            {
                "predicate": predicate,
                "program_kind": predicate.kind,
                "branch_count": (
                    len(predicate.branch_widths)
                    if predicate.kind in {"binary_recursive_dnf", "binary_exception"}
                    else 2
                    if predicate.kind == "binary_dnf"
                    else 1
                ),
                "exception_count": (
                    int(predicate.value) if predicate.kind == "binary_exception" else 0
                ),
                "literal_count": int(len(predicate.columns)),
                "codelength_saved_bits": float(savings),
                "description_bits": float(description),
                "net_mdl_bits": float(net_bits),
            }
            for net_bits, savings, description, predicate in selected
        ]
        return [predicate for _net_bits, _savings, _description, predicate in selected]

    def _binary_candidates(
        self,
        X,
        y,
        sample_weight=None,
        residual=None,
        hessian=None,
    ):
        cls = type(self)
        binary = self._screen_binary_columns(
            X,
            y,
            cls._binary_columns(X),
            sample_weight=sample_weight,
            residual=residual,
            hessian=hessian,
        )
        self._screened_binary_columns_ = tuple(int(column) for column in binary)
        if len(binary) < 4:
            return []
        y01 = np.asarray(y) == np.unique(y)[1]
        B = X[:, binary].astype(bool)
        pairs = []
        if residual is None:
            information = cls._pair_information(B, y01)
            bit_score = information.clip(min=0).sum(1)
            pair_indices = range(len(binary))
        else:
            bit_score = np.array(
                [
                    cls._score(
                        B[:, index],
                        y01,
                        sample_weight,
                        residual,
                        hessian,
                    )
                    for index in range(len(binary))
                ]
            )
            pair_indices = np.argsort(-bit_score)[: min(16, len(binary))]
            information = None
        for i, j in combinations(pair_indices, 2):
            if information is None:
                states = 2 * B[:, i].astype(np.int8) + B[:, j].astype(np.int8)
                pair_score = max(
                    cls._score(
                        states == state,
                        y01,
                        sample_weight,
                        residual,
                        hessian,
                    )
                    for state in range(4)
                )
            else:
                pair_score = float(information[i, j])
            pairs.append((float(pair_score), int(i), int(j)))
        pairs.sort(reverse=True)
        pairs = pairs[: cls.PAIR_POOL]

        scored: list[tuple[float, Predicate]] = []
        seen = set()

        for _info, i, j in pairs:
            a, b = B[:, i], B[:, j]
            # ``state`` uses 2*a+b: 0=00, 1=01, 2=10, 3=11.
            states = 2 * a.astype(np.int8) + b.astype(np.int8)
            # A depth-three tree needs multiple regions for OR/parity even
            # though both are finite facts. Compiling the pair projections
            # gives the certified booster one readable split when the target
            # really follows one of these structures. Same-category one-hot
            # pairs are excluded: OR aliases the retained 00 state and XOR is
            # identical to OR because their 11 state is impossible.
            if not self._same_exclusive_group((binary[i], binary[j])):
                for kind, mask in (
                    ("or", states != 0),
                    ("xor", (states == 1) | (states == 2)),
                ):
                    if mask.sum() < cls.MIN_SUPPORT:
                        continue
                    pred = Predicate(kind, (int(binary[i]), int(binary[j])), 1)
                    score = cls._score(mask, y01, sample_weight, residual, hessian)
                    minimum = cls.MIN_ASSOCIATION if residual is None else cls.RESIDUAL_MIN_GAIN
                    if score >= minimum and pred not in seen:
                        seen.add(pred)
                        scored.append((score, pred))
            for state in range(4):
                # For two levels of the same one-hot categorical, 11 is
                # impossible and 01/10 are aliases of a single original fact.
                # 00 is useful: it represents membership in the remaining
                # category levels, a compact fact a tree cannot express in one
                # split.
                if self._same_exclusive_group((binary[i], binary[j])) and state != 0:
                    continue
                mask = states == state
                if mask.sum() < cls.MIN_SUPPORT:
                    continue
                pred = Predicate("state", (int(binary[i]), int(binary[j])), state)
                score = cls._score(mask, y01, sample_weight, residual, hessian)
                minimum = cls.MIN_ASSOCIATION if residual is None else cls.RESIDUAL_MIN_GAIN
                if score >= minimum:
                    seen.add(pred)
                    scored.append((score, pred))

        # One level of composition: cardinality over the highest-information bits
        # catches exact-k structure while retaining a compact, readable program.
        # Groups through six facts cover the common "at least/exactly k flags"
        # patterns that otherwise require many separate tree regions.
        top_bits = np.argsort(-bit_score)[: min(cls.MAX_CARDINALITY_GROUP, len(binary))]
        for width in range(3, len(top_bits) + 1):
            for group in combinations(top_bits, width):
                cols = tuple(int(binary[i]) for i in group)
                counts = B[:, group].sum(1)
                for value in range(width + 1):
                    mask = counts == value
                    if mask.sum() < cls.MIN_SUPPORT:
                        continue
                    pred = Predicate("count", cols, value)
                    score = cls._score(mask, y01, sample_weight, residual, hessian)
                    minimum = cls.MIN_ASSOCIATION if residual is None else cls.RESIDUAL_MIN_GAIN
                    if score >= minimum and pred not in seen:
                        seen.add(pred)
                        scored.append((score, pred))
        scored.sort(key=lambda item: (-item[0], item[1].kind, item[1].columns, item[1].value))
        return [predicate for _score, predicate in scored[: cls.MAX_PREDICATES]]

    def _numeric_candidates(self, X, y, sample_weight=None):
        """Compose minority-class-enriched threshold literals into finite facts."""
        cls = type(self)
        numeric = cls._numeric_columns(X)
        if not (
            2 <= len(numeric) <= cls.MAX_BINARY_COLUMNS and len(y) * len(numeric) <= cls.MAX_PAIR_OPERATIONS
        ):
            return []
        classes, counts = np.unique(y, return_counts=True)
        y01 = np.asarray(y) == classes[int(np.argmin(counts))]
        literals = []
        for column in numeric:
            values = X[:, int(column)]
            best = None
            unique = np.unique(values)
            thresholds = (
                0.5 * (unique[:-1] + unique[1:])
                if len(unique) - 1 <= cls.MAX_EXACT_THRESHOLDS
                else np.unique(np.quantile(values, cls.THRESHOLD_QUANTILES))
            )
            for threshold in thresholds:
                lower = values <= threshold
                support = int(lower.sum())
                if support < cls.MIN_SUPPORT or len(values) - support < cls.MIN_SUPPORT:
                    continue
                lower_rate = float(y01[lower].mean())
                upper_rate = float(y01[~lower].mean())
                direction = lower_rate >= upper_rate
                mask = lower if direction else ~lower
                score = cls._association(mask, y01, sample_weight)
                candidate = (score, int(column), float(threshold), bool(direction), mask)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
            if best is not None and best[0] >= cls.MIN_ASSOCIATION:
                literals.append(best)
        literals.sort(key=lambda item: (-item[0], item[1], item[2], not item[3]))
        literals = literals[: cls.THRESHOLD_LITERAL_POOL]

        scored = []
        seen_masks = set()
        for left, right in combinations(literals, 2):
            if left[1] == right[1]:
                continue
            for kind, mask in (
                ("threshold_and", left[4] & right[4]),
                ("threshold_or", left[4] | right[4]),
            ):
                support = int(mask.sum())
                if support < cls.MIN_SUPPORT or len(mask) - support < cls.MIN_SUPPORT:
                    continue
                score = cls._association(mask, y01, sample_weight)
                if score < max(left[0], right[0]) + cls.MIN_COMPOSITION_GAIN:
                    continue
                signature = np.packbits(mask).tobytes()
                if signature in seen_masks:
                    continue
                seen_masks.add(signature)
                predicate = Predicate(
                    kind,
                    (left[1], right[1]),
                    1,
                    (left[2], right[2]),
                    (left[3], right[3]),
                )
                scored.append((score, predicate, mask))

        # A bounded second composition level captures narrow risk rules such as
        # ``anchor AND condition_a AND condition_b``. Only conjunctions whose
        # association improves over both the accepted parent and added literal
        # survive, so width does not grow merely because support shrinks.
        pair_rules = sorted(
            scored,
            key=lambda item: (
                -item[0],
                item[1].kind,
                item[1].columns,
                item[1].thresholds,
                item[1].directions,
            ),
        )
        selected_pairs = pair_rules[: cls.MAX_THRESHOLD_PREDICATES]
        triples = []
        for pair_score, pair, pair_mask in selected_pairs:
            if pair.kind != "threshold_and":
                continue
            for literal in literals:
                if literal[1] in pair.columns:
                    continue
                mask = pair_mask & literal[4]
                support = int(mask.sum())
                if support < cls.MIN_SUPPORT or len(mask) - support < cls.MIN_SUPPORT:
                    continue
                score = cls._association(mask, y01, sample_weight)
                if score < max(pair_score, literal[0]) + cls.MIN_COMPOSITION_GAIN:
                    continue
                signature = np.packbits(mask).tobytes()
                if signature in seen_masks:
                    continue
                seen_masks.add(signature)
                triples.append(
                    (
                        score,
                        Predicate(
                            "threshold_and",
                            pair.columns + (literal[1],),
                            1,
                            pair.thresholds + (literal[2],),
                            pair.directions + (literal[3],),
                        ),
                        mask,
                    )
                )
        triples.sort(
            key=lambda item: (
                -item[0],
                item[1].columns,
                item[1].thresholds,
                item[1].directions,
            )
        )
        scored = selected_pairs + triples[: cls.MAX_THRESHOLD_TRIPLES]
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].kind,
                item[1].columns,
                item[1].thresholds,
                item[1].directions,
            )
        )
        return [predicate for _score, predicate, _mask in scored]

    @staticmethod
    def _weighted_rate(mask, target, sample_weight):
        weights = np.asarray(sample_weight, dtype=float)
        mass = float(weights[mask].sum())
        return 0.0 if mass <= 0.0 else float(np.dot(weights[mask], target[mask]) / mass)

    @classmethod
    def _rare_threshold_grid(cls, values, event_mask):
        unique = np.unique(values)
        if len(unique) - 1 <= cls.MAX_EXACT_THRESHOLDS:
            return 0.5 * (unique[:-1] + unique[1:])
        pooled = np.quantile(values, cls.RARE_QUANTILES)
        event_values = values[event_mask]
        event = (
            np.quantile(event_values, cls.RARE_CLASS_QUANTILES)
            if len(event_values) >= cls.RARE_MIN_EVENTS
            else np.empty(0, dtype=float)
        )
        proposed = np.unique(np.concatenate([np.asarray(pooled, float), np.asarray(event, float)]))
        # Quantiles on discrete-valued columns often land on an observed value.
        # A tree boundary belongs between values, so retain the adjacent
        # midpoints on both sides (for example 94.5 as well as 95.5 around 95).
        cuts = []
        for value in proposed:
            insertion = int(np.searchsorted(unique, value, side="left"))
            for cut in (insertion - 1, insertion):
                if 0 <= cut < len(unique) - 1:
                    cuts.append(0.5 * (unique[cut] + unique[cut + 1]))
        thresholds = np.unique(np.asarray(cuts, dtype=float))
        if len(thresholds) > cls.RARE_THRESHOLD_MAX:
            keep = np.unique(np.linspace(0, len(thresholds) - 1, cls.RARE_THRESHOLD_MAX).astype(int))
            thresholds = thresholds[keep]
        return thresholds

    def _rare_numeric_candidates(  # noqa: C901 - finite candidate enumeration
        self,
        X,
        y,
        sample_weight=None,
        residual=None,
        hessian=None,
    ):
        """Compile bounded minority-tail, interval, and interaction facts."""
        cls = type(self)
        numeric = cls._numeric_columns(X)
        if not (
            2 <= len(numeric) <= cls.MAX_BINARY_COLUMNS and len(y) * len(numeric) <= cls.MAX_PAIR_OPERATIONS
        ):
            return []
        classes = np.unique(y)
        weights = (
            np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        if self.rare_class is None:
            class_mass = np.array([weights[np.asarray(y) == value].sum() for value in classes])
            rare_class = classes[int(np.argmin(class_mass))]
        else:
            rare_class = self.rare_class
        if rare_class not in classes:
            return []
        event = np.asarray(y) == rare_class
        residual_mode = residual is not None
        residual_column_pool = getattr(cls, "RESIDUAL_NUMERIC_COLUMN_POOL", None)
        if residual_mode and residual_column_pool is not None and len(numeric) > residual_column_pool:
            screened = []
            for column in numeric:
                values = X[:, int(column)]
                thresholds = np.unique(np.quantile(values, (0.1, 0.25, 0.5, 0.75, 0.9)))
                best = max(
                    (
                        cls._score(
                            values <= threshold,
                            event,
                            weights,
                            residual,
                            hessian,
                        )
                        for threshold in thresholds
                    ),
                    default=0.0,
                )
                screened.append((float(best), int(column)))
            screened.sort(key=lambda item: (-item[0], item[1]))
            numeric = np.asarray(
                sorted(column for _score, column in screened[:residual_column_pool]),
                dtype=int,
            )
        minimum_score = cls.RESIDUAL_MIN_GAIN if residual_mode else cls.RARE_MIN_ASSOCIATION
        composition_gain = cls.RESIDUAL_COMPOSITION_GAIN if residual_mode else cls.MIN_COMPOSITION_GAIN
        self.interval_candidates_considered_ = 0
        self.interval_predicates_selected_ = 0
        self.interval_union_candidates_ = 0
        self.interval_union_predicates_selected_ = 0
        self.residual_interval_predicates_ = 0
        self.multi_interval_columns_ = 0

        def score(mask):
            return cls._score(mask, event, weights, residual, hessian)

        def eligible(mask, candidate_score=None):
            support = int(mask.sum())
            if support < cls.MIN_SUPPORT or len(mask) - support < cls.MIN_SUPPORT:
                return False
            if residual_mode:
                return (score(mask) if candidate_score is None else candidate_score) >= minimum_score
            return int(event[mask].sum()) >= cls.RARE_MIN_EVENTS and cls._weighted_rate(
                mask, event, weights
            ) > cls._weighted_rate(~mask, event, weights)

        def evaluated_score(mask, known_score=None):
            if residual_mode:
                value = score(mask) if known_score is None else known_score
                return value if eligible(mask, value) else None
            if not eligible(mask):
                return None
            return score(mask) if known_score is None else known_score

        literals = []
        interval_scored = []
        interval_union_scored = []
        seen_masks = set()
        for column in numeric:
            column = int(column)
            values = X[:, column]
            thresholds = cls._rare_threshold_grid(values, event)
            residual_threshold_max = getattr(cls, "RESIDUAL_THRESHOLD_MAX", None)
            if (
                residual_mode
                and residual_threshold_max is not None
                and len(thresholds) > residual_threshold_max
            ):
                keep = np.unique(
                    np.linspace(
                        0,
                        len(thresholds) - 1,
                        residual_threshold_max,
                    ).astype(int)
                )
                thresholds = thresholds[keep]
            column_literals = []
            for threshold in thresholds:
                lower = values <= threshold
                support = int(lower.sum())
                if support < cls.MIN_SUPPORT or len(values) - support < cls.MIN_SUPPORT:
                    continue
                if residual_mode:
                    lower_score = score(lower)
                    upper_score = score(~lower)
                    direction = lower_score >= upper_score
                else:
                    lower_rate = cls._weighted_rate(lower, event, weights)
                    upper_rate = cls._weighted_rate(~lower, event, weights)
                    direction = lower_rate >= upper_rate
                mask = lower if direction else ~lower
                candidate_score = evaluated_score(mask)
                if candidate_score is None:
                    continue
                signature = np.packbits(mask).tobytes()
                column_literals.append(
                    (
                        candidate_score,
                        column,
                        float(threshold),
                        bool(direction),
                        mask,
                        signature,
                    )
                )
            column_literals.sort(key=lambda item: (-item[0], item[2], not item[3]))
            deduplicated = []
            column_seen = set()
            for column_literal in column_literals:
                if column_literal[5] in column_seen:
                    continue
                column_seen.add(column_literal[5])
                deduplicated.append(column_literal[:5])
                if len(deduplicated) >= cls.RARE_LITERALS_PER_COLUMN:
                    break
            literals.extend(deduplicated)

            best_literal = deduplicated[0][0] if deduplicated else 0.0
            column_intervals = []
            for lower, upper in combinations(thresholds, 2):
                mask = (values > lower) & (values <= upper)
                support = int(mask.sum())
                if support < cls.MIN_SUPPORT or len(mask) - support < cls.MIN_SUPPORT:
                    continue
                candidate_score = evaluated_score(mask)
                if candidate_score is None:
                    continue
                if candidate_score < max(minimum_score, best_literal + composition_gain):
                    continue
                column_intervals.append(
                    (
                        candidate_score,
                        NumericInterval(float(lower), float(upper)),
                    )
                )

            self.interval_candidates_considered_ += len(column_intervals)
            column_intervals.sort(key=lambda item: (-item[0], item[1].lower, item[1].upper))
            column_intervals = column_intervals[: cls.RARE_INTERVAL_POOL_PER_COLUMN]
            selected_domains: list[NumericInterval] = []
            selected_column_intervals = []
            # Keep the incumbent strongest interval. For one additional slot,
            # subtract already-selected 1D regions exactly and rank only the
            # remaining pieces. Existing predicates are never deduplicated or
            # pruned by this operation.
            while len(selected_domains) < cls.RARE_INTERVALS_PER_COLUMN:
                residual_candidates = []
                residual_seen = set()
                for raw_score, domain in column_intervals:
                    pieces: tuple[NumericInterval, ...] = (domain,)
                    for selected_domain in selected_domains:
                        pieces = tuple(
                            residual_piece
                            for piece in pieces
                            for residual_piece in piece.difference(selected_domain)
                        )
                    for piece in pieces:
                        if piece in residual_seen:
                            continue
                        residual_seen.add(piece)
                        piece_mask = (values > piece.lower) & (values <= piece.upper)
                        piece_score = evaluated_score(
                            piece_mask,
                            known_score=raw_score if piece == domain else None,
                        )
                        if piece_score is None:
                            continue
                        if piece_score < max(minimum_score, best_literal + composition_gain):
                            continue
                        signature = np.packbits(piece_mask).tobytes()
                        if signature in seen_masks:
                            continue
                        residual_candidates.append((piece_score, piece, piece_mask, signature))
                if not residual_candidates:
                    break
                residual_candidates.sort(key=lambda item: (-item[0], item[1].lower, item[1].upper))
                selected_score, selected_domain, selected_mask, signature = residual_candidates[0]
                predicate = Predicate(
                    "threshold_interval",
                    (column, column),
                    1,
                    (selected_domain.lower, selected_domain.upper),
                    (False, True),
                )
                seen_masks.add(signature)
                selected_domains.append(selected_domain)
                selected_column_intervals.append((selected_score, selected_domain, selected_mask))
                interval_scored.append((selected_score, predicate, selected_mask))

            if (
                residual_mode
                and cls.MAX_RARE_INTERVAL_UNIONS > 0
                and len(selected_column_intervals) == cls.RARE_INTERVALS_PER_COLUMN
            ):
                union_domain = NumericIntervalUnion.canonical(
                    [domain for _score, domain, _mask in selected_column_intervals]
                )
                if len(union_domain.intervals) == cls.RARE_INTERVALS_PER_COLUMN:
                    union_mask = np.logical_or.reduce(
                        [mask for _score, _domain, mask in selected_column_intervals]
                    )
                    union_score = evaluated_score(union_mask)
                    strongest_child = max(
                        child_score for child_score, _domain, _mask in selected_column_intervals
                    )
                    signature = np.packbits(union_mask).tobytes()
                    if (
                        union_score is not None
                        and union_score >= strongest_child + composition_gain
                        and signature not in seen_masks
                    ):
                        thresholds = tuple(
                            threshold
                            for interval in union_domain.intervals
                            for threshold in (interval.lower, interval.upper)
                        )
                        directions = tuple(
                            direction for _interval in union_domain.intervals for direction in (False, True)
                        )
                        seen_masks.add(signature)
                        interval_union_scored.append(
                            (
                                union_score,
                                Predicate(
                                    "threshold_union",
                                    (column,) * len(thresholds),
                                    1,
                                    thresholds,
                                    directions,
                                ),
                                union_mask,
                            )
                        )

        literals.sort(key=lambda item: (-item[0], item[1], item[2], not item[3]))
        literals = literals[: cls.RARE_LITERAL_POOL]
        pair_scored = []
        for left, right in combinations(literals, 2):
            if left[1] == right[1]:
                continue
            for kind, mask in (
                ("threshold_and", left[4] & right[4]),
                ("threshold_or", left[4] | right[4]),
            ):
                support = int(mask.sum())
                if support < cls.MIN_SUPPORT or len(mask) - support < cls.MIN_SUPPORT:
                    continue
                candidate_score = evaluated_score(mask)
                if candidate_score is None:
                    continue
                if candidate_score < max(left[0], right[0]) + composition_gain:
                    continue
                signature = np.packbits(mask).tobytes()
                if signature in seen_masks:
                    continue
                seen_masks.add(signature)
                pair_scored.append(
                    (
                        candidate_score,
                        Predicate(
                            kind,
                            (left[1], right[1]),
                            1,
                            (left[2], right[2]),
                            (left[3], right[3]),
                        ),
                        mask,
                    )
                )
        pair_scored.sort(
            key=lambda item: (
                -item[0],
                item[1].kind,
                item[1].columns,
                item[1].thresholds,
                item[1].directions,
            )
        )
        selected_pairs = pair_scored[: cls.MAX_RARE_PAIRS]
        triples = []
        for pair_score, pair, pair_mask in selected_pairs:
            if pair.kind != "threshold_and":
                continue
            for added_literal in literals:
                if added_literal[1] in pair.columns:
                    continue
                mask = pair_mask & added_literal[4]
                support = int(mask.sum())
                if support < cls.MIN_SUPPORT or len(mask) - support < cls.MIN_SUPPORT:
                    continue
                candidate_score = evaluated_score(mask)
                if candidate_score is None:
                    continue
                if candidate_score < max(pair_score, added_literal[0]) + composition_gain:
                    continue
                signature = np.packbits(mask).tobytes()
                if signature in seen_masks:
                    continue
                seen_masks.add(signature)
                triples.append(
                    (
                        candidate_score,
                        Predicate(
                            "threshold_and",
                            pair.columns + (added_literal[1],),
                            1,
                            pair.thresholds + (added_literal[2],),
                            pair.directions + (added_literal[3],),
                        ),
                        mask,
                    )
                )
        triples.sort(
            key=lambda item: (
                -item[0],
                item[1].columns,
                item[1].thresholds,
                item[1].directions,
            )
        )
        interval_scored.sort(key=lambda item: (-item[0], item[1].columns, item[1].thresholds))
        interval_union_scored.sort(key=lambda item: (-item[0], item[1].columns, item[1].thresholds))
        selected_intervals = interval_scored[: cls.MAX_RARE_INTERVALS]
        selected_interval_unions = interval_union_scored[: cls.MAX_RARE_INTERVAL_UNIONS]
        interval_columns: dict[int, int] = {}
        for _score, predicate, _mask in selected_intervals:
            column = predicate.columns[0]
            interval_columns[column] = interval_columns.get(column, 0) + 1
        self.interval_predicates_selected_ = len(selected_intervals)
        self.interval_union_candidates_ = len(interval_union_scored)
        self.interval_union_predicates_selected_ = len(selected_interval_unions)
        self.residual_interval_predicates_ = sum(max(0, count - 1) for count in interval_columns.values())
        self.multi_interval_columns_ = sum(count > 1 for count in interval_columns.values())
        selected = (
            selected_intervals + selected_interval_unions + selected_pairs + triples[: cls.MAX_RARE_TRIPLES]
        )
        selected.sort(
            key=lambda item: (
                -item[0],
                item[1].kind,
                item[1].columns,
                item[1].thresholds,
                item[1].directions,
            )
        )
        return [predicate for _score, predicate, _mask in selected]

    def _candidates(
        self,
        X,
        y,
        sample_weight=None,
        residual=None,
        hessian=None,
        mdl_baseline_probability=None,
        mdl_evidence_rows=None,
    ):
        predicates = self._binary_candidates(
            X,
            y,
            sample_weight=sample_weight,
            residual=residual,
            hessian=hessian,
        )
        if residual is None and self.mdl_beam:
            evidence_rows = self._bounded_mdl_evidence_rows(y, mdl_evidence_rows)
            evidence_weight = (
                None if sample_weight is None else np.asarray(sample_weight, dtype=float)[evidence_rows]
            )
            evidence_probability = (
                None
                if mdl_baseline_probability is None
                else np.asarray(mdl_baseline_probability, dtype=float)[evidence_rows]
            )
            self.mdl_evidence_rows_ = int(len(evidence_rows))
            predicates.extend(
                self._mdl_beam_candidates(
                    X[evidence_rows],
                    y[evidence_rows],
                    predicates,
                    sample_weight=evidence_weight,
                    baseline_probability=evidence_probability,
                )
            )
        if self.rare_rules:
            predicates.extend(
                self._rare_numeric_candidates(
                    X,
                    y,
                    sample_weight=sample_weight,
                    residual=residual,
                    hessian=hessian,
                )
            )
        elif self.numeric_rules:
            predicates.extend(self._numeric_candidates(X, y, sample_weight=sample_weight))
        return predicates

    @staticmethod
    def _predicate_family(predicate: Predicate) -> str:
        if predicate.kind in {"threshold_interval", "threshold_union"}:
            return "interval"
        if predicate.kind.startswith("threshold_"):
            return "numeric_composition"
        return "boolean"

    @classmethod
    def _family_counts(cls, predicates: Sequence[Predicate]) -> dict[str, int]:
        counts = {"boolean": 0, "interval": 0, "numeric_composition": 0}
        for predicate in predicates:
            counts[cls._predicate_family(predicate)] += 1
        return counts

    @staticmethod
    def _interval_branch_count(predicate: Predicate) -> int:
        if predicate.kind == "threshold_interval":
            return 1
        if predicate.kind == "threshold_union":
            return len(predicate.thresholds) // 2
        return 0

    def _select_residual_candidates(self, predicates: Sequence[Predicate]) -> list[Predicate]:
        """Bound mixed-schema capacity without disturbing ordinary residual maps.

        Candidate generators already rank within each family by residual gain.
        The incumbent concatenation gives Boolean programs the first 16 slots,
        leaving four of 20 for every numeric program. Reserve two more numeric
        slots only when that family's top six candidates encode at least five
        interval branches, direct evidence of a truncated multimodal
        representation. A bounded interval union counts its explicit branches.
        """
        predicates = list(predicates)
        budget = int(self.MAX_RESIDUAL_PREDICATES)
        self.residual_candidate_family_counts_ = self._family_counts(predicates)
        self.residual_allocator_ = "ordered_prefix"
        if len(predicates) <= budget:
            selected = predicates
        else:
            structural = [
                (index, predicate)
                for index, predicate in enumerate(predicates)
                if not predicate.kind.startswith("threshold_")
            ]
            numeric = [
                (index, predicate)
                for index, predicate in enumerate(predicates)
                if predicate.kind.startswith("threshold_")
            ]
            reserve = min(int(self.RESIDUAL_NUMERIC_RESERVE), budget)
            numeric_prefix = numeric[:reserve]
            interval_evidence = sum(
                self._interval_branch_count(predicate) for _index, predicate in numeric_prefix
            )
            use_reserve = bool(
                len(structural) > budget - reserve
                and len(numeric_prefix) == reserve
                and interval_evidence >= int(self.RESIDUAL_INTERVAL_TRIGGER)
            )
            if not use_reserve:
                selected = predicates[:budget]
            else:
                selected_indices = [index for index, _predicate in structural[: budget - reserve]]
                selected_indices += [index for index, _predicate in numeric_prefix]
                selected_set = set(selected_indices)
                if len(selected_indices) < budget:
                    selected_indices.extend(
                        index for index in range(len(predicates)) if index not in selected_set
                    )
                selected = [predicates[index] for index in sorted(selected_indices[:budget])]
                self.residual_allocator_ = "multimodal_numeric_reserve"
        self.residual_selected_family_counts_ = self._family_counts(selected)
        return selected

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        residual=None,
        hessian=None,
        mdl_baseline_probability=None,
        mdl_evidence_rows=None,
    ):
        X, y = np.asarray(X, float), np.asarray(y)
        if sample_weight is not None and len(sample_weight) != len(y):
            raise ValueError("sample_weight must have one value per predicate row")
        if residual is not None:
            residual = np.asarray(residual, dtype=float)
            if len(residual) != len(y):
                raise ValueError("residual must have one value per predicate row")
            hessian = np.ones(len(y), dtype=float) if hessian is None else np.asarray(hessian, dtype=float)
            if len(hessian) != len(y):
                raise ValueError("hessian must have one value per predicate row")
        if mdl_baseline_probability is not None:
            mdl_baseline_probability = np.asarray(mdl_baseline_probability, dtype=float)
            if mdl_baseline_probability.ndim != 1 or len(mdl_baseline_probability) != len(y):
                raise ValueError("mdl_baseline_probability must have one value per predicate row")
        if mdl_evidence_rows is not None:
            mdl_evidence_rows = np.asarray(mdl_evidence_rows, dtype=int)
            if (
                mdl_evidence_rows.ndim != 1
                or not len(mdl_evidence_rows)
                or np.any(mdl_evidence_rows < 0)
                or np.any(mdl_evidence_rows >= len(y))
                or len(np.unique(mdl_evidence_rows)) != len(mdl_evidence_rows)
            ):
                raise ValueError("mdl_evidence_rows must select unique in-range predicate rows")
        predicates = (
            self._candidates(
                X,
                y,
                sample_weight=sample_weight,
                residual=residual,
                hessian=hessian,
                mdl_baseline_probability=mdl_baseline_probability,
                mdl_evidence_rows=mdl_evidence_rows,
            )
            if self._fittable(X, y, numeric_rules=self.numeric_rules)
            else []
        )
        self.predicates = self._select_residual_candidates(predicates) if residual is not None else predicates
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        if not self.predicates:
            return X
        extra: list[np.ndarray] = [
            self._predicate_values(X, predicate).astype(float) for predicate in self.predicates
        ]
        return np.concatenate([X, np.column_stack(extra)], axis=1)

    def names(self, source_names=None):
        def label(j):
            if source_names is not None and j < len(source_names):
                return str(source_names[j])
            return f"feature[{j}]"

        out = []
        for predicate in self.predicates:
            if predicate.kind == "state":
                a, b = predicate.columns
                av, bv = divmod(predicate.value, 2)
                out.append(f"{label(a)}={av} AND {label(b)}={bv}")
            elif predicate.kind == "or":
                a, b = predicate.columns
                out.append(f"{label(a)} OR {label(b)}")
            elif predicate.kind == "xor":
                a, b = predicate.columns
                out.append(f"{label(a)} XOR {label(b)}")
            elif predicate.kind == "binary_and":
                terms = [
                    label(column) if positive else f"NOT {label(column)}"
                    for column, positive in zip(
                        predicate.columns,
                        predicate.directions,
                        strict=True,
                    )
                ]
                out.append(" AND ".join(terms))
            elif predicate.kind in {"binary_dnf", "binary_recursive_dnf"}:
                branch_widths = self._binary_dnf_widths(predicate)
                terms = [
                    label(column) if positive else f"NOT {label(column)}"
                    for column, positive in zip(
                        predicate.columns,
                        predicate.directions,
                        strict=True,
                    )
                ]
                branches = []
                start = 0
                for width in branch_widths:
                    branches.append(f"({' AND '.join(terms[start : start + width])})")
                    start += width
                out.append(" OR ".join(branches))
            elif predicate.kind == "binary_exception":
                branch_widths = self._binary_exception_widths(predicate)
                terms = [
                    label(column) if positive else f"NOT {label(column)}"
                    for column, positive in zip(
                        predicate.columns,
                        predicate.directions,
                        strict=True,
                    )
                ]
                branches = []
                start = 0
                for width in branch_widths:
                    branches.append(f"({' AND '.join(terms[start : start + width])})")
                    start += width
                exceptions = " OR ".join(branches[1:])
                out.append(f"{branches[0]} AND NOT ({exceptions})")
            elif predicate.kind == "threshold_union":
                terms = []
                for column, threshold, direction in zip(
                    predicate.columns, predicate.thresholds, predicate.directions, strict=False
                ):
                    operator = "<=" if direction else ">"
                    terms.append(f"{label(column)} {operator} {threshold:g}")
                if len(terms) < 2 or len(terms) % 2:
                    raise ValueError("threshold unions require complete interval branches")
                branches = [f"({terms[index]} AND {terms[index + 1]})" for index in range(0, len(terms), 2)]
                out.append(" OR ".join(branches))
            elif predicate.kind in {
                "threshold_and",
                "threshold_or",
                "threshold_interval",
            }:
                terms = []
                for column, threshold, direction in zip(
                    predicate.columns, predicate.thresholds, predicate.directions, strict=False
                ):
                    operator = "<=" if direction else ">"
                    terms.append(f"{label(column)} {operator} {threshold:g}")
                joiner = " AND " if predicate.kind in {"threshold_and", "threshold_interval"} else " OR "
                out.append(joiner.join(terms))
            else:
                cols = ", ".join(label(j) for j in predicate.columns)
                out.append(f"count({cols}) == {predicate.value}")
        return out


class _MulticlassResidualClassMap(SymbolicPredicateMap):
    """One class's residual compiler on a small, genuinely unseen verifier."""

    MIN_FIT_ROWS = 96
    ENFORCE_BINARY_OPERATION_BUDGET = True
    RESIDUAL_NUMERIC_COLUMN_POOL = 24
    RESIDUAL_THRESHOLD_MAX = 12


class MulticlassResidualPredicateMap(SymbolicPredicateMap):
    """Merge bounded one-vs-rest residual programs into one replayable schema.

    Each class gets its own Newton residual objective from the same coupled
    softmax probabilities. Predicates are then interleaved by within-class rank
    so a large or easy class cannot consume the entire finite feature budget.
    The resulting object is still a plain ``SymbolicPredicateMap`` at serving
    time; class attribution is diagnostics only.
    """

    MAX_MULTICLASS_PREDICATES = 16
    MIN_CLASS_EVIDENCE = 4
    MIN_FIT_ROWS = _MulticlassResidualClassMap.MIN_FIT_ROWS

    def __init__(self, seed=0, exclusive_groups=(), max_predicates=None):
        super().__init__(
            seed=seed,
            exclusive_groups=exclusive_groups,
            rare_rules=True,
            rare_class=1,
        )
        self.max_predicates = int(
            self.MAX_MULTICLASS_PREDICATES if max_predicates is None else max_predicates
        )
        self.predicate_classes_: list[object] = []
        self.predicate_updates_: list[tuple[float, float]] = []
        self.class_maps_: dict[object, SymbolicPredicateMap] = {}
        self.proposal_objective_ = "multiclass_booster_residual_newton_gain"

    def fit(self, X, y, probabilities, classes=None, sample_weight=None):  # type: ignore[override]
        X, y = np.asarray(X, float), np.asarray(y)
        classes = np.unique(y) if classes is None else np.asarray(classes)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(y), len(classes)):
            raise ValueError("probabilities must have one column per supplied class")
        if set(np.unique(y).tolist()) != set(classes.tolist()):
            raise ValueError("supplied classes do not match multiclass target")
        if sample_weight is not None and len(sample_weight) != len(y):
            raise ValueError("sample_weight must have one value per predicate row")
        self.class_maps_ = {}
        ranked = []

        for class_position, class_label in enumerate(classes):
            target = (y == class_label).astype(np.int8)
            probability = np.clip(probabilities[:, class_position], 1e-9, 1.0 - 1e-9)
            mapper = _MulticlassResidualClassMap(
                seed=self.seed + 104729 * class_position,
                exclusive_groups=self.exclusive_groups,
                rare_rules=True,
                rare_class=1,
            )
            if min(int(target.sum()), int((1 - target).sum())) >= self.MIN_CLASS_EVIDENCE:
                mapper.fit(
                    X,
                    target,
                    sample_weight=sample_weight,
                    residual=target - probability,
                    hessian=probability * (1.0 - probability),
                )
            self.class_maps_[class_label] = mapper
            ranked.append((class_label, mapper.predicates))

        predicates = []
        predicate_classes = []
        seen = set()
        max_rank = max((len(items) for _label, items in ranked), default=0)
        for rank in range(max_rank):
            for class_label, items in ranked:
                if rank >= len(items) or items[rank] in seen:
                    continue
                seen.add(items[rank])
                predicates.append(items[rank])
                predicate_classes.append(class_label)
                if len(predicates) >= self.max_predicates:
                    break
            if len(predicates) >= self.max_predicates:
                break
        self.predicates = predicates
        self.predicate_classes_ = predicate_classes
        self.residual_candidate_family_counts_ = {
            family: int(
                sum(
                    getattr(mapper, "residual_candidate_family_counts_", {}).get(family, 0)
                    for mapper in self.class_maps_.values()
                )
            )
            for family in ("boolean", "interval", "numeric_composition")
        }
        self.residual_selected_family_counts_ = self._family_counts(self.predicates)
        self.interval_union_candidates_ = int(
            sum(getattr(mapper, "interval_union_candidates_", 0) for mapper in self.class_maps_.values())
        )
        self.interval_union_predicates_selected_ = int(
            sum(
                getattr(mapper, "interval_union_predicates_selected_", 0)
                for mapper in self.class_maps_.values()
            )
        )
        self.residual_allocator_ = (
            "within_class_multimodal_numeric_reserve"
            if any(
                getattr(mapper, "residual_allocator_", None) == "multimodal_numeric_reserve"
                for mapper in self.class_maps_.values()
            )
            else "within_class_ordered_prefix"
        )
        weights = (
            np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        derived = self.transform(X)[:, X.shape[1] :]
        class_positions = {label: index for index, label in enumerate(classes)}
        updates = []
        for column, class_label in enumerate(predicate_classes):
            index = class_positions[class_label]
            target = (y == class_label).astype(float)
            probability = np.clip(probabilities[:, index], 1e-9, 1.0 - 1e-9)
            residual = weights * (target - probability)
            hessian = weights * probability * (1.0 - probability)
            mask = derived[:, column] > 0.5

            def leaf_value(rows, *, residual_values=residual, hessian_values=hessian):
                value = residual_values[rows].sum() / (hessian_values[rows].sum() + 1.0)
                return float(np.clip(value, -4.0, 4.0))

            updates.append((leaf_value(~mask), leaf_value(mask)))
        self.predicate_updates_ = updates
        return self

    def residual_score_update(self, scores, X, classes, learning_rate=0.05):
        """Apply the finite class-owned Newton stump head to raw softmax scores."""
        scores = np.asarray(scores, dtype=float)
        classes = np.asarray(classes)
        if scores.shape != (len(X), len(classes)):
            raise ValueError("scores must have one column per supplied class")
        class_index = {label: index for index, label in enumerate(classes)}
        if any(label not in class_index for label in self.predicate_classes_):
            raise ValueError("predicate class is absent from supplied score columns")
        out = scores.copy()
        if not self.predicates:
            return out
        derived = self.transform(X)[:, np.asarray(X).shape[1] :]
        for column, (class_label, update) in enumerate(
            zip(self.predicate_classes_, self.predicate_updates_, strict=False)
        ):
            mask = derived[:, column] > 0.5
            out[:, class_index[class_label]] += float(learning_rate) * np.where(
                mask,
                update[1],
                update[0],
            )
        return out


class MulticlassCrossfitPredicateMap(MulticlassResidualPredicateMap):
    """Merge fold-local predicates using only their untouched OOF rows.

    Each source predicate is scored and assigned Newton leaves on the outer
    validation rows that its proposing mapper never observed. This keeps the
    deployed finite map aligned with the representation that passed the gate,
    rather than recompiling an unrelated map from pooled residuals.
    """

    MIN_CROSSFIT_SUPPORT = 4

    def __init__(self, seed=0, exclusive_groups=(), max_predicates=None):
        super().__init__(
            seed=seed,
            exclusive_groups=exclusive_groups,
            max_predicates=max_predicates,
        )
        self.predicate_gains_: list[float] = []
        self.predicate_evidence_rows_: list[int] = []
        self.proposal_objective_ = "multiclass_crossfit_fold_predicate_merge"

    def fit_from_folds(
        self,
        X,
        y,
        probabilities,
        classes,
        fold_maps,
        valid_rows,
        sample_weight=None,
    ):
        X, y = np.asarray(X, float), np.asarray(y)
        classes = np.asarray(classes)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(y), len(classes)):
            raise ValueError("probabilities must have one column per supplied class")
        if len(fold_maps) != len(valid_rows):
            raise ValueError("each fold map needs aligned validation rows")
        if set(np.unique(y).tolist()) != set(classes.tolist()):
            raise ValueError("supplied classes do not match cross-fit target")
        evidence_rows = np.concatenate([np.asarray(rows, dtype=int) for rows in valid_rows])
        if (
            len(evidence_rows) != len(y)
            or np.any(evidence_rows < 0)
            or np.any(evidence_rows >= len(y))
            or not np.array_equal(np.sort(evidence_rows), np.arange(len(y)))
        ):
            raise ValueError("validation rows must partition the cross-fit evidence")
        weights = (
            np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        if len(weights) != len(y):
            raise ValueError("sample_weight must have one value per evidence row")
        class_index = {label: index for index, label in enumerate(classes)}
        self.interval_union_candidates_ = int(
            sum(getattr(mapper, "interval_union_candidates_", 0) for mapper in fold_maps)
        )
        self.interval_union_predicates_selected_ = int(
            sum(getattr(mapper, "interval_union_predicates_selected_", 0) for mapper in fold_maps)
        )
        statistics: dict[Any, Any] = {}

        for mapper, rows in zip(fold_maps, valid_rows, strict=False):
            rows = np.asarray(rows, dtype=int)
            if not mapper.predicates:
                continue
            if len(mapper.predicates) != len(mapper.predicate_classes_):
                raise ValueError("fold predicate ownership is incomplete")
            derived = mapper.transform(X[rows])[:, X.shape[1] :]
            for column, (predicate, owner) in enumerate(
                zip(mapper.predicates, mapper.predicate_classes_, strict=False)
            ):
                if owner not in class_index:
                    continue
                index = class_index[owner]
                target = (y[rows] == owner).astype(float)
                probability = np.clip(
                    probabilities[rows, index],
                    1e-9,
                    1.0 - 1e-9,
                )
                residual = weights[rows] * (target - probability)
                hessian = weights[rows] * probability * (1.0 - probability)
                mask = derived[:, column] > 0.5
                key = (owner, predicate)
                state = statistics.setdefault(
                    key,
                    {
                        "gradient_false": 0.0,
                        "gradient_true": 0.0,
                        "hessian_false": 0.0,
                        "hessian_true": 0.0,
                        "support_false": 0,
                        "support_true": 0,
                        "weight": 0.0,
                        "rows": 0,
                    },
                )
                state["gradient_false"] += float(residual[~mask].sum())
                state["gradient_true"] += float(residual[mask].sum())
                state["hessian_false"] += float(hessian[~mask].sum())
                state["hessian_true"] += float(hessian[mask].sum())
                state["support_false"] += int((~mask).sum())
                state["support_true"] += int(mask.sum())
                state["weight"] += float(weights[rows].sum())
                state["rows"] += int(len(rows))

        ranked_by_class: dict[Any, list[Any]] = {label: [] for label in classes}
        for (owner, predicate), state in statistics.items():
            if (
                state["support_false"] < self.MIN_CROSSFIT_SUPPORT
                or state["support_true"] < self.MIN_CROSSFIT_SUPPORT
            ):
                continue
            gradient_total = state["gradient_false"] + state["gradient_true"]
            hessian_total = state["hessian_false"] + state["hessian_true"]
            gain = (
                0.5
                * (
                    state["gradient_false"] ** 2 / (state["hessian_false"] + 1.0)
                    + state["gradient_true"] ** 2 / (state["hessian_true"] + 1.0)
                    - gradient_total**2 / (hessian_total + 1.0)
                )
                / max(state["weight"], 1e-12)
            )
            if gain < self.RESIDUAL_MIN_GAIN:
                continue
            update = (
                float(
                    np.clip(
                        state["gradient_false"] / (state["hessian_false"] + 1.0),
                        -4.0,
                        4.0,
                    )
                ),
                float(
                    np.clip(
                        state["gradient_true"] / (state["hessian_true"] + 1.0),
                        -4.0,
                        4.0,
                    )
                ),
            )
            ranked_by_class[owner].append((float(gain), predicate, update, int(state["rows"])))

        for items in ranked_by_class.values():
            items.sort(
                key=lambda item: (
                    -item[0],
                    item[1].kind,
                    item[1].columns,
                    item[1].value,
                    item[1].thresholds,
                    item[1].directions,
                )
            )

        selected = []
        max_rank = max((len(items) for items in ranked_by_class.values()), default=0)
        for rank in range(max_rank):
            for class_label in classes:
                items = ranked_by_class[class_label]
                if rank < len(items):
                    selected.append((class_label, *items[rank]))
                    if len(selected) >= self.max_predicates:
                        break
            if len(selected) >= self.max_predicates:
                break

        self.predicates = [predicate for _owner, _gain, predicate, _update, _rows in selected]
        self.predicate_classes_ = [owner for owner, _gain, _predicate, _update, _rows in selected]
        self.predicate_gains_ = [float(gain) for _owner, gain, _predicate, _update, _rows in selected]
        self.predicate_updates_ = [update for _owner, _gain, _predicate, update, _rows in selected]
        self.predicate_evidence_rows_ = [int(rows) for _owner, _gain, _predicate, _update, rows in selected]
        self.residual_selected_family_counts_ = self._family_counts(self.predicates)
        self.residual_allocator_ = "crossfit_oof_gain_with_class_balance"
        self.class_maps_ = {}
        return self
