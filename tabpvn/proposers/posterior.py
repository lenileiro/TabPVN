"""Transparent posterior challengers for categorical evidence.

The certified booster remains the baseline predictor.  This module builds a
small, auditable alternative from finite category facts: class counts for one
category or a bounded pair of categories, shrunk toward the global class prior
with a Dirichlet posterior. A bounded alternative can sequentially combine at
most three non-overlapping factors. Sparse pairs can instead shrink toward the
sequential posterior of their two single-category parents. The caller owns
cross-fit admission; this primitive only fits evidence and performs the declared
arithmetic.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _posterior_margin(probability) -> float:
    probability = np.asarray(probability, float)
    if len(probability) == 2:
        return float(abs(probability[0] - probability[1]))
    ordered = np.sort(probability)
    return float(ordered[-1] - ordered[-2])


def _unique_code_rows(codes):
    """Return unique encoded facts and the map back to source-row order."""
    codes = np.asarray(codes)
    if len(codes) < 2:
        return codes, np.arange(len(codes), dtype=np.int64)
    return np.unique(codes, axis=0, return_inverse=True)


_BATCHED_POSTERIOR_MIN_LOOKUPS = 8_192
_BATCHED_POSTERIOR_MAX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class _EvidenceTable:
    family: tuple[int, ...]
    counts: dict[tuple[int, ...], np.ndarray]
    information: float


class CategoricalPosteriorChallenger:
    """Dirichlet-smoothed evidence over atomic category facts.

    Every query either selects its strongest supported single/category-pair
    table or greedily pools at most three tables whose source groups do not
    overlap. Pooling is the explicit conditional-independence approximation:
    posterior/prior likelihood ratios are multiplied sequentially. A finite OOF
    gate chooses smoothing, aggregation, and discount, rejecting correlated
    evidence when that approximation does not transfer.
    """

    AGGREGATIONS = ("strongest", "disjoint_pool")
    SMOOTHING = ("global", "hierarchical")
    MAX_EVIDENCE_FACTORS = 3

    def __init__(
        self,
        X,
        y,
        classes,
        groups: Iterable[Iterable[int]],
        *,
        metadata: Iterable[dict[str, Any]] | None = None,
        max_pair_families: int = 24,
        max_focus_groups: int = 16,
        aggregation: str = "strongest",
        smoothing: str = "global",
    ):
        X, y = np.asarray(X, float), np.asarray(y)
        self.groups = tuple(tuple(int(column) for column in group) for group in groups)
        self.classes = np.asarray(classes)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row per label")
        if len(self.groups) < 2:
            raise ValueError("categorical posterior evidence needs at least two category groups")
        if len(self.classes) < 2:
            raise ValueError("categorical posterior evidence needs at least two classes")
        if any(not group or max(group) >= X.shape[1] for group in self.groups):
            raise ValueError("category groups must reference columns in X")
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {self.AGGREGATIONS}")
        if smoothing not in self.SMOOTHING:
            raise ValueError(f"smoothing must be one of {self.SMOOTHING}")
        self.aggregation = aggregation
        self.smoothing = smoothing

        class_index = {_python_scalar(label): index for index, label in enumerate(self.classes)}
        try:
            self.yidx = np.asarray([class_index[_python_scalar(label)] for label in y], dtype=np.int32)
        except KeyError as exc:
            raise ValueError("y contains a label outside classes") from exc
        self.C = len(self.classes)
        self.codes = self._codes(X)
        self.prior_counts = np.bincount(self.yidx, minlength=self.C).astype(np.int64)
        self.prior = self.prior_counts.astype(float) / max(float(self.prior_counts.sum()), 1.0)
        # One empirical pseudo-observation per class.  Support below this amount
        # cannot override the prior, which prevents singleton category levels
        # from becoming confident rules.
        self.prior_strength = float(self.C)
        self.minimum_support = int(np.ceil(self.prior_strength))
        self.metadata = self._normalize_metadata(metadata)

        singles = [self._build_table((group,)) for group in range(len(self.groups))]
        focus = self._focus_groups(singles, max_focus_groups)
        pairs = [self._build_table(pair) for pair in combinations(focus, 2)]
        pairs.sort(key=lambda table: (-table.information, table.family))
        selected_pairs = pairs[: max(0, int(max_pair_families))]
        # Stable family ordering makes evidence selection and certificates
        # bit-identical even when mutual-information scores tie.
        self.tables = tuple(singles + sorted(selected_pairs, key=lambda table: table.family))
        self.single_tables = {table.family[0]: table for table in singles}
        pair_support = [int(counts.sum()) for table in selected_pairs for counts in table.counts.values()]
        self.hierarchical_candidate = bool(
            pair_support and float(np.median(pair_support)) <= max(6.0 * self.C, 8.0)
        )

    def _normalize_metadata(self, metadata):
        if metadata is None:
            return tuple(
                {"name": f"category[{index}]", "levels": tuple(range(len(group)))}
                for index, group in enumerate(self.groups)
            )
        normalized = tuple(dict(item) for item in metadata)
        if len(normalized) != len(self.groups):
            raise ValueError("metadata must have one entry per category group")
        return normalized

    def _codes(self, X):
        X = np.asarray(X, float)
        out: np.ndarray = np.full((len(X), len(self.groups)), -1, dtype=np.int16)
        for group_index, group in enumerate(self.groups):
            block = X[:, group]
            present = block.max(1) > 0.5
            out[present, group_index] = block[present].argmax(1)
        return out

    def _build_table(self, family):
        family = tuple(int(group) for group in family)
        counts: dict[tuple[int, ...], np.ndarray] = {}
        local_codes = self.codes[:, family]
        valid = np.all(local_codes >= 0, axis=1)
        valid_codes = local_codes[valid]
        if len(valid_codes):
            dimensions = tuple(len(self.groups[group]) for group in family)
            multipliers = np.array(
                [int(np.prod(dimensions[index + 1 :], dtype=np.int64)) for index in range(len(family))],
                dtype=np.int64,
            )
            flat_codes = valid_codes @ multipliers
            _unique_codes, first_rows, inverse = np.unique(
                flat_codes,
                return_index=True,
                return_inverse=True,
            )
            flat_bins = inverse * self.C + self.yidx[valid]
            class_counts = np.bincount(
                flat_bins,
                minlength=len(first_rows) * self.C,
            ).reshape(len(first_rows), self.C)
            # Preserve first-observation order so information accumulation and
            # any tied family ranking remain bit-identical to the scalar path.
            for index in np.argsort(first_rows):
                key = tuple(int(level) for level in valid_codes[first_rows[index]])
                counts[key] = class_counts[index].astype(np.int64, copy=False)
        information = 0.0
        total = float(valid.sum())
        if total:
            for bucket in counts.values():
                support = float(bucket.sum())
                joint = bucket.astype(float) / total
                positive = joint > 0
                expected = (support / total) * self.prior
                information += float(np.sum(joint[positive] * np.log(joint[positive] / expected[positive])))
        return _EvidenceTable(family=family, counts=counts, information=information)

    @staticmethod
    def _focus_groups(singles, limit):
        """Keep high-signal groups plus deterministic breadth for pair search."""
        count = len(singles)
        limit = min(count, max(2, int(limit)))
        if count <= limit:
            return tuple(range(count))
        ranked = sorted(range(count), key=lambda index: (-singles[index].information, index))
        signal_count = max(1, limit - max(1, limit // 4))
        chosen = ranked[:signal_count]
        # Uniformly spaced groups preserve an interaction-only exploration path
        # when their individual mutual information is zero.
        for index in np.linspace(0, count - 1, count, dtype=int):
            if index not in chosen:
                chosen.append(int(index))
            if len(chosen) == limit:
                break
        return tuple(sorted(chosen))

    def _single_match(self, group, level, cache=None):
        key = (int(group), int(level))
        if cache is not None and (cached := cache.get(key)) is not None:
            return cached
        table = self.single_tables[int(group)]
        levels = (int(level),)
        counts = table.counts[levels]
        support = int(counts.sum())
        posterior = (counts + self.prior_strength * self.prior) / (support + self.prior_strength)
        strength = _posterior_margin(posterior) * support / (support + self.prior_strength)
        match = (
            table,
            levels,
            counts,
            support,
            posterior,
            strength,
            self.prior,
            (),
        )
        if cache is not None:
            cache[key] = match
        return match

    def _posterior_match(
        self,
        table,
        levels,
        counts,
        smoothing,
        *,
        match_cache=None,
        single_cache=None,
    ):
        cache_key = (smoothing, table.family, levels)
        if match_cache is not None and (cached := match_cache.get(cache_key)) is not None:
            return cached
        support = int(counts.sum())
        shrinkage_prior = self.prior
        parents = ()
        if smoothing == "hierarchical" and len(table.family) > 1:
            parents = tuple(
                self._single_match(group, level, single_cache)
                for group, level in zip(table.family, levels, strict=False)
            )
            parent_posteriors: list[np.ndarray] = []
            parent_match: Any
            for parent_match in parents:
                parent_posteriors.append(parent_match[4])
            shrinkage_prior = self._pool_posteriors(self.prior, parent_posteriors)
        posterior = (counts + self.prior_strength * shrinkage_prior) / (support + self.prior_strength)
        margin = _posterior_margin(posterior)
        strength = margin * support / (support + self.prior_strength)
        match = (
            table,
            levels,
            counts,
            support,
            posterior,
            strength,
            shrinkage_prior,
            parents,
        )
        if match_cache is not None:
            match_cache[cache_key] = match
        return match

    def _candidate_matches(self, code, smoothing=None, *, match_cache=None, single_cache=None):
        smoothing = self.smoothing if smoothing is None else smoothing
        if smoothing not in self.SMOOTHING:
            raise ValueError(f"smoothing must be one of {self.SMOOTHING}")
        matches = []
        for table in self.tables:
            levels = tuple(int(code[group]) for group in table.family)
            if any(level < 0 for level in levels):
                continue
            counts = table.counts.get(levels)
            if counts is None:
                continue
            support = int(counts.sum())
            if support < self.minimum_support:
                continue
            match = self._posterior_match(
                table,
                levels,
                counts,
                smoothing,
                match_cache=match_cache,
                single_cache=single_cache,
            )
            strength = match[5]
            rank = (
                strength,
                len(table.family),
                support,
                tuple(-group for group in table.family),
            )
            matches.append((rank, match))
        matches.sort(key=lambda item: item[0], reverse=True)
        return tuple(match for _rank, match in matches)

    def _select_candidates(self, candidates, aggregation):
        if aggregation == "strongest":
            return candidates[:1]
        selected = []
        used_groups: set[int] = set()
        for match in candidates:
            family = match[0].family
            if used_groups.isdisjoint(family):
                selected.append(match)
                used_groups.update(family)
            if len(selected) == self.MAX_EVIDENCE_FACTORS:
                break
        return tuple(selected)

    def _select_matches(self, code, aggregation, smoothing=None):
        return self._select_candidates(self._candidate_matches(code, smoothing=smoothing), aggregation)

    def _match(self, code):
        selected = self._select_matches(code, "strongest", self.smoothing)
        return selected[0] if selected else None

    @staticmethod
    def _pool_posteriors(prior, posteriors):
        """Sequential Bayes update from disjoint posterior/prior ratios."""
        prior = np.asarray(prior, float)
        if not posteriors:
            return prior.copy()
        log_probability = np.log(np.clip(prior, 1e-300, 1.0))
        for posterior in posteriors:
            log_probability += np.log(np.clip(posterior, 1e-300, 1.0))
            log_probability -= np.log(np.clip(prior, 1e-300, 1.0))
        log_probability -= log_probability.max()
        probability = np.exp(log_probability)
        return probability / probability.sum()

    def _posterior_from_codes(
        self,
        codes,
        *,
        return_matches,
        aggregation,
        smoothing,
        match_cache=None,
        single_cache=None,
    ):
        unique_codes, inverse = _unique_code_rows(codes)
        probabilities = np.tile(self.prior, (len(unique_codes), 1))
        matches = []
        match_cache = {} if match_cache is None else match_cache
        single_cache = {} if single_cache is None else single_cache
        for row, code in enumerate(unique_codes):
            candidates = self._candidate_matches(
                code,
                smoothing,
                match_cache=match_cache,
                single_cache=single_cache,
            )
            selected = self._select_candidates(candidates, aggregation)
            if selected:
                if aggregation == "strongest":
                    probabilities[row] = selected[0][4]
                else:
                    probabilities[row] = self._pool_posteriors(self.prior, [match[4] for match in selected])
            if return_matches:
                matches.append(selected)
        probabilities = probabilities[inverse]
        if return_matches:
            source_matches = [matches[index] for index in inverse]
            return probabilities, source_matches, codes
        return probabilities

    def posterior(self, X, *, return_matches=False, aggregation=None, smoothing=None):
        aggregation = self.aggregation if aggregation is None else aggregation
        smoothing = self.smoothing if smoothing is None else smoothing
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {self.AGGREGATIONS}")
        if smoothing not in self.SMOOTHING:
            raise ValueError(f"smoothing must be one of {self.SMOOTHING}")
        return self._posterior_from_codes(
            self._codes(X),
            return_matches=return_matches,
            aggregation=aggregation,
            smoothing=smoothing,
        )

    def _posterior_modes_from_unique_scalar(
        self,
        unique_codes,
        inverse,
        *,
        smoothing,
        match_cache=None,
        single_cache=None,
    ):
        probabilities = {
            aggregation: np.tile(self.prior, (len(unique_codes), 1)) for aggregation in self.AGGREGATIONS
        }
        match_cache = {} if match_cache is None else match_cache
        single_cache = {} if single_cache is None else single_cache
        for row, code in enumerate(unique_codes):
            candidates = self._candidate_matches(
                code,
                smoothing,
                match_cache=match_cache,
                single_cache=single_cache,
            )
            for aggregation in self.AGGREGATIONS:
                selected = self._select_candidates(candidates, aggregation)
                if not selected:
                    continue
                if aggregation == "strongest":
                    probabilities[aggregation][row] = selected[0][4]
                else:
                    probabilities[aggregation][row] = self._pool_posteriors(
                        self.prior, [match[4] for match in selected]
                    )
        return {aggregation: probability[inverse] for aggregation, probability in probabilities.items()}

    def _posterior_modes_from_codes_scalar(
        self,
        codes,
        *,
        smoothing=None,
        match_cache=None,
        single_cache=None,
    ):
        """Reference row-wise implementation retained for proofs and parity tests."""
        unique_codes, inverse = _unique_code_rows(codes)
        return self._posterior_modes_from_unique_scalar(
            unique_codes,
            inverse,
            smoothing=self.smoothing if smoothing is None else smoothing,
            match_cache=match_cache,
            single_cache=single_cache,
        )

    def _compiled_table_candidates(
        self,
        unique_codes,
        tables,
        smoothing,
        *,
        match_cache,
        single_cache,
    ):
        """Gather supported finite-table matches for an encoded batch."""
        row_count, table_count = len(unique_codes), len(tables)
        posteriors = np.broadcast_to(self.prior, (row_count, table_count, self.C)).copy()
        strengths: np.ndarray = np.full((row_count, table_count), -np.inf, dtype=float)
        supports: np.ndarray = np.zeros((row_count, table_count), dtype=np.int64)

        for table_index, table in enumerate(tables):
            family = table.family
            dimensions = tuple(len(self.groups[group]) for group in family)
            multipliers = np.array(
                [int(np.prod(dimensions[index + 1 :], dtype=np.int64)) for index in range(len(family))],
                dtype=np.int64,
            )
            known_levels = np.asarray(tuple(table.counts), dtype=np.int64)
            if not len(known_levels):
                continue
            known_keys = known_levels @ multipliers
            key_order = np.argsort(known_keys)
            sorted_keys = known_keys[key_order]

            local_codes = unique_codes[:, family]
            valid = np.all(
                (local_codes >= 0) & (local_codes < np.asarray(dimensions, dtype=np.int64)),
                axis=1,
            )
            rows = np.flatnonzero(valid)
            if not len(rows):
                continue
            query_keys = local_codes[valid] @ multipliers
            positions = np.searchsorted(sorted_keys, query_keys)
            bounded = np.minimum(positions, len(sorted_keys) - 1)
            found = (positions < len(sorted_keys)) & (sorted_keys[bounded] == query_keys)
            rows = rows[found]
            sources = key_order[bounded[found]]

            for source in np.unique(sources):
                levels = tuple(int(level) for level in known_levels[source])
                counts = table.counts[levels]
                support = int(counts.sum())
                if support < self.minimum_support:
                    continue
                match = self._posterior_match(
                    table,
                    levels,
                    counts,
                    smoothing,
                    match_cache=match_cache,
                    single_cache=single_cache,
                )
                selected_rows = rows[sources == source]
                posteriors[selected_rows, table_index] = match[4]
                strengths[selected_rows, table_index] = match[5]
                supports[selected_rows, table_index] = support
        return posteriors, strengths, supports

    @staticmethod
    def _compiled_candidate_order(tables, strengths, supports):
        """Vectorized equivalent of the scalar candidate-rank tuple."""
        row_count, table_count = strengths.shape
        lengths = np.array([len(table.family) for table in tables], dtype=np.int16)
        family_order = {
            family: index for index, family in enumerate(sorted(table.family for table in tables))
        }
        families = np.array([family_order[table.family] for table in tables], dtype=np.int32)
        return np.lexsort(
            (
                np.broadcast_to(families, (row_count, table_count)),
                -supports,
                -np.broadcast_to(lengths, (row_count, table_count)),
                -strengths,
            ),
            axis=1,
        )

    def _posterior_modes_from_unique_batched(
        self,
        unique_codes,
        inverse,
        *,
        smoothing,
        match_cache=None,
        single_cache=None,
    ):
        match_cache = {} if match_cache is None else match_cache
        single_cache = {} if single_cache is None else single_cache
        posteriors, strengths, supports = self._compiled_table_candidates(
            unique_codes,
            self.tables,
            smoothing,
            match_cache=match_cache,
            single_cache=single_cache,
        )
        order = self._compiled_candidate_order(self.tables, strengths, supports)
        probabilities = {
            aggregation: np.tile(self.prior, (len(unique_codes), 1)) for aggregation in self.AGGREGATIONS
        }
        if len(unique_codes):
            rows = np.arange(len(unique_codes))
            strongest = order[:, 0]
            matched = np.isfinite(strengths[rows, strongest])
            probabilities["strongest"][matched] = posteriors[rows[matched], strongest[matched]]

        pooled = probabilities["disjoint_pool"]
        for row, ranked_tables in enumerate(order):
            selected = []
            used_groups: set[int] = set()
            for table_index in ranked_tables:
                if not np.isfinite(strengths[row, table_index]):
                    break
                family = self.tables[int(table_index)].family
                if used_groups.isdisjoint(family):
                    selected.append(posteriors[row, table_index])
                    used_groups.update(family)
                if len(selected) == self.MAX_EVIDENCE_FACTORS:
                    break
            if selected:
                pooled[row] = self._pool_posteriors(self.prior, selected)
        return {aggregation: probability[inverse] for aggregation, probability in probabilities.items()}

    def _posterior_modes_from_codes(
        self,
        codes,
        *,
        smoothing=None,
        match_cache=None,
        single_cache=None,
    ):
        smoothing = self.smoothing if smoothing is None else smoothing
        unique_codes, inverse = _unique_code_rows(codes)
        lookups = len(unique_codes) * len(self.tables)
        working_bytes = lookups * (self.C * 8 + 32)
        if lookups < _BATCHED_POSTERIOR_MIN_LOOKUPS or working_bytes > _BATCHED_POSTERIOR_MAX_BYTES:
            return self._posterior_modes_from_unique_scalar(
                unique_codes,
                inverse,
                smoothing=smoothing,
                match_cache=match_cache,
                single_cache=single_cache,
            )
        return self._posterior_modes_from_unique_batched(
            unique_codes,
            inverse,
            smoothing=smoothing,
            match_cache=match_cache,
            single_cache=single_cache,
        )

    def posterior_modes(self, X):
        """Compute every aggregation for the configured smoothing mode."""
        return self._posterior_modes_from_codes(self._codes(X))

    def posterior_grid(self, X):
        """Compute every smoothing/aggregation candidate on one encoded batch."""
        codes = self._codes(X)
        unique_codes, inverse = _unique_code_rows(codes)
        probabilities = {
            (smoothing, aggregation): np.tile(self.prior, (len(unique_codes), 1))
            for smoothing in self.SMOOTHING
            for aggregation in self.AGGREGATIONS
        }
        match_cache: dict[Any, Any] = {}
        single_cache: dict[Any, Any] = {}
        for row, code in enumerate(unique_codes):
            global_candidates = self._candidate_matches(
                code,
                "global",
                match_cache=match_cache,
                single_cache=single_cache,
            )
            candidate_modes = {"global": global_candidates}
            candidate_modes["hierarchical"] = (
                self._candidate_matches(
                    code,
                    "hierarchical",
                    match_cache=match_cache,
                    single_cache=single_cache,
                )
                if self.hierarchical_candidate
                else global_candidates
            )
            for smoothing, candidates in candidate_modes.items():
                for aggregation in self.AGGREGATIONS:
                    selected = self._select_candidates(candidates, aggregation)
                    if not selected:
                        continue
                    if aggregation == "strongest":
                        probabilities[(smoothing, aggregation)][row] = selected[0][4]
                    else:
                        probabilities[(smoothing, aggregation)][row] = self._pool_posteriors(
                            self.prior, [match[4] for match in selected]
                        )
        return {mode: probability[inverse] for mode, probability in probabilities.items()}

    @staticmethod
    def combine_from_posterior(base_probability, posterior, prior, weight):
        """Apply a discounted posterior/prior likelihood ratio to a baseline."""
        base_probability = np.asarray(base_probability, float)
        posterior = np.asarray(posterior, float)
        prior = np.asarray(prior, float)
        if base_probability.shape != posterior.shape or base_probability.ndim != 2:
            raise ValueError("base and posterior probabilities must have the same 2-D shape")
        if prior.shape != (base_probability.shape[1],):
            raise ValueError("prior must have one value per probability column")
        if not 0.0 < float(weight) <= 1.0:
            raise ValueError("posterior evidence weight must be in (0, 1]")
        if (
            not np.isfinite(base_probability).all()
            or not np.isfinite(posterior).all()
            or not np.isfinite(prior).all()
            or (base_probability < 0).any()
            or (posterior < 0).any()
            or (prior <= 0).any()
        ):
            raise ValueError("probabilities and prior must be finite and non-negative")
        log_probability = np.log(np.clip(base_probability, 1e-300, 1.0))
        log_probability += float(weight) * (
            np.log(np.clip(posterior, 1e-300, 1.0)) - np.log(np.clip(prior, 1e-300, 1.0))
        )
        log_probability -= log_probability.max(1, keepdims=True)
        combined = np.exp(log_probability)
        return combined / combined.sum(1, keepdims=True)

    def combine(self, base_probability, X, weight, *, aggregation=None, smoothing=None):
        posterior = self.posterior(X, aggregation=aggregation, smoothing=smoothing)
        return self.combine_from_posterior(base_probability, posterior, self.prior, weight)

    def _conditions(self, family, levels):
        conditions = []
        for group, level in zip(family, levels, strict=False):
            meta = self.metadata[group]
            declared_levels = tuple(meta.get("levels", ()))
            value = declared_levels[level] if 0 <= level < len(declared_levels) else int(level)
            conditions.append(
                {
                    "group": int(group),
                    "name": str(meta.get("name", f"category[{group}]")),
                    "level_index": int(level),
                    "level": _python_scalar(value),
                }
            )
        return conditions

    def _factor_record(self, match):
        (
            table,
            levels,
            counts,
            support,
            posterior,
            strength,
            shrinkage_prior,
            parents,
        ) = match
        record = {
            "family": [int(group) for group in table.family],
            "conditions": self._conditions(table.family, levels),
            "class_counts": [int(value) for value in counts],
            "support": int(support),
            "shrinkage_prior": np.asarray(shrinkage_prior, float).tolist(),
            "posterior": posterior.tolist(),
            "evidence_strength": float(strength),
        }
        if parents:
            record["parent_factors"] = [self._factor_record(parent) for parent in parents]
        return record

    def evidence(self, X, row, base_probability, weight):
        """Return the selected finite facts and re-checkable posterior arithmetic."""
        posterior, matches, _codes = self.posterior(X, return_matches=True)
        base_probability = np.asarray(base_probability, float)
        if base_probability.shape != posterior.shape:
            raise ValueError("base_probability must align with X and classes")
        combined = self.combine_from_posterior(base_probability, posterior, self.prior, weight)
        selected = matches[int(row)]
        factors = [self._factor_record(match) for match in selected]
        family = [group for factor in factors for group in factor["family"]]
        conditions = [condition for factor in factors for condition in factor["conditions"]]
        base_index = int(base_probability[int(row)].argmax())
        prediction_index = int(combined[int(row)].argmax())
        record = {
            "kind": (
                "categorical_dirichlet_posterior"
                if self.aggregation == "strongest"
                else "categorical_dirichlet_posterior_pool"
            ),
            "aggregation": self.aggregation,
            "smoothing": self.smoothing,
            "classes": [_python_scalar(value) for value in self.classes],
            "family": family,
            "conditions": conditions,
            "global_class_counts": [int(value) for value in self.prior_counts],
            "minimum_support": int(self.minimum_support),
            "prior_strength": float(self.prior_strength),
            "prior": self.prior.tolist(),
            "posterior": posterior[int(row)].tolist(),
            "weight": float(weight),
            "base_probability": base_probability[int(row)].tolist(),
            "combined_probability": combined[int(row)].tolist(),
            "base_prediction": _python_scalar(self.classes[base_index]),
            "prediction": _python_scalar(self.classes[prediction_index]),
            "override": bool(base_index != prediction_index),
        }
        if self.aggregation == "strongest":
            factor = (
                factors[0]
                if factors
                else {
                    "family": [],
                    "conditions": [],
                    "class_counts": [0] * self.C,
                    "support": 0,
                    "shrinkage_prior": self.prior.tolist(),
                    "posterior": self.prior.tolist(),
                    "evidence_strength": 0.0,
                }
            )
            record.update(
                class_counts=factor["class_counts"],
                support=factor["support"],
                shrinkage_prior=factor["shrinkage_prior"],
                parent_factors=factor.get("parent_factors", []),
                evidence_strength=factor["evidence_strength"],
            )
        else:
            record.update(
                factors=factors,
                maximum_factors=self.MAX_EVIDENCE_FACTORS,
            )
        record["verified"] = self.verify_evidence(record)
        return record

    @staticmethod
    def verify_evidence(record, tol=1e-9):  # noqa: C901 - fail-closed certificate verifier
        """Recompute a posterior certificate without model state."""
        try:
            global_counts = np.asarray(record["global_class_counts"], float)
            classes = list(record["classes"])
            if (
                record.get("kind")
                not in {
                    "categorical_dirichlet_posterior",
                    "categorical_dirichlet_posterior_pool",
                }
                or global_counts.ndim != 1
                or len(global_counts) != len(classes)
                or not np.isfinite(global_counts).all()
                or (global_counts <= 0).any()
                or not np.allclose(global_counts, np.round(global_counts), atol=tol)
            ):
                return False
            prior = global_counts / global_counts.sum()
            strength = float(record["prior_strength"])
            minimum_support = float(record["minimum_support"])
            smoothing = record.get("smoothing", "global")
            if (
                not np.isfinite(strength)
                or strength <= 0
                or minimum_support != float(np.ceil(strength))
                or smoothing not in CategoricalPosteriorChallenger.SMOOTHING
            ):
                return False

            def verify_factor(factor, *, factor_smoothing, allow_empty=False):
                counts = np.asarray(factor["class_counts"], float)
                family = [int(group) for group in factor["family"]]
                conditions = list(factor["conditions"])
                if (
                    counts.shape != global_counts.shape
                    or not np.isfinite(counts).all()
                    or (counts < 0).any()
                    or not np.allclose(counts, np.round(counts), atol=tol)
                    or len(family) != len(conditions)
                    or len(family) != len(set(family))
                    or any(
                        int(condition["group"]) != group or int(condition["level_index"]) < 0
                        for group, condition in zip(family, conditions, strict=False)
                    )
                ):
                    raise ValueError("invalid posterior factor")
                support = float(counts.sum())
                if support == 0:
                    if not allow_empty or family or conditions:
                        raise ValueError("empty posterior factor")
                elif support < minimum_support or len(family) not in (1, 2, 3):
                    raise ValueError("unsupported posterior factor")
                parents = list(factor.get("parent_factors", []))
                shrinkage_prior = np.asarray(factor.get("shrinkage_prior", prior), float)
                expected_prior = prior
                if support == 0 or len(family) <= 1 or factor_smoothing == "global":
                    if parents:
                        raise ValueError("unexpected posterior parents")
                else:
                    if len(parents) != len(family):
                        raise ValueError("hierarchical conjunction needs one parent per fact")
                    parent_posteriors = []
                    for index, parent in enumerate(parents):
                        parent_posterior, parent_family, parent_conditions = verify_factor(
                            parent,
                            factor_smoothing="global",
                        )
                        if (
                            parent_family != [family[index]]
                            or len(parent_conditions) != 1
                            or parent_conditions[0] != conditions[index]
                        ):
                            raise ValueError("hierarchical parent does not match pair fact")
                        parent_posteriors.append(parent_posterior)
                    expected_prior = CategoricalPosteriorChallenger._pool_posteriors(prior, parent_posteriors)
                if (
                    shrinkage_prior.shape != prior.shape
                    or not np.isfinite(shrinkage_prior).all()
                    or (shrinkage_prior <= 0).any()
                    or not np.allclose(shrinkage_prior, expected_prior, atol=tol)
                ):
                    raise ValueError("invalid shrinkage prior")
                posterior = (counts + strength * shrinkage_prior) / (support + strength)
                evidence_strength = (
                    _posterior_margin(posterior) * support / (support + strength)
                    if support >= minimum_support
                    else 0.0
                )
                if (
                    support != float(factor["support"])
                    or not np.allclose(posterior, factor["posterior"], atol=tol)
                    or abs(evidence_strength - float(factor["evidence_strength"])) > tol
                ):
                    raise ValueError("posterior factor arithmetic mismatch")
                return posterior, family, conditions

            family: list[int]
            conditions: list[dict[str, Any]]
            if record["kind"] == "categorical_dirichlet_posterior":
                if record.get("aggregation", "strongest") != "strongest":
                    return False
                posterior, family, conditions = verify_factor(
                    {
                        "family": record["family"],
                        "conditions": record["conditions"],
                        "class_counts": record["class_counts"],
                        "support": record["support"],
                        "shrinkage_prior": record.get("shrinkage_prior", prior),
                        "parent_factors": record.get("parent_factors", []),
                        "posterior": record["posterior"],
                        "evidence_strength": record["evidence_strength"],
                    },
                    factor_smoothing=smoothing,
                    allow_empty=True,
                )
            else:
                if (
                    record.get("aggregation") != "disjoint_pool"
                    or int(record["maximum_factors"]) != CategoricalPosteriorChallenger.MAX_EVIDENCE_FACTORS
                ):
                    return False
                factors = list(record["factors"])
                if len(factors) > CategoricalPosteriorChallenger.MAX_EVIDENCE_FACTORS:
                    return False
                posteriors: list[np.ndarray] = []
                family = []
                conditions = []
                used_groups: set[int] = set()
                for factor in factors:
                    local_posterior, local_family, local_conditions = verify_factor(
                        factor,
                        factor_smoothing=smoothing,
                    )
                    if not used_groups.isdisjoint(local_family):
                        return False
                    used_groups.update(local_family)
                    family.extend(local_family)
                    conditions.extend(local_conditions)
                    posteriors.append(local_posterior)
                if family != list(record["family"]) or conditions != list(record["conditions"]):
                    return False
                posterior = CategoricalPosteriorChallenger._pool_posteriors(prior, posteriors)

            base = np.asarray(record["base_probability"], float)[None, :]
            if base.shape != (1, len(classes)):
                return False
            combined = CategoricalPosteriorChallenger.combine_from_posterior(
                base, posterior[None, :], prior, float(record["weight"])
            )[0]
            base_prediction = classes[int(base.argmax(1)[0])]
            prediction = classes[int(combined.argmax())]
            return bool(
                np.allclose(prior, record["prior"], atol=tol)
                and np.allclose(posterior, record["posterior"], atol=tol)
                and np.allclose(combined, record["combined_probability"], atol=tol)
                and base_prediction == record["base_prediction"]
                and prediction == record["prediction"]
                and bool(base_prediction != prediction) == bool(record["override"])
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError):
            return False

    def report(self):
        pairs = [table for table in self.tables if len(table.family) == 2]
        return {
            "groups": len(self.groups),
            "families": len(self.tables),
            "pair_families": len(pairs),
            "patterns": int(sum(len(table.counts) for table in self.tables)),
            "prior_strength": self.prior_strength,
            "minimum_support": self.minimum_support,
            "aggregation": self.aggregation,
            "smoothing": self.smoothing,
            "hierarchical_candidate": self.hierarchical_candidate,
            "maximum_factors": self.MAX_EVIDENCE_FACTORS,
        }


class NumericIntervalPosteriorChallenger:
    """Finite Dirichlet evidence over supervised, fold-local quantile intervals.

    Numeric values are converted into a small one-hot vocabulary of intervals,
    then delegated to the same count-table arithmetic as categorical evidence.
    Feature ranking and cut points are learned only from the rows passed to this
    object, so a caller can safely construct one object per OOF fitting fold.
    """

    # A fixed fine partition is safer than selecting resolution per dataset: the
    # latter won OOF at ten bins but failed to transfer on the motivating fold.
    # Dirichlet shrinkage and minimum support keep sparse 16-bin cells bounded.
    QUANTILE_BINS = 16
    MAX_FEATURES = 12
    MAX_TRIPLE_FEATURES = 8
    MAX_TRIPLE_FAMILIES = 12
    TRIPLE_FALLBACK = "strongest_triple_fallback"
    BASE_AGGREGATIONS = CategoricalPosteriorChallenger.AGGREGATIONS
    AGGREGATIONS = BASE_AGGREGATIONS + (TRIPLE_FALLBACK,)
    SMOOTHING = CategoricalPosteriorChallenger.SMOOTHING

    def __init__(
        self,
        X,
        y,
        classes,
        columns: Iterable[int] | None = None,
        *,
        names: Iterable[str] | None = None,
        aggregation: str = "strongest",
        smoothing: str = "global",
    ):
        X, y = np.asarray(X, float), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError("finite 2-D X must have one row per label")
        requested = tuple(range(X.shape[1])) if columns is None else tuple(int(c) for c in columns)
        if len(requested) != len(set(requested)) or any(c < 0 or c >= X.shape[1] for c in requested):
            raise ValueError("numeric columns must be unique valid columns in X")
        declared_names = tuple(str(c) for c in requested) if names is None else tuple(str(n) for n in names)
        if len(declared_names) != len(requested):
            raise ValueError("names must align with numeric columns")
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {self.AGGREGATIONS}")

        class_rows = [np.flatnonzero(y == value) for value in np.asarray(classes)]
        ranked = []
        for column, name in zip(requested, declared_names, strict=False):
            values = X[:, column]
            variance = float(np.var(values))
            if not np.isfinite(variance) or variance <= 1e-12:
                continue
            mean = float(values.mean())
            between = sum(
                len(rows) * (float(values[rows].mean()) - mean) ** 2 for rows in class_rows if len(rows)
            ) / max(len(values), 1)
            ranked.append((float(between / variance), -column, column, name))
        ranked.sort(reverse=True)

        selected = []
        quantiles = np.linspace(0.0, 1.0, self.QUANTILE_BINS + 1)[1:-1]
        for score, _negative_column, column, name in ranked:
            cuts = np.unique(np.quantile(X[:, column], quantiles)).astype(float)
            cuts = cuts[np.isfinite(cuts)]
            if len(cuts):
                selected.append((column, name, float(score), cuts))
            if len(selected) == self.MAX_FEATURES:
                break
        if len(selected) < 2:
            raise ValueError("numeric interval evidence needs two varying numeric features")

        self.features = tuple(selected)
        encoded, groups, metadata = self._encode_with_schema(X)
        delegate_aggregation = "strongest" if aggregation == self.TRIPLE_FALLBACK else aggregation
        self._delegate = CategoricalPosteriorChallenger(
            encoded,
            y,
            classes,
            groups,
            metadata=metadata,
            aggregation=delegate_aggregation,
            smoothing=smoothing,
        )
        singles = [table for table in self._delegate.tables if len(table.family) == 1]
        focus = self._delegate._focus_groups(singles, self.MAX_TRIPLE_FEATURES)
        triples = [self._delegate._build_table(family) for family in combinations(focus, 3)]
        triples.sort(key=lambda table: (-table.information, table.family))
        self.triple_tables = tuple(triples[: self.MAX_TRIPLE_FAMILIES])
        self.classes = self._delegate.classes
        self.prior = self._delegate.prior
        self.aggregation = aggregation
        self.smoothing = smoothing
        self.hierarchical_candidate = self._delegate.hierarchical_candidate

    def _interval_specs(self):
        out = []
        for group, (column, name, score, cuts) in enumerate(self.features):
            levels = []
            for level in range(len(cuts) + 1):
                levels.append(
                    {
                        "column": int(column),
                        "lower": None if level == 0 else float(cuts[level - 1]),
                        "upper": None if level == len(cuts) else float(cuts[level]),
                    }
                )
            out.append(
                {
                    "group": int(group),
                    "column": int(column),
                    "name": name,
                    "score": float(score),
                    "cut_points": [float(value) for value in cuts],
                    "levels": tuple(levels),
                }
            )
        return out

    def _encode_with_schema(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("numeric interval queries must be a finite 2-D matrix")
        specs = self._interval_specs()
        width = sum(len(spec["levels"]) for spec in specs)
        encoded: np.ndarray = np.zeros((len(X), width), dtype=float)
        groups, metadata, offset = [], [], 0
        rows = np.arange(len(X))
        for spec in specs:
            cuts = np.asarray(spec["cut_points"], float)
            codes = np.searchsorted(cuts, X[:, spec["column"]], side="right")
            group = tuple(range(offset, offset + len(cuts) + 1))
            encoded[rows, offset + codes] = 1.0
            groups.append(group)
            metadata.append({"name": spec["name"], "levels": spec["levels"]})
            offset += len(group)
        return encoded, tuple(groups), tuple(metadata)

    def _encode(self, X):
        return self._encode_with_schema(X)[0]

    def _interval_codes(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("numeric interval queries must be a finite 2-D matrix")
        codes: np.ndarray = np.empty((len(X), len(self.features)), dtype=np.int16)
        for group, (column, _name, _score, cuts) in enumerate(self.features):
            codes[:, group] = np.searchsorted(cuts, X[:, column], side="right")
        return codes

    def posterior_grid(self, X):
        return self._delegate.posterior_grid(self._encode(X))

    def posterior_modes(self, X):
        """Compute aggregation candidates for the configured smoothing only."""
        return self._delegate._posterior_modes_from_codes(self._interval_codes(X))

    def _triple_posterior_codes_scalar(
        self,
        codes,
        *,
        return_matches=False,
        match_cache=None,
        single_cache=None,
    ):
        delegate = self._delegate
        unique_codes, inverse = _unique_code_rows(codes)
        probabilities = np.tile(delegate.prior, (len(unique_codes), 1))
        selected_matches = []
        match_cache = {} if match_cache is None else match_cache
        single_cache = {} if single_cache is None else single_cache
        for row, code in enumerate(unique_codes):
            selected = None
            selected_rank = None
            for table in self.triple_tables:
                levels = tuple(int(code[group]) for group in table.family)
                if any(level < 0 for level in levels):
                    continue
                counts = table.counts.get(levels)
                if counts is None or int(counts.sum()) < delegate.minimum_support:
                    continue
                match = delegate._posterior_match(
                    table,
                    levels,
                    counts,
                    self.smoothing,
                    match_cache=match_cache,
                    single_cache=single_cache,
                )
                rank = (
                    match[5],
                    match[3],
                    tuple(-group for group in table.family),
                )
                if selected_rank is None or rank > selected_rank:
                    selected, selected_rank = match, rank
            if selected is not None:
                probabilities[row] = selected[4]
            if return_matches:
                selected_matches.append((selected,) if selected is not None else ())
        probabilities = probabilities[inverse]
        if return_matches:
            return probabilities, [selected_matches[index] for index in inverse]
        return probabilities

    def _triple_posterior_codes(
        self,
        codes,
        *,
        return_matches=False,
        match_cache=None,
        single_cache=None,
    ):
        unique_codes, inverse = _unique_code_rows(codes)
        lookups = len(unique_codes) * len(self.triple_tables)
        working_bytes = lookups * (self._delegate.C * 8 + 32)
        if (
            return_matches
            or lookups < _BATCHED_POSTERIOR_MIN_LOOKUPS
            or working_bytes > _BATCHED_POSTERIOR_MAX_BYTES
        ):
            return self._triple_posterior_codes_scalar(
                codes,
                return_matches=return_matches,
                match_cache=match_cache,
                single_cache=single_cache,
            )

        delegate = self._delegate
        match_cache = {} if match_cache is None else match_cache
        single_cache = {} if single_cache is None else single_cache
        posteriors, strengths, supports = delegate._compiled_table_candidates(
            unique_codes,
            self.triple_tables,
            self.smoothing,
            match_cache=match_cache,
            single_cache=single_cache,
        )
        order = delegate._compiled_candidate_order(self.triple_tables, strengths, supports)
        probabilities = np.tile(delegate.prior, (len(unique_codes), 1))
        if len(unique_codes):
            rows = np.arange(len(unique_codes))
            strongest = order[:, 0]
            matched = np.isfinite(strengths[rows, strongest])
            probabilities[matched] = posteriors[rows[matched], strongest[matched]]
        return probabilities[inverse]

    def _triple_posterior_encoded(self, encoded, *, return_matches=False):
        return self._triple_posterior_codes(
            self._delegate._codes(encoded),
            return_matches=return_matches,
        )

    def triple_posterior(self, X, *, return_matches=False):
        """Return the strongest supported bounded three-interval conjunction."""
        return self._triple_posterior_codes(
            self._interval_codes(X),
            return_matches=return_matches,
        )

    def posterior_modes_with_triple(self, X):
        """Compute pair/pool and triple candidates from one interval encoding."""
        codes = self._interval_codes(X)
        match_cache: dict[Any, Any] = {}
        single_cache: dict[Any, Any] = {}
        modes = self._delegate._posterior_modes_from_codes(
            codes,
            match_cache=match_cache,
            single_cache=single_cache,
        )
        triple = self._triple_posterior_codes(
            codes,
            match_cache=match_cache,
            single_cache=single_cache,
        )
        return modes, triple

    @staticmethod
    def combine_from_posterior(base_probability, posterior, prior, weight):
        return CategoricalPosteriorChallenger.combine_from_posterior(
            base_probability, posterior, prior, weight
        )

    @staticmethod
    def combine_with_triple_fallback_from_posteriors(
        base_probability,
        incumbent_posterior,
        triple_posterior,
        prior,
        weight,
        *,
        return_mask=False,
    ):
        """Keep incumbent decisions and use a triple only on untouched labels."""
        incumbent = CategoricalPosteriorChallenger.combine_from_posterior(
            base_probability, incumbent_posterior, prior, weight
        )
        triple = CategoricalPosteriorChallenger.combine_from_posterior(
            base_probability, triple_posterior, prior, weight
        )
        base_class = np.asarray(base_probability).argmax(1)
        incumbent_class = incumbent.argmax(1)
        triple_class = triple.argmax(1)
        use_triple = (incumbent_class == base_class) & (triple_class != base_class)
        combined = incumbent.copy()
        combined[use_triple] = triple[use_triple]
        return (combined, use_triple) if return_mask else combined

    def combine(self, base_probability, X, weight):
        codes = self._interval_codes(X)
        match_cache: dict[Any, Any] = {}
        single_cache: dict[Any, Any] = {}
        if self.aggregation == self.TRIPLE_FALLBACK:
            incumbent = self._delegate._posterior_from_codes(
                codes,
                return_matches=False,
                aggregation="strongest",
                smoothing=self.smoothing,
                match_cache=match_cache,
                single_cache=single_cache,
            )
            triple = self._triple_posterior_codes(
                codes,
                match_cache=match_cache,
                single_cache=single_cache,
            )
            return self.combine_with_triple_fallback_from_posteriors(
                base_probability, incumbent, triple, self.prior, weight
            )
        posterior = self._delegate._posterior_from_codes(
            codes,
            return_matches=False,
            aggregation=self.aggregation,
            smoothing=self.smoothing,
            match_cache=match_cache,
            single_cache=single_cache,
        )
        return self.combine_from_posterior(base_probability, posterior, self.prior, weight)

    @staticmethod
    def _numeric_kind(categorical_kind):
        return (
            "numeric_interval_dirichlet_posterior"
            if categorical_kind == "categorical_dirichlet_posterior"
            else "numeric_interval_dirichlet_posterior_pool"
        )

    def _decorate_numeric_record(self, record, X, row):
        record["kind"] = self._numeric_kind(record["kind"])
        record["interval_features"] = [
            {key: value for key, value in spec.items() if key != "levels"} for spec in self._interval_specs()
        ]
        observed = {int(column): float(X[int(row), column]) for column, *_rest in self.features}

        def decorate_condition(condition):
            if condition.get("kind") == "numeric_interval":
                return
            interval = condition.pop("level")
            condition.update(
                kind="numeric_interval",
                column=int(interval["column"]),
                lower=interval["lower"],
                upper=interval["upper"],
                lower_inclusive=True,
                upper_inclusive=False,
                observed=observed[int(interval["column"])],
            )

        def decorate_factor(factor):
            for condition in factor.get("conditions", []):
                decorate_condition(condition)
            for parent in factor.get("parent_factors", []):
                decorate_factor(parent)

        for condition in record.get("conditions", []):
            decorate_condition(condition)
        for factor in record.get("factors", []):
            decorate_factor(factor)
        for parent in record.get("parent_factors", []):
            decorate_factor(parent)
        record["verified"] = self.verify_evidence(record)
        return record

    def _strongest_record(self, match, base_probability, row, weight):
        delegate = self._delegate
        factor = delegate._factor_record(match)
        posterior = np.asarray(match[4], float)
        base = np.asarray(base_probability, float)[int(row)]
        combined = self.combine_from_posterior(base[None, :], posterior[None, :], delegate.prior, weight)[0]
        base_index = int(base.argmax())
        prediction_index = int(combined.argmax())
        return {
            "kind": "categorical_dirichlet_posterior",
            "aggregation": "strongest",
            "smoothing": self.smoothing,
            "classes": [_python_scalar(value) for value in delegate.classes],
            "family": factor["family"],
            "conditions": factor["conditions"],
            "global_class_counts": [int(value) for value in delegate.prior_counts],
            "minimum_support": int(delegate.minimum_support),
            "prior_strength": float(delegate.prior_strength),
            "prior": delegate.prior.tolist(),
            "posterior": posterior.tolist(),
            "weight": float(weight),
            "base_probability": base.tolist(),
            "combined_probability": combined.tolist(),
            "base_prediction": _python_scalar(delegate.classes[base_index]),
            "prediction": _python_scalar(delegate.classes[prediction_index]),
            "override": bool(base_index != prediction_index),
            "class_counts": factor["class_counts"],
            "support": factor["support"],
            "shrinkage_prior": factor["shrinkage_prior"],
            "parent_factors": factor.get("parent_factors", []),
            "evidence_strength": factor["evidence_strength"],
        }

    def evidence(self, X, row, base_probability, weight):
        X = np.asarray(X, float)
        encoded = self._encode(X)
        incumbent_record = self._decorate_numeric_record(
            self._delegate.evidence(encoded, row, base_probability, weight), X, row
        )
        if self.aggregation != self.TRIPLE_FALLBACK:
            return incumbent_record

        incumbent = self._delegate.posterior(encoded, aggregation="strongest", smoothing=self.smoothing)
        triple, triple_matches = self._triple_posterior_encoded(encoded, return_matches=True)
        _combined, use_triple = self.combine_with_triple_fallback_from_posteriors(
            base_probability,
            incumbent,
            triple,
            self.prior,
            weight,
            return_mask=True,
        )
        if not use_triple[int(row)]:
            return incumbent_record

        selected = triple_matches[int(row)]
        if not selected:
            raise ValueError("triple fallback selected without finite evidence")
        record = self._decorate_numeric_record(
            self._strongest_record(selected[0], base_probability, row, weight), X, row
        )
        record["decision_mode"] = self.TRIPLE_FALLBACK
        record["fallback_incumbent"] = incumbent_record
        record["verified"] = self.verify_evidence(record)
        return record

    @staticmethod
    def verify_evidence(record, tol=1e-9):
        """Recheck both interval membership and delegated count arithmetic."""
        try:
            kinds = {
                "numeric_interval_dirichlet_posterior": "categorical_dirichlet_posterior",
                "numeric_interval_dirichlet_posterior_pool": "categorical_dirichlet_posterior_pool",
            }
            if record.get("kind") not in kinds:
                return False
            features = list(record["interval_features"])
            by_group = {}
            for index, feature in enumerate(features):
                group = int(feature["group"])
                column = int(feature["column"])
                cuts = np.asarray(feature["cut_points"], float)
                if (
                    group != index
                    or group in by_group
                    or column < 0
                    or cuts.ndim != 1
                    or not np.isfinite(cuts).all()
                    or (len(cuts) and np.any(np.diff(cuts) <= 0))
                ):
                    return False
                by_group[group] = (column, str(feature["name"]), cuts)

            def verify_condition(condition):
                group = int(condition["group"])
                level = int(condition["level_index"])
                column, name, cuts = by_group[group]
                lower = None if level == 0 else float(cuts[level - 1])
                upper = None if level == len(cuts) else float(cuts[level])
                observed = float(condition["observed"])
                if level < 0 or level > len(cuts) or not np.isfinite(observed):
                    raise ValueError("invalid numeric interval level")
                if (
                    condition.get("kind") != "numeric_interval"
                    or int(condition["column"]) != column
                    or str(condition["name"]) != name
                    or condition.get("lower_inclusive") is not True
                    or condition.get("upper_inclusive") is not False
                    or (lower is None) != (condition.get("lower") is None)
                    or (upper is None) != (condition.get("upper") is None)
                ):
                    raise ValueError("numeric interval declaration mismatch")
                if lower is not None and (
                    abs(float(condition["lower"]) - lower) > tol or observed < lower - tol
                ):
                    raise ValueError("numeric interval lower bound mismatch")
                if upper is not None and (abs(float(condition["upper"]) - upper) > tol or observed >= upper):
                    raise ValueError("numeric interval upper bound mismatch")

            def verify_factor(factor):
                for condition in factor.get("conditions", []):
                    verify_condition(condition)
                for parent in factor.get("parent_factors", []):
                    verify_factor(parent)

            for condition in record.get("conditions", []):
                verify_condition(condition)
            for factor in record.get("factors", []):
                verify_factor(factor)
            for parent in record.get("parent_factors", []):
                verify_factor(parent)

            decision_mode = record.get("decision_mode")
            if decision_mode is not None:
                if (
                    decision_mode != NumericIntervalPosteriorChallenger.TRIPLE_FALLBACK
                    or record.get("aggregation") != "strongest"
                    or len(record.get("family", [])) != 3
                    or record.get("override") is not True
                ):
                    return False
                incumbent = record["fallback_incumbent"]
                if (
                    incumbent.get("decision_mode") is not None
                    or not NumericIntervalPosteriorChallenger.verify_evidence(incumbent, tol=tol)
                    or incumbent.get("override") is not False
                    or incumbent.get("prediction") != record.get("base_prediction")
                    or incumbent.get("base_prediction") != record.get("base_prediction")
                    or incumbent.get("classes") != record.get("classes")
                    or incumbent.get("interval_features") != record.get("interval_features")
                    or abs(float(incumbent["weight"]) - float(record["weight"])) > tol
                    or not np.allclose(
                        incumbent["base_probability"],
                        record["base_probability"],
                        atol=tol,
                    )
                ):
                    return False
            elif "fallback_incumbent" in record:
                return False

            delegated = copy.deepcopy(record)
            delegated["kind"] = kinds[record["kind"]]
            return CategoricalPosteriorChallenger.verify_evidence(delegated, tol=tol)
        except (KeyError, TypeError, ValueError, IndexError):
            return False

    def report(self):
        out = self._delegate.report()
        out.update(
            kind="numeric_interval",
            aggregation=self.aggregation,
            quantile_bins=self.QUANTILE_BINS,
            triple_feature_limit=self.MAX_TRIPLE_FEATURES,
            triple_family_limit=self.MAX_TRIPLE_FAMILIES,
            triple_families=len(self.triple_tables),
            triple_patterns=int(sum(len(table.counts) for table in self.triple_tables)),
            selected_features=[
                {
                    "column": int(column),
                    "name": name,
                    "score": float(score),
                    "cut_points": [float(value) for value in cuts],
                }
                for column, name, score, cuts in self.features
            ],
        )
        return out


__all__ = ["CategoricalPosteriorChallenger", "NumericIntervalPosteriorChallenger"]
