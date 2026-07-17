"""Relational input detection and deterministic kernel-derived features."""

from __future__ import annotations

import hashlib
from array import array
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

Triple = tuple[str, str, str]
RelationChain = tuple[str, ...]
ReachVector = tuple[frozenset[str], ...]
SuffixTable = dict[str, frozenset[str]]
ChainScore = tuple[int, int, RelationChain]
PromotionSpec = tuple[RelationChain, str]
PromotionBuffer = dict[tuple[str, str], set[str]]
IdArray: TypeAlias = NDArray[np.int64]
EncodedSparseReach: TypeAlias = tuple[tuple[int, IdArray], ...]
EncodedPromotionBuffer: TypeAlias = dict[str, list[tuple[int, IdArray]]]

_RELATION_HIERARCHY_STATE_CAP = 2_000_000
_RELATION_MATERIALIZATION_CHUNK_ROWS = 2048
_RELATION_BUILD_CHUNK_EDGES = 1_000_000
_RELATION_RESIDENT_ARRAY_BYTES = 128 * 1024 * 1024
_RELATION_HEAD_BUCKET_WIDTH = 4096
_EMPTY_IDS: IdArray = np.empty(0, dtype=np.int64)
_EMPTY_IDS.flags.writeable = False
_EntityT = TypeVar("_EntityT")


class _HierarchyOverflow(RuntimeError):
    """The exact half-path cache exceeded its fixed endpoint-membership budget."""


@dataclass(slots=True)
class _ChainTrieNode:
    """One shared relation prefix used during chunked chain execution."""

    children: dict[int, _ChainTrieNode] = field(default_factory=dict)
    feature_indices: list[int] = field(default_factory=list)
    promoted_relations: list[str] = field(default_factory=list)


def _merge_id_arrays(values: Iterable[IdArray]) -> IdArray:
    arrays = [value for value in values if value.size]
    if not arrays:
        return _EMPTY_IDS
    if len(arrays) == 1:
        return arrays[0]
    return np.unique(np.concatenate(arrays))


@dataclass(slots=True)
class _RelationSegment:
    """Immutable sparse CSR segment grouped by integer head and relation IDs."""

    heads: IdArray
    head_group_offsets: IdArray
    relations: IdArray
    tail_offsets: IdArray
    tails: IdArray
    head_lookup: IdArray | None
    head_lookup_start: int

    @classmethod
    def from_arrays(
        cls,
        relation_ids: IdArray,
        head_ids: IdArray,
        tail_ids: IdArray,
    ) -> _RelationSegment:
        if not (relation_ids.size == head_ids.size == tail_ids.size):
            raise ValueError("edge arrays must have equal lengths")
        if not head_ids.size:
            empty_head_offsets: IdArray = np.zeros(1, dtype=np.int64)
            empty_tail_offsets: IdArray = np.zeros(1, dtype=np.int64)
            empty_head_offsets.flags.writeable = False
            empty_tail_offsets.flags.writeable = False
            return cls(
                _EMPTY_IDS,
                empty_head_offsets,
                _EMPTY_IDS,
                empty_tail_offsets,
                _EMPTY_IDS,
                None,
                0,
            )

        order = np.lexsort((tail_ids, relation_ids, head_ids))
        sorted_heads = np.asarray(head_ids[order], dtype=np.int64)
        sorted_relations = np.asarray(relation_ids[order], dtype=np.int64)
        sorted_tails = np.asarray(tail_ids[order], dtype=np.int64)
        distinct = np.ones(sorted_heads.size, dtype=bool)
        distinct[1:] = (
            (sorted_heads[1:] != sorted_heads[:-1])
            | (sorted_relations[1:] != sorted_relations[:-1])
            | (sorted_tails[1:] != sorted_tails[:-1])
        )
        sorted_heads = sorted_heads[distinct]
        sorted_relations = sorted_relations[distinct]
        sorted_tails = sorted_tails[distinct]

        group_starts_mask = np.ones(sorted_heads.size, dtype=bool)
        group_starts_mask[1:] = (sorted_heads[1:] != sorted_heads[:-1]) | (
            sorted_relations[1:] != sorted_relations[:-1]
        )
        group_starts: IdArray = np.flatnonzero(group_starts_mask).astype(np.int64, copy=False)
        group_heads = sorted_heads[group_starts]
        group_relations = sorted_relations[group_starts]
        tail_offsets: IdArray = np.empty(group_starts.size + 1, dtype=np.int64)
        tail_offsets[:-1] = group_starts
        tail_offsets[-1] = sorted_tails.size

        head_starts_mask = np.ones(group_heads.size, dtype=bool)
        head_starts_mask[1:] = group_heads[1:] != group_heads[:-1]
        head_starts: IdArray = np.flatnonzero(head_starts_mask).astype(np.int64, copy=False)
        head_group_offsets: IdArray = np.empty(head_starts.size + 1, dtype=np.int64)
        head_group_offsets[:-1] = head_starts
        head_group_offsets[-1] = group_heads.size
        heads = np.asarray(group_heads[head_starts], dtype=np.int64)
        head_lookup: IdArray | None = None
        head_lookup_start = int(heads[0])
        lookup_size = int(heads[-1]) - head_lookup_start + 1
        if lookup_size <= 10_000_000 and lookup_size <= max(1024, 4 * heads.size):
            head_lookup = np.full(lookup_size, -1, dtype=np.int64)
            head_lookup[heads - head_lookup_start] = np.arange(heads.size, dtype=np.int64)
        arrays = (heads, head_group_offsets, group_relations, tail_offsets, sorted_tails)
        for values in arrays:
            values.flags.writeable = False
        if head_lookup is not None:
            head_lookup.flags.writeable = False
        return cls(
            heads,
            head_group_offsets,
            np.asarray(group_relations, dtype=np.int64),
            tail_offsets,
            np.asarray(sorted_tails, dtype=np.int64),
            head_lookup,
            head_lookup_start,
        )

    def _group_bounds(self, head_id: int) -> tuple[int, int] | None:
        if self.head_lookup is not None:
            lookup_index = head_id - self.head_lookup_start
            if lookup_index < 0 or lookup_index >= self.head_lookup.size:
                return None
            position = int(self.head_lookup[lookup_index])
            if position < 0:
                return None
        else:
            position = int(np.searchsorted(self.heads, head_id))
            if position >= self.heads.size or int(self.heads[position]) != head_id:
                return None
        return int(self.head_group_offsets[position]), int(self.head_group_offsets[position + 1])

    def iter_edges(self, head_id: int) -> Iterable[tuple[int, IdArray]]:
        bounds = self._group_bounds(head_id)
        if bounds is None:
            return
        start, stop = bounds
        for group_index in range(start, stop):
            tail_start = int(self.tail_offsets[group_index])
            tail_stop = int(self.tail_offsets[group_index + 1])
            yield int(self.relations[group_index]), self.tails[tail_start:tail_stop]

    def targets(self, head_id: int, relation_id: int) -> IdArray:
        bounds = self._group_bounds(head_id)
        if bounds is None:
            return _EMPTY_IDS
        start, stop = bounds
        relation_slice = self.relations[start:stop]
        relative = int(np.searchsorted(relation_slice, relation_id))
        if relative >= relation_slice.size or int(relation_slice[relative]) != relation_id:
            return _EMPTY_IDS
        group_index = start + relative
        tail_start = int(self.tail_offsets[group_index])
        tail_stop = int(self.tail_offsets[group_index + 1])
        return self.tails[tail_start:tail_stop]

    def route_single_head(
        self,
        head_id: int,
        row_index: int,
        allowed: frozenset[int],
        routed: defaultdict[int, list[tuple[int, IdArray]]],
    ) -> None:
        bounds = self._group_bounds(head_id)
        if bounds is None:
            return
        start, stop = bounds
        for group_index in range(start, stop):
            relation_id = int(self.relations[group_index])
            if relation_id not in allowed:
                continue
            tail_start = int(self.tail_offsets[group_index])
            tail_stop = int(self.tail_offsets[group_index + 1])
            routed[relation_id].append((row_index, self.tails[tail_start:tail_stop]))

    def to_memmap(self, directory: Path, segment_index: int) -> _RelationSegment:
        """Persist every owned array and reopen it read-only."""

        def persist(name: str, values: IdArray) -> IdArray:
            if not values.size:
                return _EMPTY_IDS
            path = directory / f"segment-{segment_index:06d}-{name}.npy"
            writable: IdArray = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.int64,
                shape=values.shape,
            )
            writable[:] = values
            writable.flush()
            del writable
            loaded = cast(IdArray, np.load(path, mmap_mode="r", allow_pickle=False))
            loaded.flags.writeable = False
            return loaded

        return _RelationSegment(
            persist("heads", self.heads),
            persist("head-group-offsets", self.head_group_offsets),
            persist("relations", self.relations),
            persist("tail-offsets", self.tail_offsets),
            persist("tails", self.tails),
            None if self.head_lookup is None else persist("head-lookup", self.head_lookup),
            self.head_lookup_start,
        )

    @property
    def array_bytes(self) -> int:
        return int(
            self.heads.nbytes
            + self.head_group_offsets.nbytes
            + self.relations.nbytes
            + self.tail_offsets.nbytes
            + self.tails.nbytes
            + (0 if self.head_lookup is None else self.head_lookup.nbytes)
        )

    @property
    def mapped_array_count(self) -> int:
        arrays = (
            self.heads,
            self.head_group_offsets,
            self.relations,
            self.tail_offsets,
            self.tails,
        )
        count = sum(isinstance(values, np.memmap) for values in arrays)
        return count + int(self.head_lookup is not None and isinstance(self.head_lookup, np.memmap))


class _CompactRelationGraph:
    """Integer-coded append-only graph used by the production feature path."""

    def __init__(
        self,
        *,
        build_chunk_edges: int = _RELATION_BUILD_CHUNK_EDGES,
        resident_array_bytes: int = _RELATION_RESIDENT_ARRAY_BYTES,
    ) -> None:
        if build_chunk_edges <= 0:
            raise ValueError("build_chunk_edges must be positive")
        if resident_array_bytes < 0:
            raise ValueError("resident_array_bytes must be non-negative")
        self._node_ids: dict[str, int] | None = {}
        self._node_names: list[str] = []
        self._relation_ids: dict[str, int] = {}
        self._relation_names: list[str] = []
        self._segments: list[_RelationSegment] = []
        self._segment_buckets: defaultdict[int, list[int]] = defaultdict(list)
        self._build_chunk_edges = int(build_chunk_edges)
        self._resident_array_bytes = int(resident_array_bytes)
        self._spill_directory: TemporaryDirectory[str] | None = None
        self._next_segment_index = 0

    def _start_spilling(self) -> None:
        if self._spill_directory is not None:
            return
        self._spill_directory = TemporaryDirectory(prefix="tabpvn-relational-")
        directory = Path(self._spill_directory.name)
        self._segments = [
            segment.to_memmap(directory, segment_index)
            for segment_index, segment in enumerate(self._segments)
        ]
        self._next_segment_index = len(self._segments)

    def _append_segment(self, segment: _RelationSegment) -> None:
        if not segment.tails.size:
            return
        if (
            self._spill_directory is None
            and self.array_bytes + segment.array_bytes > self._resident_array_bytes
        ):
            self._start_spilling()
        if self._spill_directory is not None:
            directory = Path(self._spill_directory.name)
            segment = segment.to_memmap(directory, self._next_segment_index)
            self._next_segment_index += 1
        segment_index = len(self._segments)
        self._segments.append(segment)
        buckets = np.unique(segment.heads // _RELATION_HEAD_BUCKET_WIDTH)
        for bucket in buckets:
            self._segment_buckets[int(bucket)].append(segment_index)

    def _intern_node(self, name: str) -> int:
        if self._node_ids is None:
            raise RuntimeError("node vocabulary is frozen")
        existing = self._node_ids.get(name)
        if existing is not None:
            return existing
        node_id = len(self._node_names)
        self._node_ids[name] = node_id
        self._node_names.append(name)
        return node_id

    def _intern_relation(self, name: str) -> int:
        existing = self._relation_ids.get(name)
        if existing is not None:
            return existing
        relation_id = len(self._relation_names)
        self._relation_ids[name] = relation_id
        self._relation_names.append(name)
        return relation_id

    @classmethod
    def from_triples(
        cls,
        triples: Iterable[Triple],
        entities: Sequence[str],
        *,
        build_chunk_edges: int = _RELATION_BUILD_CHUNK_EDGES,
        resident_array_bytes: int = _RELATION_RESIDENT_ARRAY_BYTES,
    ) -> _CompactRelationGraph:
        graph = cls(
            build_chunk_edges=build_chunk_edges,
            resident_array_bytes=resident_array_bytes,
        )
        for entity in entities:
            graph._intern_node(entity)
        relation_values = array("q")
        head_values = array("q")
        tail_values = array("q")

        def flush() -> None:
            nonlocal relation_values, head_values, tail_values
            segment = _RelationSegment.from_arrays(
                np.frombuffer(relation_values, dtype=np.int64),
                np.frombuffer(head_values, dtype=np.int64),
                np.frombuffer(tail_values, dtype=np.int64),
            )
            graph._append_segment(segment)
            relation_values = array("q")
            head_values = array("q")
            tail_values = array("q")

        for relation, head, tail in triples:
            relation_values.append(graph._intern_relation(relation))
            head_values.append(graph._intern_node(head))
            tail_values.append(graph._intern_node(tail))
            if len(relation_values) >= graph._build_chunk_edges:
                flush()
        flush()
        return graph

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Mapping[tuple[str, str], set[str]],
        entities: Sequence[str],
        *,
        build_chunk_edges: int = _RELATION_BUILD_CHUNK_EDGES,
        resident_array_bytes: int = _RELATION_RESIDENT_ARRAY_BYTES,
    ) -> _CompactRelationGraph:
        triples = (
            (relation, head, tail)
            for (relation, head), tails in sorted(adjacency.items())
            for tail in sorted(tails)
        )
        return cls.from_triples(
            triples,
            entities,
            build_chunk_edges=build_chunk_edges,
            resident_array_bytes=resident_array_bytes,
        )

    def node_ids(self, names: Sequence[str]) -> tuple[int, ...]:
        if self._node_ids is None:
            raise RuntimeError("node vocabulary is frozen")
        return tuple(self._node_ids[name] for name in names)

    def freeze_nodes(self) -> None:
        self._node_ids = None

    def node_name(self, node_id: int) -> str:
        return self._node_names[node_id]

    def relation_id(self, name: str) -> int:
        return self._relation_ids[name]

    def relation_name(self, relation_id: int) -> str:
        return self._relation_names[relation_id]

    @property
    def relation_names(self) -> tuple[str, ...]:
        return tuple(self._relation_names)

    def iter_edges(self, head_id: int) -> Iterable[tuple[int, IdArray]]:
        bucket = head_id // _RELATION_HEAD_BUCKET_WIDTH
        for segment_index in self._segment_buckets.get(bucket, ()):
            yield from self._segments[segment_index].iter_edges(head_id)

    def targets(self, head_id: int, relation_id: int) -> IdArray:
        bucket = head_id // _RELATION_HEAD_BUCKET_WIDTH
        return _merge_id_arrays(
            self._segments[segment_index].targets(head_id, relation_id)
            for segment_index in self._segment_buckets.get(bucket, ())
        )

    @property
    def single_segment(self) -> _RelationSegment | None:
        return self._segments[0] if len(self._segments) == 1 else None

    def mining_adjacency(
        self,
        source_ids: Sequence[int],
        relation_names: Sequence[str],
        max_depth: int,
    ) -> defaultdict[tuple[str, str], set[str]]:
        allowed = {
            relation_id
            for name in relation_names
            if (relation_id := self._relation_ids.get(name)) is not None
        }
        adjacency: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        frontier = set(source_ids)
        expanded: set[int] = set()
        for _depth in range(max_depth):
            current = frontier - expanded
            if not current:
                break
            next_frontier: set[int] = set()
            for head_id in sorted(current):
                for relation_id, tails in self.iter_edges(head_id):
                    if relation_id not in allowed:
                        continue
                    key = (self.relation_name(relation_id), self.node_name(head_id))
                    adjacency[key].update(self.node_name(int(tail_id)) for tail_id in tails)
                    next_frontier.update(int(tail_id) for tail_id in tails)
            expanded.update(current)
            frontier = next_frontier
        return adjacency

    def add_promotions(self, additions: EncodedPromotionBuffer) -> None:
        relation_values = array("q")
        head_values = array("q")
        tail_values = array("q")

        def flush() -> None:
            nonlocal relation_values, head_values, tail_values
            segment = _RelationSegment.from_arrays(
                np.frombuffer(relation_values, dtype=np.int64),
                np.frombuffer(head_values, dtype=np.int64),
                np.frombuffer(tail_values, dtype=np.int64),
            )
            self._append_segment(segment)
            relation_values = array("q")
            head_values = array("q")
            tail_values = array("q")

        for relation_name in sorted(additions):
            relation_id = self._intern_relation(relation_name)
            for head_id, tails in additions[relation_name]:
                for tail_id in tails:
                    relation_values.append(relation_id)
                    head_values.append(head_id)
                    tail_values.append(int(tail_id))
                    if len(relation_values) >= self._build_chunk_edges:
                        flush()
        flush()

    @property
    def array_bytes(self) -> int:
        return sum(segment.array_bytes for segment in self._segments)

    @property
    def edge_count(self) -> int:
        return sum(int(segment.tails.size) for segment in self._segments)

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def is_spilled(self) -> bool:
        return self._spill_directory is not None

    @property
    def mapped_array_count(self) -> int:
        return sum(segment.mapped_array_count for segment in self._segments)


class _RelationPathIndex:
    """Topology-only half-path index for exact bidirectional chain discovery.

    Forward tables retain reachable middle nodes for each scanned source. Suffix
    tables start only from those real middle nodes, so disconnected graph regions
    are never scanned. The selected chain remains the shortcut witness; no
    synthetic relation enters the output.
    """

    def __init__(
        self,
        adjacency: Mapping[tuple[str, str], set[str]],
        sources: Sequence[str],
        relations: Sequence[str] | None = None,
        state_cap: int = _RELATION_HIERARCHY_STATE_CAP,
    ) -> None:
        if state_cap <= 0:
            raise ValueError("state_cap must be positive")
        self._adjacency = adjacency
        self._relations = (
            tuple(sorted({relation for relation, _head in adjacency}))
            if relations is None
            else tuple(dict.fromkeys(relations))
        )
        self._outgoing: dict[str, dict[str, set[str]]] = {}
        self.sources = tuple(sources)
        self.state_cap = int(state_cap)
        self.state_size = 0
        self._forward: list[dict[RelationChain, ReachVector]] = [
            {(): tuple(frozenset((source,)) for source in self.sources)}
        ]
        self._suffix: list[dict[RelationChain, SuffixTable]] = [{}]
        self._suffix_starts: frozenset[str] | None = None

    def _charge(self, endpoint_memberships: int) -> None:
        self.state_size += int(endpoint_memberships)
        if self.state_size > self.state_cap:
            raise _HierarchyOverflow

    def edges(self, node: str) -> Mapping[str, set[str]]:
        """Return lazily indexed outgoing edges for one reachable node."""
        cached = self._outgoing.get(node)
        if cached is not None:
            return cached
        relation_map = {
            relation: tails
            for relation in self._relations
            if (tails := self._adjacency.get((relation, node)))
        }
        self._outgoing[node] = relation_map
        return relation_map

    def forward_layers(self, max_depth: int) -> list[dict[RelationChain, ReachVector]]:
        """Return exact source-to-middle states through ``max_depth`` relations."""
        while len(self._forward) <= max_depth:
            previous = self._forward[-1]
            layer: dict[RelationChain, ReachVector] = {}
            for prefix, reach in sorted(previous.items()):
                relations = {
                    relation for frontier in reach for node in frontier for relation in self.edges(node)
                }
                for relation in sorted(relations):
                    rows = []
                    for frontier in reach:
                        tails = {tail for node in frontier for tail in self.edges(node).get(relation, ())}
                        rows.append(frozenset(tails))
                    if any(rows):
                        state = tuple(rows)
                        self._charge(sum(len(values) for values in state))
                        layer[prefix + (relation,)] = state
            self._forward.append(layer)
        return self._forward

    def suffix_layers(
        self,
        max_depth: int,
        starts: Iterable[str],
    ) -> list[dict[RelationChain, SuffixTable]]:
        """Return exact middle-to-tail states for the requested meeting nodes."""
        requested_starts = frozenset(starts)
        if self._suffix_starts is None:
            self._suffix_starts = requested_starts
        elif requested_starts != self._suffix_starts:
            raise ValueError("suffix layers are bound to one set of meeting nodes")

        if max_depth >= 1 and len(self._suffix) == 1:
            first_mutable: dict[RelationChain, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
            for head in sorted(requested_starts):
                relation_map = self.edges(head)
                for relation, tails in relation_map.items():
                    target = first_mutable[(relation,)][head]
                    previous_size = len(target)
                    target.update(tails)
                    self._charge(len(target) - previous_size)
            first = {
                pattern: {start: frozenset(tails) for start, tails in table.items()}
                for pattern, table in first_mutable.items()
            }
            self._suffix.append(first)

        while len(self._suffix) <= max_depth:
            previous = self._suffix[-1]
            layer_mutable: dict[RelationChain, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
            for suffix, table in sorted(previous.items()):
                for start, frontier in sorted(table.items()):
                    relations = {relation for middle in frontier for relation in self.edges(middle)}
                    for relation in sorted(relations):
                        expanded_tails = {
                            tail for middle in frontier for tail in self.edges(middle).get(relation, ())
                        }
                        if expanded_tails:
                            target = layer_mutable[suffix + (relation,)][start]
                            previous_size = len(target)
                            target.update(expanded_tails)
                            self._charge(len(target) - previous_size)
            layer = {
                pattern: {start: frozenset(tails) for start, tails in table.items()}
                for pattern, table in layer_mutable.items()
            }
            self._suffix.append(layer)
        return self._suffix


def _is_relational(data: Any) -> bool:
    """Return whether ``data`` has the supported relation/head/tail shape."""
    return bool(
        isinstance(data, (list, tuple))
        and data
        and isinstance(data[0], (list, tuple))
        and len(data[0]) == 3
        and all(isinstance(value, str) for value in data[0])
    )


def _chain_reach(
    adjacency: Mapping[tuple[str, str], set[str]],
    pattern: RelationChain,
    entity: str,
) -> set[str]:
    """Return tails reachable from ``entity`` through one relation chain."""
    frontier = {entity}
    for relation in pattern:
        next_frontier = set()
        for node in frontier:
            next_frontier |= adjacency.get((relation, node), set())
        frontier = next_frontier
        if not frontier:
            break
    return frontier


def _candidate_chains(relations: Sequence[str], max_length: int) -> list[RelationChain]:
    if max_length <= 0:
        raise ValueError("max_len must be positive")
    return [chain for length in range(1, max_length + 1) for chain in product(relations, repeat=length)]


def _bounded_entity_scan(entities: Sequence[_EntityT], scan_cap: int) -> tuple[_EntityT, ...]:
    if scan_cap <= 0:
        raise ValueError("scan_cap must be positive")
    if len(entities) <= scan_cap:
        return tuple(entities)
    indices = np.unique(np.linspace(0, len(entities) - 1, scan_cap).astype(int))
    return tuple(entities[int(index)] for index in indices)


def _score_counts(
    counts: Iterable[int],
    pattern: RelationChain,
    discriminative: bool,
) -> ChainScore | None:
    values = tuple(counts)
    nonzero = sum(value != 0 for value in values)
    if nonzero == 0:
        return None
    if not discriminative:
        return nonzero, 0, pattern
    distinct = len(set(values))
    return (distinct, nonzero, pattern) if distinct > 1 else None


def _score_key(score: ChainScore) -> tuple[int, int, RelationChain]:
    return -score[0], -score[1], score[2]


def _trim_scores(scored: list[ChainScore], max_features: int, *, force: bool = False) -> None:
    """Retain the exact best scores while bounding discovery memory by O(k)."""
    if not force and len(scored) <= max(256, 4 * max_features):
        return
    scored.sort(key=_score_key)
    del scored[max_features:]


def _ranked_patterns(scored: list[ChainScore], max_features: int) -> list[RelationChain]:
    _trim_scores(scored, max_features, force=True)
    return [score[2] for score in scored]


def _mine_chains_exhaustive(
    adjacency: Mapping[tuple[str, str], set[str]],
    relations: Sequence[str],
    entities: Sequence[str],
    max_length: int,
    max_features: int,
    discriminative: bool = True,
    scan_cap: int = 4000,
) -> list[RelationChain]:
    """Reference implementation that scores every relation string exactly."""
    if max_features < 0:
        raise ValueError("max_features must be non-negative")
    scan = _bounded_entity_scan(entities, scan_cap)
    scored: list[ChainScore] = []
    for pattern in _candidate_chains(relations, max_length):
        score = _score_counts(
            (len(_chain_reach(adjacency, pattern, entity)) for entity in scan),
            pattern,
            discriminative,
        )
        if score is not None:
            scored.append(score)
    scored.sort(key=_score_key)
    return [score[2] for score in scored[:max_features]]


def _meeting_nodes(forward: Sequence[dict[RelationChain, ReachVector]]) -> frozenset[str]:
    return frozenset(
        node for layer in forward[1:] for reach in layer.values() for frontier in reach for node in frontier
    )


def _suffixes_by_start(layer: Mapping[RelationChain, SuffixTable]) -> dict[str, tuple[RelationChain, ...]]:
    mutable: dict[str, list[RelationChain]] = defaultdict(list)
    for pattern, table in layer.items():
        for start in table:
            mutable[start].append(pattern)
    return {start: tuple(sorted(patterns)) for start, patterns in mutable.items()}


def _record_score(
    scored: list[ChainScore],
    score: ChainScore | None,
    max_features: int,
) -> None:
    if score is None:
        return
    scored.append(score)
    _trim_scores(scored, max_features)


def _mine_chains_bidirectional(
    index: _RelationPathIndex,
    max_length: int,
    max_features: int,
    discriminative: bool,
) -> list[RelationChain]:
    """Score only topology-valid chains by joining exact cached half paths."""
    left_depth = (max_length + 1) // 2
    right_depth = max_length // 2
    forward = index.forward_layers(left_depth)
    suffix = index.suffix_layers(right_depth, _meeting_nodes(forward))
    suffix_indexes: list[dict[str, tuple[RelationChain, ...]]] = [{}]
    suffix_indexes.extend(_suffixes_by_start(layer) for layer in suffix[1:])
    scored: list[ChainScore] = []

    for length in range(1, max_length + 1):
        prefix_depth = (length + 1) // 2
        suffix_depth = length // 2
        for prefix, reach in sorted(forward[prefix_depth].items()):
            if suffix_depth == 0:
                _record_score(
                    scored,
                    _score_counts((len(frontier) for frontier in reach), prefix, discriminative),
                    max_features,
                )
                continue

            candidate_suffixes = {
                candidate
                for frontier in reach
                for middle in frontier
                for candidate in suffix_indexes[suffix_depth].get(middle, ())
            }
            for suffix_pattern in sorted(candidate_suffixes):
                table = suffix[suffix_depth][suffix_pattern]
                counts = []
                for frontier in reach:
                    tails = {tail for middle in frontier for tail in table.get(middle, ())}
                    counts.append(len(tails))
                pattern = prefix + suffix_pattern
                _record_score(
                    scored,
                    _score_counts(counts, pattern, discriminative),
                    max_features,
                )
    return _ranked_patterns(scored, max_features)


def _mine_chains_streaming(
    index: _RelationPathIndex,
    sources: Sequence[str],
    max_length: int,
    max_features: int,
    discriminative: bool,
) -> list[RelationChain]:
    """Exact bounded-memory fallback for indexes that exceed the state budget."""
    scored: list[ChainScore] = []
    initial: ReachVector = tuple(frozenset((source,)) for source in sources)

    def visit(pattern: RelationChain, reach: ReachVector) -> None:
        if len(pattern) >= max_length:
            return
        relations = {relation for frontier in reach for node in frontier for relation in index.edges(node)}
        for relation in sorted(relations):
            rows = tuple(
                frozenset(tail for node in frontier for tail in index.edges(node).get(relation, ()))
                for frontier in reach
            )
            if not any(rows):
                continue
            child = pattern + (relation,)
            _record_score(
                scored,
                _score_counts((len(frontier) for frontier in rows), child, discriminative),
                max_features,
            )
            visit(child, rows)

    visit((), initial)
    return _ranked_patterns(scored, max_features)


def _mine_chains(
    adjacency: Mapping[tuple[str, str], set[str]],
    relations: Sequence[str],
    entities: Sequence[str],
    max_length: int,
    max_features: int,
    discriminative: bool = True,
    scan_cap: int = 4000,
    state_cap: int = _RELATION_HIERARCHY_STATE_CAP,
) -> list[RelationChain]:
    """Mine exact chains with a topology cache and bounded-memory fallback."""
    if max_length <= 0:
        raise ValueError("max_len must be positive")
    if max_features < 0:
        raise ValueError("max_features must be non-negative")
    scan = _bounded_entity_scan(entities, scan_cap)
    if max_features == 0 or not scan or not relations:
        return []
    index = _RelationPathIndex(adjacency, scan, relations, state_cap)
    try:
        return _mine_chains_bidirectional(index, max_length, max_features, discriminative)
    except _HierarchyOverflow:
        return _mine_chains_streaming(
            index,
            scan,
            max_length,
            max_features,
            discriminative,
        )


def _mine_graph_chains(
    graph: _CompactRelationGraph,
    entity_names: Sequence[str],
    entity_ids: Sequence[int],
    relation_names: Sequence[str],
    max_length: int,
    max_features: int,
    discriminative: bool = True,
    scan_cap: int = 4000,
    state_cap: int = _RELATION_HIERARCHY_STATE_CAP,
) -> list[RelationChain]:
    """Mine against only the compact graph region reachable by the bounded scan."""
    if max_length <= 0:
        raise ValueError("max_len must be positive")
    if max_features < 0:
        raise ValueError("max_features must be non-negative")
    scan_names = _bounded_entity_scan(entity_names, scan_cap)
    scan_ids = _bounded_entity_scan(entity_ids, scan_cap)
    if max_features == 0 or not scan_names or not relation_names:
        return []
    local_adjacency = graph.mining_adjacency(scan_ids, relation_names, max_length)
    return _mine_chains(
        local_adjacency,
        relation_names,
        scan_names,
        max_length,
        max_features,
        discriminative,
        scan_cap,
        state_cap,
    )


def _trie_terminal(root: _ChainTrieNode, pattern: Sequence[int]) -> _ChainTrieNode:
    node = root
    for relation_id in pattern:
        child = node.children.get(relation_id)
        if child is None:
            child = _ChainTrieNode()
            node.children[relation_id] = child
        node = child
    return node


def _build_chain_trie(
    graph: _CompactRelationGraph,
    feature_patterns: Sequence[RelationChain],
    promotions: Sequence[PromotionSpec],
) -> _ChainTrieNode:
    root = _ChainTrieNode()
    for index, pattern in enumerate(feature_patterns):
        relation_ids = tuple(graph.relation_id(relation) for relation in pattern)
        _trie_terminal(root, relation_ids).feature_indices.append(index)
    for pattern, promoted_relation in promotions:
        relation_ids = tuple(graph.relation_id(relation) for relation in pattern)
        _trie_terminal(root, relation_ids).promoted_relations.append(promoted_relation)
    return root


def _route_sparse_reach(
    graph: _CompactRelationGraph,
    reach: EncodedSparseReach,
    relation_ids: Iterable[int],
) -> dict[int, EncodedSparseReach]:
    allowed = frozenset(relation_ids)
    routed: defaultdict[int, list[tuple[int, IdArray]]] = defaultdict(list)
    segment = graph.single_segment
    if segment is not None and all(frontier.size == 1 for _row_index, frontier in reach):
        for row_index, frontier in reach:
            segment.route_single_head(int(frontier[0]), row_index, allowed, routed)
        return {relation_id: tuple(rows) for relation_id, rows in routed.items()}

    for row_index, frontier in reach:
        row_targets: defaultdict[int, list[IdArray]] = defaultdict(list)
        for node_id in frontier:
            for relation_id, tails in graph.iter_edges(int(node_id)):
                if relation_id in allowed:
                    row_targets[relation_id].append(tails)
        for relation_id, arrays in row_targets.items():
            tails = _merge_id_arrays(arrays)
            if tails.size:
                routed[relation_id].append((row_index, tails))
    return {relation_id: tuple(rows) for relation_id, rows in routed.items()}


def _materialize_graph_chains(
    graph: _CompactRelationGraph,
    entity_ids: Sequence[int],
    feature_patterns: Sequence[RelationChain],
    promotions: Sequence[PromotionSpec] = (),
    *,
    binary: bool = False,
    chunk_rows: int = _RELATION_MATERIALIZATION_CHUNK_ROWS,
) -> tuple[list[NDArray[np.float64]], EncodedPromotionBuffer]:
    """Execute shared chain prefixes once per entity chunk.

    Promotion writes are staged so every chain observes the same round snapshot.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    values: list[NDArray[np.float64]] = [
        np.zeros(len(entity_ids), dtype=np.float64) for _pattern in feature_patterns
    ]
    additions: defaultdict[str, list[tuple[int, IdArray]]] = defaultdict(list)
    root = _build_chain_trie(graph, feature_patterns, promotions)
    if not root.children or not entity_ids:
        return values, dict(additions)

    def visit(
        node: _ChainTrieNode,
        reach: EncodedSparseReach,
        chunk_entities: Sequence[int],
        start: int,
    ) -> None:
        if node.feature_indices:
            row_indices = np.fromiter(
                (start + row_index for row_index, _frontier in reach),
                dtype=np.intp,
                count=len(reach),
            )
            for feature_index in node.feature_indices:
                if binary:
                    values[feature_index][row_indices] = 1.0
                else:
                    values[feature_index][row_indices] = np.fromiter(
                        (len(frontier) for _row_index, frontier in reach),
                        dtype=np.float64,
                        count=len(reach),
                    )

        for promoted_relation in node.promoted_relations:
            for row_index, tails in reach:
                additions[promoted_relation].append((chunk_entities[row_index], tails))

        child_reaches = _route_sparse_reach(graph, reach, node.children)
        for relation_id, child in sorted(node.children.items()):
            child_reach = child_reaches.get(relation_id)
            if child_reach is not None:
                visit(child, child_reach, chunk_entities, start)

    for start in range(0, len(entity_ids), chunk_rows):
        chunk_entities = entity_ids[start : start + chunk_rows]
        initial: EncodedSparseReach = tuple(
            (row_index, np.asarray((entity_id,), dtype=np.int64))
            for row_index, entity_id in enumerate(chunk_entities)
        )
        visit(root, initial, chunk_entities, start)
    return values, dict(additions)


def _materialize_chains(
    adjacency: Mapping[tuple[str, str], set[str]],
    entities: Sequence[str],
    feature_patterns: Sequence[RelationChain],
    promotions: Sequence[PromotionSpec] = (),
    *,
    binary: bool = False,
    chunk_rows: int = _RELATION_MATERIALIZATION_CHUNK_ROWS,
) -> tuple[list[NDArray[np.float64]], PromotionBuffer]:
    """Compatibility wrapper around the integer materialization engine."""
    graph = _CompactRelationGraph.from_adjacency(adjacency, entities)
    entity_ids = graph.node_ids(entities)
    values, encoded = _materialize_graph_chains(
        graph,
        entity_ids,
        feature_patterns,
        promotions,
        binary=binary,
        chunk_rows=chunk_rows,
    )
    decoded: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for relation, rows in encoded.items():
        for head_id, tails in rows:
            decoded[(relation, graph.node_name(head_id))].update(
                graph.node_name(int(tail_id)) for tail_id in tails
            )
    return values, dict(decoded)


def _remember_column(
    column: NDArray[np.float64],
    seen: dict[bytes, list[NDArray[np.float64]]],
) -> bool:
    """Record one column using bounded signatures with exact collision checks."""
    digest = hashlib.blake2b(memoryview(column).cast("B"), digest_size=16).digest()
    candidates = seen.setdefault(digest, [])
    if any(np.array_equal(column, candidate) for candidate in candidates):
        return False
    candidates.append(column)
    return True


def derive_features(
    triples: Iterable[Triple],
    entities: Iterable[str],
    max_len: int = 2,
    max_features: int = 64,
    rounds: int = 1,
    binary: bool = False,
    return_names: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], list[str]]:
    """Materialize deterministic relation-chain counts aligned to ``entities``.

    Every generated value corresponds to a chain-rule closure in ``FOLKernel``.
    Later rounds promote proven chains as facts so the next round can compose
    them into longer concepts.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    entity_list = list(entities)
    graph = _CompactRelationGraph.from_triples(triples, entity_list)
    entity_ids = graph.node_ids(entity_list)
    graph.freeze_nodes()

    columns: list[NDArray[np.float64]] = []
    names: list[str] = []
    seen: dict[bytes, list[NDArray[np.float64]]] = {}
    promoted: set[str] = set()
    for round_index in range(rounds):
        relation_list = sorted(graph.relation_names)
        feature_patterns = _mine_graph_chains(
            graph,
            entity_list,
            entity_ids,
            relation_list,
            max_len,
            max_features,
        )

        promotions: list[PromotionSpec] = []
        if round_index < rounds - 1:
            for pattern in _mine_graph_chains(
                graph,
                entity_list,
                entity_ids,
                relation_list,
                max_len,
                max_features,
                discriminative=False,
            ):
                promoted_relation = "@" + "→".join(pattern)
                if promoted_relation in promoted:
                    continue
                promoted.add(promoted_relation)
                promotions.append((pattern, promoted_relation))

        round_columns, additions = _materialize_graph_chains(
            graph,
            entity_ids,
            feature_patterns,
            promotions,
            binary=binary,
        )
        for pattern, column in zip(feature_patterns, round_columns, strict=True):
            if _remember_column(column, seen):
                columns.append(column)
                names.append("→".join(pattern))

        graph.add_promotions(additions)

    matrix = np.column_stack(columns) if columns else np.zeros((len(entity_list), 0), dtype=float)
    return (matrix, names) if return_names else matrix


__all__ = ["derive_features"]
