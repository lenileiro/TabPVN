"""Exact speed audit for shared-prefix relational feature materialization."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from itertools import product

import numpy as np

from tabpvn.relational import (
    _chain_reach,
    _CompactRelationGraph,
    _materialize_graph_chains,
)


def _typed_graph(source_count: int, max_length: int):
    sources = [f"s{index}" for index in range(source_count)]
    adjacency = defaultdict(set)
    for index, source in enumerate(sources):
        node = source
        for depth in range(max_length):
            relation = f"r{2 * depth + ((index >> depth) & 1)}"
            tail = f"l{depth + 1}_{index}"
            adjacency[(relation, node)].add(tail)
            node = tail
    patterns = [
        tuple(f"r{2 * level + bit}" for level, bit in enumerate(bits))
        for depth in range(1, max_length + 1)
        for bits in product((0, 1), repeat=depth)
    ]
    return adjacency, sources, patterns


def audit(source_count: int = 20_000, max_length: int = 4):
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    adjacency, sources, patterns = _typed_graph(source_count, max_length)

    started = time.perf_counter()
    independent = [
        np.fromiter(
            (len(_chain_reach(adjacency, pattern, source)) for source in sources),
            dtype=np.float64,
            count=source_count,
        )
        for pattern in patterns
    ]
    independent_seconds = time.perf_counter() - started

    triples = ((relation, head, tail) for (relation, head), tails in adjacency.items() for tail in tails)
    started = time.perf_counter()
    graph = _CompactRelationGraph.from_triples(triples, sources)
    entity_ids = graph.node_ids(sources)
    graph.freeze_nodes()
    compact_build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    shared, additions = _materialize_graph_chains(graph, entity_ids, patterns)
    shared_seconds = time.perf_counter() - started
    exact_match = not additions and all(
        np.array_equal(actual, expected) for actual, expected in zip(shared, independent, strict=True)
    )
    return {
        "sources": source_count,
        "patterns": len(patterns),
        "max_length": max_length,
        "independent_seconds": independent_seconds,
        "compact_build_seconds": compact_build_seconds,
        "shared_trie_seconds": shared_seconds,
        "speedup_execution": independent_seconds / shared_seconds,
        "speedup_including_build": independent_seconds / (compact_build_seconds + shared_seconds),
        "compact_array_bytes": graph.array_bytes,
        "compact_bytes_per_edge": graph.array_bytes / graph.edge_count,
        "exact_match": exact_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(audit(args.sources, args.max_length), indent=2))


if __name__ == "__main__":
    main()
