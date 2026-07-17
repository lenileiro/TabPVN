"""Bounded exactness and speed audit for relational hierarchy discovery."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from tabpvn.relational import (
    _meeting_nodes,
    _mine_chains,
    _mine_chains_exhaustive,
    _RelationPathIndex,
)


def _typed_graph(source_count: int, max_length: int):
    relations = [f"r{index}" for index in range(2 * max_length + 2)]
    sources = [f"s{index}" for index in range(source_count)]
    adjacency = defaultdict(set)
    for index, source in enumerate(sources):
        node = source
        for depth in range(max_length):
            relation = f"r{2 * depth + ((index >> depth) & 1)}"
            tail = f"l{depth + 1}_{index}"
            adjacency[(relation, node)].add(tail)
            node = tail

    # These real relations increase the old Cartesian search space but are
    # disconnected from every scanned source.
    for index in range(min(200, source_count)):
        relation = relations[-2 + index % 2]
        adjacency[(relation, f"x{index}")].add(f"y{index}")
    return adjacency, relations, sources


def audit(source_count: int = 600, max_length: int = 4, exhaustive_cap: int = 20_000):
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    adjacency, relations, sources = _typed_graph(source_count, max_length)
    candidates = sum(len(relations) ** depth for depth in range(1, max_length + 1))

    started = time.perf_counter()
    selected = _mine_chains(adjacency, relations, sources, max_length, max_features=64)
    hierarchy_seconds = time.perf_counter() - started

    index = _RelationPathIndex(adjacency, sources, relations)
    forward = index.forward_layers((max_length + 1) // 2)
    index.suffix_layers(max_length // 2, _meeting_nodes(forward))

    exhaustive_seconds = None
    exact_match = None
    speedup = None
    if candidates <= exhaustive_cap:
        started = time.perf_counter()
        reference = _mine_chains_exhaustive(
            adjacency,
            relations,
            sources,
            max_length,
            max_features=64,
        )
        exhaustive_seconds = time.perf_counter() - started
        exact_match = selected == reference
        speedup = exhaustive_seconds / hierarchy_seconds

    return {
        "sources": source_count,
        "relations": len(relations),
        "max_length": max_length,
        "cartesian_candidate_strings": candidates,
        "topology_valid_prefixes": sum(len(layer) for layer in forward[1:]),
        "index_endpoint_memberships": index.state_size,
        "selected_chains": len(selected),
        "hierarchy_seconds": hierarchy_seconds,
        "exhaustive_seconds": exhaustive_seconds,
        "speedup": speedup,
        "exact_match": exact_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=600)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--exhaustive-cap", type=int, default=20_000)
    args = parser.parse_args()
    print(json.dumps(audit(args.sources, args.max_length, args.exhaustive_cap), indent=2))


if __name__ == "__main__":
    main()
