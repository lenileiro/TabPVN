"""Bounded exactness audit for chunked memory-mapped relational segments."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from benchmark.relational_materialization_audit import _typed_graph
from tabpvn.relational import _CompactRelationGraph, _materialize_graph_chains


def _triples(adjacency):
    return ((relation, head, tail) for (relation, head), tails in adjacency.items() for tail in tails)


def audit(source_count: int = 5_000, max_length: int = 4, chunk_edges: int = 2_500):
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if chunk_edges <= 0:
        raise ValueError("chunk_edges must be positive")
    adjacency, sources, patterns = _typed_graph(source_count, max_length)

    started = time.perf_counter()
    resident = _CompactRelationGraph.from_triples(_triples(adjacency), sources)
    resident_ids = resident.node_ids(sources)
    resident.freeze_nodes()
    resident_build_seconds = time.perf_counter() - started
    expected, _additions = _materialize_graph_chains(resident, resident_ids, patterns)

    started = time.perf_counter()
    mapped = _CompactRelationGraph.from_triples(
        _triples(adjacency),
        sources,
        build_chunk_edges=chunk_edges,
        resident_array_bytes=0,
    )
    mapped_ids = mapped.node_ids(sources)
    mapped.freeze_nodes()
    mapped_build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    actual, additions = _materialize_graph_chains(mapped, mapped_ids, patterns)
    mapped_execute_seconds = time.perf_counter() - started
    exact_match = not additions and all(
        np.array_equal(actual_column, expected_column)
        for actual_column, expected_column in zip(actual, expected, strict=True)
    )
    return {
        "sources": source_count,
        "physical_edges": mapped.edge_count,
        "patterns": len(patterns),
        "chunk_edges": chunk_edges,
        "resident_build_seconds": resident_build_seconds,
        "mapped_build_seconds": mapped_build_seconds,
        "mapped_execute_seconds": mapped_execute_seconds,
        "segments": mapped.segment_count,
        "mapped_arrays": mapped.mapped_array_count,
        "mapped_edge_array_bytes": mapped.array_bytes,
        "spilled": mapped.is_spilled,
        "exact_match": exact_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=5_000)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--chunk-edges", type=int, default=2_500)
    args = parser.parse_args()
    print(json.dumps(audit(args.sources, args.max_length, args.chunk_edges), indent=2))


if __name__ == "__main__":
    main()
