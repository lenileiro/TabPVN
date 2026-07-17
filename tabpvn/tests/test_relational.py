"""Exactness contracts for topology-indexed relational feature discovery."""

from collections import defaultdict

import numpy as np

from tabpvn.relational import (
    Triple,
    _chain_reach,
    _CompactRelationGraph,
    _materialize_chains,
    _mine_chains,
    _mine_chains_exhaustive,
    derive_features,
)


def _adjacency(triples: list[Triple]) -> defaultdict[tuple[str, str], set[str]]:
    adjacency: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for relation, head, tail in triples:
        adjacency[(relation, head)].add(tail)
    return adjacency


def test_compact_graph_deduplicates_edges_and_appends_promotion_segments():
    triples = [
        ("r", "u0", "a"),
        ("r", "u0", "a"),
        ("r", "u0", "b"),
        ("s", "a", "z"),
        ("noise", "x", "y"),
    ]
    graph = _CompactRelationGraph.from_triples(triples, ["u0", "u1"])
    u0, u1, a, b, z = graph.node_ids(["u0", "u1", "a", "b", "z"])

    assert graph.edge_count == 4
    np.testing.assert_array_equal(graph.targets(u0, graph.relation_id("r")), sorted([a, b]))
    local = graph.mining_adjacency([u0], ["r", "s"], max_depth=2)
    assert local == {
        ("r", "u0"): {"a", "b"},
        ("s", "a"): {"z"},
    }

    graph.add_promotions(
        {
            "r": [(u0, np.asarray([z], dtype=np.int64))],
            "@r": [(u1, np.asarray([a], dtype=np.int64))],
        }
    )
    np.testing.assert_array_equal(graph.targets(u0, graph.relation_id("r")), sorted([a, b, z]))
    np.testing.assert_array_equal(graph.targets(u1, graph.relation_id("@r")), [a])
    assert graph.edge_count == 6


def test_chunked_graph_spills_and_merges_readonly_segments():
    triples = [
        ("r", "u0", "a"),
        ("r", "u0", "a"),
        ("r", "u0", "b"),
        ("s", "a", "z"),
        ("r", "u1", "b"),
        ("s", "b", "z"),
    ]
    graph = _CompactRelationGraph.from_triples(
        triples,
        ["u0", "u1"],
        build_chunk_edges=2,
        resident_array_bytes=0,
    )
    u0, u1, a, b, z = graph.node_ids(["u0", "u1", "a", "b", "z"])

    assert graph.is_spilled
    assert graph.segment_count == 3
    assert graph.mapped_array_count >= 15
    np.testing.assert_array_equal(graph.targets(u0, graph.relation_id("r")), sorted([a, b]))
    assert graph.mining_adjacency([u0, u1], ["r", "s"], max_depth=2) == {
        ("r", "u0"): {"a", "b"},
        ("r", "u1"): {"b"},
        ("s", "a"): {"z"},
        ("s", "b"): {"z"},
    }

    graph.add_promotions(
        {
            "r": [(u0, np.asarray([z], dtype=np.int64))],
            "@r": [(u1, np.asarray([a, b], dtype=np.int64))],
        }
    )
    np.testing.assert_array_equal(graph.targets(u0, graph.relation_id("r")), sorted([a, b, z]))
    np.testing.assert_array_equal(graph.targets(u1, graph.relation_id("@r")), sorted([a, b]))
    assert graph.segment_count == 5
    assert graph.mapped_array_count >= 25


def test_hierarchy_matches_exhaustive_mining_on_sparse_graphs():
    relations = [f"r{index}" for index in range(4)]
    nodes = [f"n{index}" for index in range(16)]
    for seed in range(4):
        rng = np.random.default_rng(seed)
        triples = [
            (
                str(rng.choice(relations)),
                str(rng.choice(nodes)),
                str(rng.choice(nodes)),
            )
            for _ in range(45)
        ]
        adjacency = _adjacency(triples)
        for discriminative in (False, True):
            expected = _mine_chains_exhaustive(
                adjacency,
                relations,
                nodes[:8],
                max_length=4,
                max_features=19,
                discriminative=discriminative,
            )
            actual = _mine_chains(
                adjacency,
                relations,
                nodes[:8],
                max_length=4,
                max_features=19,
                discriminative=discriminative,
            )
            assert actual == expected


def test_hierarchy_overflow_uses_exact_streaming_fallback():
    triples = [
        ("owns", "u0", "a0"),
        ("owns", "u0", "a1"),
        ("owns", "u1", "a1"),
        ("links", "a0", "b0"),
        ("links", "a1", "b1"),
        ("flags", "b1", "risk"),
    ]
    adjacency = _adjacency(triples)
    relations = ["flags", "links", "owns"]
    expected = _mine_chains_exhaustive(adjacency, relations, ["u0", "u1", "u2"], 3, 12)
    for state_cap in (1, 5):
        actual = _mine_chains(
            adjacency,
            relations,
            ["u0", "u1", "u2"],
            3,
            12,
            state_cap=state_cap,
        )
        assert actual == expected


def test_mining_honors_the_relation_vocabulary():
    triples = [
        ("allowed", "u0", "a"),
        ("allowed", "u0", "b"),
        ("ignored", "u1", "a"),
    ]
    adjacency = _adjacency(triples)

    actual = _mine_chains(adjacency, ["allowed"], ["u0", "u1"], 2, 8)
    expected = _mine_chains_exhaustive(adjacency, ["allowed"], ["u0", "u1"], 2, 8)

    assert actual == expected == [("allowed",)]


def test_derived_columns_are_deterministic_exact_chain_witnesses():
    triples = [
        ("owns", "u0", "a0"),
        ("owns", "u0", "a1"),
        ("owns", "u1", "a1"),
        ("links", "a0", "risk"),
        ("links", "a1", "safe"),
    ]
    entities = ["u0", "u1", "u2"]
    matrix, names = derive_features(triples, entities, max_len=2, return_names=True)
    reversed_matrix, reversed_names = derive_features(
        list(reversed(triples)),
        entities,
        max_len=2,
        return_names=True,
    )

    assert names == reversed_names
    np.testing.assert_array_equal(matrix, reversed_matrix)
    adjacency = _adjacency(triples)
    for index, name in enumerate(names):
        pattern = tuple(name.split("→"))
        expected = [len(_chain_reach(adjacency, pattern, entity)) for entity in entities]
        np.testing.assert_array_equal(matrix[:, index], expected)


def test_chunked_materialization_matches_counts_binary_and_promotions():
    triples = [
        ("owns", "u0", "a0"),
        ("owns", "u0", "a1"),
        ("owns", "u1", "a1"),
        ("links", "a0", "b0"),
        ("links", "a1", "b1"),
        ("flags", "b1", "risk"),
    ]
    adjacency = _adjacency(triples)
    entities = ["u0", "u1", "u2"]
    patterns = [("owns",), ("owns", "links"), ("owns", "links", "flags")]
    promotions = [(patterns[1], "@owns→links")]

    counts, additions = _materialize_chains(
        adjacency,
        entities,
        patterns,
        promotions,
        chunk_rows=2,
    )
    binary, binary_additions = _materialize_chains(
        adjacency,
        entities,
        patterns,
        promotions,
        binary=True,
        chunk_rows=1,
    )

    for pattern, count_column, binary_column in zip(patterns, counts, binary, strict=True):
        expected = np.array([len(_chain_reach(adjacency, pattern, entity)) for entity in entities])
        np.testing.assert_array_equal(count_column, expected)
        np.testing.assert_array_equal(binary_column, expected > 0)
    expected_additions = {
        ("@owns→links", "u0"): {"b0", "b1"},
        ("@owns→links", "u1"): {"b1"},
    }
    assert additions == binary_additions == expected_additions


def test_incremental_rounds_preserve_deep_chain_discovery():
    triples = [
        ("r1", "e0", "a0"),
        ("r1", "e1", "a1"),
        ("r2", "a0", "b0"),
        ("r2", "a1", "b1"),
        ("r3", "b0", "risk"),
    ]
    entities = ["e0", "e1", "e2"]

    shallow = derive_features(triples, entities, max_len=2, rounds=1)
    deep, names = derive_features(triples, entities, max_len=2, rounds=2, return_names=True)

    target = np.array([1.0, 0.0, 0.0])
    assert not any(np.array_equal(shallow[:, index], target) for index in range(shallow.shape[1]))
    assert "@r1→r2→r3" in names
    np.testing.assert_array_equal(deep[:, names.index("@r1→r2→r3")], target)
