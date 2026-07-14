"""Deterministic, precision-gated relation-chain induction."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from core.kernel_fol import FOLKernel

Triple: TypeAlias = tuple[str, str, str]
EntityPair: TypeAlias = tuple[str, str]
RelationChain: TypeAlias = tuple[str, ...]
RuleScore: TypeAlias = tuple[float, float, int, RelationChain]
Adjacency: TypeAlias = Mapping[tuple[str, str], set[str]]

SCRATCH = Path(os.environ.get("TABPVN_DATA", "~/.cache/tabpvn")).expanduser()


def load_kg(name: str, inverses: bool = False) -> list[Triple]:
    """Load tab-separated head/relation/tail triples from the data directory."""
    if not name or Path(name).name != name:
        raise ValueError("knowledge-graph name must be a non-empty file stem")
    path = SCRATCH / f"kg_{name}.txt"
    base: list[Triple] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                head, relation, tail = parts
                base.append((relation, head, tail))
    if inverses:
        base.extend((relation + "_inv", tail, head) for relation, head, tail in tuple(base))
    return base


def mine_bodies(
    base: Iterable[Triple],
    positives: Iterable[EntityPair],
    relations: Iterable[str],
    max_len: int = 2,
    sample: int = 60,
    cap: int = 6000,
) -> Counter[RelationChain]:
    """Count relation sequences that connect known positive entity pairs."""
    if max_len <= 0 or sample <= 0 or cap <= 0:
        raise ValueError("max_len, sample, and cap must be positive")
    adjacency: dict[tuple[str, str], set[str]] = defaultdict(set)
    for relation, head, tail in base:
        adjacency[(relation, head)].add(tail)
    relation_list = sorted(relations)
    frequency: Counter[RelationChain] = Counter()
    for source, target in sorted(positives)[:sample]:
        frontier: list[tuple[str, RelationChain]] = [(source, ())]
        for _depth in range(max_len):
            next_frontier: list[tuple[str, RelationChain]] = []
            for node, path in frontier:
                for relation in relation_list:
                    for neighbor in sorted(adjacency.get((relation, node), ())):
                        sequence = path + (relation,)
                        if neighbor == target:
                            frequency[sequence] += 1
                        next_frontier.append((neighbor, sequence))
            frontier = next_frontier[:cap]
    return frequency


def verify(
    base: Iterable[Triple],
    body: Sequence[str],
    positives: Iterable[EntityPair],
    head: str = "t",
) -> tuple[float, float, int]:
    """Verify one candidate chain and return precision, recall, and support."""
    rule = (
        (head, "X0", f"X{len(body)}"),
        [(body[index], f"X{index}", f"X{index + 1}") for index in range(len(body))],
    )
    facts, _ = FOLKernel([rule]).closure(base)
    derived = {(fact[1], fact[2]) for fact in facts if len(fact) == 3 and fact[0] == head}
    if not derived:
        return 0.0, 0.0, 0
    positive_set = set(positives)
    if not positive_set:
        return 0.0, 0.0, len(derived)
    true_positive_count = len(derived & positive_set)
    return (
        true_positive_count / len(derived),
        true_positive_count / len(positive_set),
        len(derived),
    )


def _derive_chain(
    adjacency: Adjacency,
    first_hop: Mapping[str, set[EntityPair]],
    body: Sequence[str],
) -> set[EntityPair]:
    if not body:
        return set()
    current = set(first_hop.get(body[0], set()))
    for relation in body[1:]:
        next_pairs = {
            (source, target)
            for source, intermediate in current
            for target in adjacency.get((relation, intermediate), ())
        }
        current = next_pairs
        if not current:
            break
    return current


def _score_chain(
    adjacency: Adjacency,
    first_hop: Mapping[str, set[EntityPair]],
    body: Sequence[str],
    positives: set[EntityPair],
) -> tuple[float, float, int]:
    derived = _derive_chain(adjacency, first_hop, body)
    if not derived or not positives:
        return 0.0, 0.0, len(derived)
    true_positive_count = len(derived & positives)
    return (
        true_positive_count / len(derived),
        true_positive_count / len(positives),
        len(derived),
    )


def induce(
    base: Sequence[Triple],
    rels: Iterable[str],
    target: str,
    tau: float = 0.7,
    min_sup: int = 8,
    topk: int = 3,
) -> tuple[set[EntityPair], list[RuleScore]]:
    """Mine and retain non-circular chains meeting precision and support gates."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    if min_sup <= 0 or topk < 0:
        raise ValueError("min_sup must be positive and topk must be non-negative")

    positives = {(head, tail) for relation, head, tail in base if relation == target}
    if len(positives) < min_sup or topk == 0:
        return positives, []
    frequency = mine_bodies(base, positives, rels)
    adjacency: dict[tuple[str, str], set[str]] = defaultdict(set)
    first_hop: dict[str, set[EntityPair]] = defaultdict(set)
    for relation, head, tail in base:
        adjacency[(relation, head)].add(tail)
        first_hop[relation].add((head, tail))

    seen: set[RelationChain] = set()
    retained: list[RuleScore] = []
    inverse_target = target + "_inv"
    ranked = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[:40]
    for sequence, _frequency in ranked:
        if sequence == (target,) or target in sequence or inverse_target in sequence or sequence in seen:
            continue
        seen.add(sequence)
        precision, recall, support = _score_chain(
            adjacency,
            first_hop,
            sequence,
            positives,
        )
        if precision >= tau and support >= min_sup and recall >= 0.1:
            retained.append((precision, recall, support, sequence))
    retained.sort(key=lambda rule: (-rule[0], -rule[1]))
    return positives, retained[:topk]


__all__ = ["induce", "load_kg", "mine_bodies", "verify"]
