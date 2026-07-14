"""Small trusted Datalog kernel with proof-carrying derivations.

The kernel supports n-ary facts, stratified negation, comparisons, arithmetic,
and explicit inequality. A proof embeds each Horn rule it used so verification
does not require the original kernel instance.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, TypeAlias

Term: TypeAlias = Any
Atom: TypeAlias = tuple[Any, ...]
Fact: TypeAlias = tuple[Any, ...]
Binding: TypeAlias = dict[str, Any]
Rule: TypeAlias = tuple[Atom, Sequence[Atom]]
ProofNode: TypeAlias = tuple[Fact, str | Rule, list["ProofNode"]]
ProvenanceEntry: TypeAlias = tuple[str | int, list[Fact]]
Provenance: TypeAlias = dict[Fact, ProvenanceEntry]

_COMPARISONS: dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
    "in": lambda value, container: value in container,
    "not in": lambda value, container: value not in container,
}
_ARITHMETIC: dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}
_SIDE_CONDITIONS = {"not", "cmp", "is", "neq"}
_MAX_PROOF_DEPTH = 512


def _is_var(term: Term) -> bool:
    return isinstance(term, str) and term[:1].isupper()


def _resolve(term: Term, binding: Mapping[str, Any]) -> Any:
    return binding[term] if _is_var(term) else term


def _evaluate(expression: Term, binding: Mapping[str, Any]) -> Any:
    if isinstance(expression, tuple):
        if len(expression) != 3 or expression[0] not in _ARITHMETIC:
            raise ValueError(f"invalid arithmetic expression: {expression!r}")
        return _ARITHMETIC[expression[0]](
            _evaluate(expression[1], binding),
            _evaluate(expression[2], binding),
        )
    return binding[expression] if _is_var(expression) else expression


class FOLKernel:
    """Evaluate stratified Horn rules and retain derivation provenance."""

    def __init__(self, rules: Iterable[Rule]):
        self.rules = list(rules)
        self.stratum = self._stratify()

    def _idb(self) -> set[Any]:
        return {rule[0][0] for rule in self.rules}

    def _stratify(self) -> dict[Any, int]:
        intensional = self._idb()
        strata = dict.fromkeys(intensional, 0)
        for _iteration in range(len(intensional) + 1):
            changed = False
            for head, body in self.rules:
                head_predicate = head[0]
                for atom in body:
                    negative = atom[0] == "not"
                    predicate = atom[1][0] if negative else atom[0]
                    if predicate not in intensional:
                        continue
                    required = strata[predicate] + int(negative)
                    if required > strata[head_predicate]:
                        strata[head_predicate] = required
                        changed = True
            if not changed:
                break
        for head, body in self.rules:
            for atom in body:
                if atom[0] == "not" and atom[1][0] in intensional and strata[atom[1][0]] >= strata[head[0]]:
                    raise ValueError(f"not stratifiable: negation on {atom[1][0]!r} at or above {head[0]!r}")
        return strata

    @staticmethod
    def _add_index(index: dict[Any, Any], fact: Fact) -> None:
        predicate_index = index.setdefault(fact[0], {})
        for position in range(1, len(fact)):
            predicate_index.setdefault(position, {}).setdefault(fact[position], []).append(fact)

    @staticmethod
    def _candidates(
        atom: Atom,
        by_predicate: Mapping[Any, Sequence[Fact]],
        index: Mapping[Any, Any],
        binding: Mapping[str, Any],
    ) -> Sequence[Fact]:
        predicate = atom[0]
        best: Sequence[Fact] | None = None
        predicate_index = index.get(predicate, {})
        for position in range(1, len(atom)):
            term = atom[position]
            if _is_var(term):
                if term not in binding:
                    continue
                value = binding[term]
            else:
                value = term
            bucket = predicate_index.get(position, {}).get(value, ())
            if best is None or len(bucket) < len(best):
                best = bucket
        return by_predicate.get(predicate, ()) if best is None else best

    def _match(
        self,
        body: Sequence[Atom],
        by_predicate: Mapping[Any, Sequence[Fact]],
        index: Mapping[Any, Any],
        facts: set[Fact],
        binding: Binding,
        premises: list[Fact],
    ) -> Iterator[tuple[Binding, list[Fact]]]:
        if not body:
            yield dict(binding), list(premises)
            return

        atom = body[0]
        rest = body[1:]
        tag = atom[0]
        if tag == "not":
            grounded = (atom[1][0],) + tuple(_resolve(term, binding) for term in atom[1][1:])
            if grounded not in facts:
                yield from self._match(rest, by_predicate, index, facts, binding, premises)
            return
        if tag == "cmp":
            comparison = _COMPARISONS.get(atom[1])
            if comparison is None:
                raise ValueError(f"unsupported comparison operator: {atom[1]!r}")
            if comparison(_resolve(atom[2], binding), _resolve(atom[3], binding)):
                yield from self._match(rest, by_predicate, index, facts, binding, premises)
            return
        if tag == "is":
            updated = dict(binding)
            updated[atom[1]] = _evaluate(atom[2], binding)
            yield from self._match(rest, by_predicate, index, facts, updated, premises)
            return
        if tag == "neq":
            if binding.get(atom[1]) != binding.get(atom[2]):
                yield from self._match(rest, by_predicate, index, facts, binding, premises)
            return

        for fact in self._candidates(atom, by_predicate, index, binding):
            if len(fact) != len(atom):
                continue
            updated = dict(binding)
            matches = True
            for term, value in zip(atom[1:], fact[1:], strict=True):
                if _is_var(term):
                    if term in updated and updated[term] != value:
                        matches = False
                        break
                    updated[term] = value
                elif term != value:
                    matches = False
                    break
            if matches:
                yield from self._match(
                    rest,
                    by_predicate,
                    index,
                    facts,
                    updated,
                    premises + [fact],
                )

    def closure(self, base_facts: Iterable[Fact]) -> tuple[set[Fact], Provenance]:
        """Evaluate each stratum to a fixpoint and return facts plus provenance."""
        base = list(base_facts)
        facts = set(base)
        provenance: Provenance = {fact: ("base", []) for fact in base}
        by_predicate: dict[Any, list[Fact]] = {}
        index: dict[Any, Any] = {}
        for fact in facts:
            by_predicate.setdefault(fact[0], []).append(fact)
            self._add_index(index, fact)

        for stratum in range(max(self.stratum.values(), default=0) + 1):
            layer = [
                (rule_index, rule)
                for rule_index, rule in enumerate(self.rules)
                if self.stratum[rule[0][0]] == stratum
            ]
            changed = True
            while changed:
                changed = False
                for rule_index, (head, body) in layer:
                    for binding, premises in self._match(
                        body,
                        by_predicate,
                        index,
                        facts,
                        {},
                        [],
                    ):
                        new_fact = (head[0],) + tuple(_resolve(term, binding) for term in head[1:])
                        if new_fact in facts:
                            continue
                        facts.add(new_fact)
                        provenance[new_fact] = (rule_index, premises)
                        by_predicate.setdefault(new_fact[0], []).append(new_fact)
                        self._add_index(index, new_fact)
                        changed = True
        return facts, provenance

    def proof(self, fact: Fact, provenance: Provenance) -> ProofNode:
        """Build a self-contained proof tree for one derived fact."""
        rule_reference, premises = provenance[fact]
        rule: str | Rule = "base" if rule_reference == "base" else self.rules[int(rule_reference)]
        return (
            fact,
            rule,
            [self.proof(premise, provenance) for premise in premises],
        )


def _same_rule(left: Any, right: Any) -> bool:
    """Compare Horn clauses while tolerating list/tuple representation."""

    def normalize(value: Any) -> Any:
        return tuple(normalize(item) for item in value) if isinstance(value, (list, tuple)) else value

    return normalize(left) == normalize(right)


def _unify_premises(positive: Sequence[Atom], premises: Sequence[Fact]) -> Binding | None:
    if len(positive) != len(premises):
        return None
    binding: Binding = {}
    for atom, premise in zip(positive, premises, strict=True):
        if premise[0] != atom[0] or len(premise) != len(atom):
            return None
        for term, value in zip(atom[1:], premise[1:], strict=True):
            if _is_var(term):
                if term in binding and binding[term] != value:
                    return None
                binding[term] = value
            elif term != value:
                return None
    return binding


def _side_conditions_hold(body: Sequence[Atom], binding: Binding) -> bool:
    for atom in body:
        if atom[0] == "cmp":
            comparison = _COMPARISONS.get(atom[1])
            if comparison is None or not comparison(
                _resolve(atom[2], binding),
                _resolve(atom[3], binding),
            ):
                return False
        elif atom[0] == "is":
            binding[atom[1]] = _evaluate(atom[2], binding)
        elif atom[0] == "neq" and binding.get(atom[1]) == binding.get(atom[2]):
            return False
    return True


def _verify_proof(
    node: Any,
    base_facts: set[Fact] | None,
    trusted_rules: tuple[Rule, ...] | None,
    active: set[int],
    depth: int,
) -> bool:
    if isinstance(node, dict):
        node = node.get("proof")
    if node == "root":
        return True
    if node is None:
        return False
    if depth > _MAX_PROOF_DEPTH or not isinstance(node, (list, tuple)) or len(node) != 3:
        return False

    identity = id(node)
    if identity in active:
        return False
    active.add(identity)
    try:
        fact, rule, children = node
        fact = tuple(fact)
        if not isinstance(children, (list, tuple)):
            return False
        if rule == "base":
            return not children and (base_facts is None or fact in base_facts)
        if not isinstance(rule, (list, tuple)) or len(rule) != 2:
            return False
        if not all(_verify_proof(child, base_facts, trusted_rules, active, depth + 1) for child in children):
            return False

        head, body = rule
        if not isinstance(head, (list, tuple)) or not isinstance(body, (list, tuple)):
            return False
        positive = [atom for atom in body if atom[0] not in _SIDE_CONDITIONS]
        premises = [child[0] for child in children]
        trusted = trusted_rules is not None and any(
            _same_rule(rule, candidate) for candidate in trusted_rules
        )
        if not positive and not trusted:
            return False
        if trusted_rules is not None and not trusted:
            return False
        binding = _unify_premises(positive, premises)
        if binding is None or not _side_conditions_hold(body, binding):
            return False
        grounded_head = (head[0],) + tuple(_resolve(term, binding) for term in head[1:])
        return grounded_head == fact
    finally:
        active.remove(identity)


def check_proof(
    node: Any,
    base_facts: Iterable[Fact] | None = None,
    rules: Iterable[Rule] | None = None,
) -> bool:
    """Independently verify a proof tree and fail closed on malformed input."""
    try:
        declared_facts = None if base_facts is None else {tuple(fact) for fact in base_facts}
        trusted_rules = None if rules is None else tuple(rules)
        return _verify_proof(node, declared_facts, trusted_rules, set(), 0)
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def show(node: ProofNode, depth: int = 0) -> None:
    """Print a compact human-readable proof tree."""
    fact, rule, children = node
    arguments = ",".join(str(argument) for argument in fact[1:])
    tag = "fact" if rule == "base" else f"rule: {rule[0][0]} :- {', '.join(atom[0] for atom in rule[1])}"
    print("   " * depth + f"{fact[0]}({arguments})   [{tag}]")
    for child in children:
        show(child, depth + 1)


__all__ = ["FOLKernel", "check_proof", "show"]
