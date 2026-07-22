"""Public proof replies and opt-in verification artifacts.

The default response is intentionally implementation-neutral. Detailed model
derivations live in a separate audit artifact so applications do not expose
tree layout, score arithmetic, or verifier internals to their users.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.kernel_fol import check_proof as check_kernel_proof
from tabpvn.attestations import (
    SignedTargetAttestation,
    TargetAttestation,
    attestation_payload,
    is_signed_attestation,
    verify_attestation_payload,
)

PROOF_SCHEMA = "tabpvn.proof/3"
PROOF_ARTIFACT_SCHEMA = "tabpvn.proof-artifact/1"
_SIDE_CONDITIONS = {"not", "cmp", "is", "neq"}
_POSTERIOR_KINDS = {
    "categorical_dirichlet_posterior",
    "categorical_dirichlet_posterior_pool",
    "numeric_interval_dirichlet_posterior",
    "numeric_interval_dirichlet_posterior_pool",
}
_AFFINE_DECISION_KIND = "affine_logit_decision"
_PUBLIC_OPERATORS = {
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
    "==": "eq",
    "=": "eq",
    "!=": "neq",
    "in": "in",
    "not in": "not_in",
}


def is_proof_response(value: Any) -> bool:
    """Return whether *value* uses the public TabPVN proof-response schema."""
    return isinstance(value, Mapping) and value.get("schema") == PROOF_SCHEMA


def is_proof_artifact(value: Any) -> bool:
    """Return whether *value* is an opt-in independently checkable artifact."""
    return isinstance(value, Mapping) and value.get("schema") == PROOF_ARTIFACT_SCHEMA


def _plain_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            pass
    return value


def _format_value(value: Any) -> str:
    value = _plain_scalar(value)
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    return str(value)


def _feature_name(index: Any, feature_names: Sequence[Any] | None) -> str:
    try:
        position = int(index)
    except (TypeError, ValueError):
        return f"feature[{index}]"
    if feature_names is not None and 0 <= position < len(feature_names):
        return str(feature_names[position])
    return f"feature[{position}]"


def _node_fact(node: Any) -> tuple[Any, ...] | None:
    if not isinstance(node, (list, tuple)) or len(node) != 3:
        return None
    fact = node[0]
    return tuple(fact) if isinstance(fact, (list, tuple)) else None


def _is_variable(term: Any) -> bool:
    return isinstance(term, str) and term[:1].isupper()


def _resolve(term: Any, binding: Mapping[str, Any]) -> Any:
    return binding.get(term, term) if _is_variable(term) else term


def _category_labels(columns: Sequence[Any], feature_names: Sequence[Any] | None) -> tuple[str, list[str]]:
    names = [_feature_name(column, feature_names) for column in columns]
    split = [name.split("=", 1) for name in names]
    if split and all(len(parts) == 2 and parts[0] == split[0][0] for parts in split):
        return split[0][0], [parts[1] for parts in split]
    return "category", names


def _category_condition_record(
    columns: Sequence[Any],
    operator: str,
    levels: Any,
    observed: Any,
    feature_names: Sequence[Any] | None,
) -> dict[str, Any]:
    label, values = _category_labels(columns, feature_names)
    allowed_indices = list(levels) if isinstance(levels, (list, tuple)) else [levels]
    allowed = [values[int(level)] for level in allowed_indices if 0 <= int(level) < len(values)]
    try:
        observed_index = int(observed)
    except (TypeError, ValueError):
        observed_index = -1
    observed_value = values[observed_index] if 0 <= observed_index < len(values) else "unseen"
    return {
        "feature": label,
        "operator": _PUBLIC_OPERATORS.get(operator, operator),
        "value": allowed,
        "observed": observed_value,
    }


def _proof_context(
    node: Any,
) -> tuple[Sequence[Any], dict[str, Any], dict[str, tuple[str, Any, Any]]] | None:
    if not isinstance(node, (list, tuple)) or len(node) != 3:
        return None
    _fact, rule, children = node
    if rule == "base" or not isinstance(rule, (list, tuple)) or len(rule) != 2:
        return None
    body = rule[1]
    if not isinstance(body, (list, tuple)) or not isinstance(children, (list, tuple)):
        return None
    positive = [atom for atom in body if isinstance(atom, (list, tuple)) and atom[0] not in _SIDE_CONDITIONS]
    if len(positive) != len(children):
        return None

    binding: dict[str, Any] = {}
    sources: dict[str, tuple[str, Any, Any]] = {}
    for atom, child in zip(positive, children, strict=True):
        child_fact = _node_fact(child)
        if child_fact is None or len(atom) != len(child_fact):
            return None
        for term, value in zip(atom[1:], child_fact[1:], strict=True):
            if _is_variable(term):
                binding[term] = value
        if atom[0] == "feat" and len(atom) == 4 and _is_variable(atom[3]):
            sources[atom[3]] = ("numeric", atom[2], child_fact[3])
        elif atom[0] == "cat" and len(atom) == 4 and _is_variable(atom[3]):
            sources[atom[3]] = ("category", atom[2], child_fact[3])
    return body, binding, sources


def proof_condition_records(
    node: Any,
    feature_names: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Return typed threshold/category conditions embedded in one proof."""
    if isinstance(node, Mapping):
        node = node.get("proof")
    if node == "root":
        return []
    context = _proof_context(node)
    if context is None:
        return []
    body, binding, sources = context
    conditions: list[dict[str, Any]] = []
    for atom in body:
        if not isinstance(atom, (list, tuple)) or len(atom) != 4 or atom[0] != "cmp":
            continue
        operator, left, right = str(atom[1]), atom[2], atom[3]
        source = sources.get(left) if _is_variable(left) else None
        if source is None:
            continue
        kind, feature, observed = source
        threshold = _resolve(right, binding)
        if kind == "category" and isinstance(feature, (list, tuple)):
            condition = _category_condition_record(
                feature,
                operator,
                threshold,
                observed,
                feature_names,
            )
        else:
            condition = {
                "feature": _feature_name(feature, feature_names),
                "operator": _PUBLIC_OPERATORS.get(operator, operator),
                "value": _plain_scalar(threshold),
                "observed": _plain_scalar(observed),
            }
        if condition not in conditions:
            conditions.append(condition)
    return conditions


def proof_conditions(node: Any, feature_names: Sequence[Any] | None = None) -> list[str]:
    """Render proof conditions for the detailed human-readable artifact."""
    records = proof_condition_records(node, feature_names)
    if not records:
        candidate = node.get("proof") if isinstance(node, Mapping) else node
        return ["always applies"] if candidate == "root" else []
    display = {
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "eq": "=",
        "neq": "!=",
    }
    conditions: list[str] = []
    for record in records:
        operator = record["operator"]
        if operator in {"in", "not_in"}:
            relation = "is one of" if operator == "in" else "is not one of"
            values = ", ".join(_format_value(value) for value in record["value"])
            statement = f"{record['feature']} {relation} {{{values}}}"
        else:
            statement = (
                f"{record['feature']} {display.get(operator, operator)} {_format_value(record['value'])}"
            )
        conditions.append(f"{statement} (observed {_format_value(record['observed'])})")
    return conditions


def _term_effect(term: Mapping[str, Any]) -> tuple[str, float]:
    if "contribution" in term:
        return "prediction", float(term["contribution"])
    if "logit_contribution" in term:
        return "winning class score", float(term["logit_contribution"])
    return "prediction", 0.0


def _additive_evidence(
    payload: Mapping[str, Any], feature_names: Sequence[Any] | None
) -> list[dict[str, Any]]:
    terms = payload.get("terms_shown", [])
    if not isinstance(terms, (list, tuple)):
        return []
    groups: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    for term in terms:
        if not isinstance(term, Mapping):
            continue
        node = term.get("proof")
        conditions = tuple(proof_conditions(node, feature_names))
        condition_records = proof_condition_records(node, feature_names)
        metric, contribution = _term_effect(term)
        key = conditions, metric
        group = groups.setdefault(
            key,
            {
                "conditions": list(conditions),
                "condition_records": condition_records,
                "metric": metric,
                "contribution": 0.0,
                "stages": [],
                "regions": [],
                "verified": True,
                "count": 0,
            },
        )
        group["contribution"] += contribution
        group["count"] += 1
        if term.get("stage") is not None:
            group["stages"].append(int(term["stage"]))
        group["regions"].append(term.get("region", "root"))
        group["verified"] = bool(group["verified"] and check_kernel_proof(node))

    evidence: list[dict[str, Any]] = []
    for index, group in enumerate(groups.values(), start=1):
        count = int(group["count"])
        if count == 1:
            stage = group["stages"][0] if group["stages"] else None
            region = group["regions"][0]
            label = f"Stage {stage}, region {region}" if stage is not None else f"Region {region}"
            effect = f"adds {float(group['contribution']):+.6g} to the {group['metric']}"
        else:
            label = f"{count} matching stages"
            effect = (
                f"add {float(group['contribution']):+.6g} in total to the {group['metric']} "
                f"across {count} stages"
            )
        item: dict[str, Any] = {
            "id": f"prediction-{index}",
            "kind": "decision_region",
            "label": label,
            "conditions": group["conditions"],
            "condition_records": group["condition_records"],
            "effect": effect,
            "verified": bool(group["verified"]),
        }
        if group["stages"]:
            item["stages"] = group["stages"]
        evidence.append(item)
    return evidence


def _posterior_condition(condition: Mapping[str, Any]) -> str:
    if condition.get("kind") != "numeric_interval":
        return f"{condition.get('name', 'category')} = {_format_value(condition.get('level'))}"
    name = str(condition.get("name", "feature"))
    lower, upper = condition.get("lower"), condition.get("upper")
    observed = condition.get("observed")
    if lower is None:
        interval = f"{name} < {_format_value(upper)}"
    elif upper is None:
        interval = f"{name} >= {_format_value(lower)}"
    else:
        interval = f"{_format_value(lower)} <= {name} < {_format_value(upper)}"
    return f"{interval} (observed {_format_value(observed)})"


def _posterior_condition_record(condition: Mapping[str, Any]) -> dict[str, Any]:
    feature = str(condition.get("name", "feature"))
    if condition.get("kind") != "numeric_interval":
        value = _plain_scalar(condition.get("level"))
        return {
            "feature": feature,
            "operator": "eq",
            "value": value,
            "observed": value,
        }
    lower, upper = condition.get("lower"), condition.get("upper")
    if lower is None:
        operator, value = "lt", _plain_scalar(upper)
    elif upper is None:
        operator, value = "gte", _plain_scalar(lower)
    else:
        operator = "between"
        value = {
            "lower": _plain_scalar(lower),
            "upper": _plain_scalar(upper),
            "lower_inclusive": bool(condition.get("lower_inclusive", True)),
            "upper_inclusive": bool(condition.get("upper_inclusive", False)),
        }
    return {
        "feature": feature,
        "operator": operator,
        "value": value,
        "observed": _plain_scalar(condition.get("observed")),
    }


def _posterior_evidence(payload: Mapping[str, Any], verified: bool) -> list[dict[str, Any]]:
    raw_conditions = payload.get("conditions", [])
    conditions = [
        _posterior_condition(condition) for condition in raw_conditions if isinstance(condition, Mapping)
    ]
    condition_records = [
        _posterior_condition_record(condition)
        for condition in raw_conditions
        if isinstance(condition, Mapping)
    ]
    base_prediction = _format_value(payload.get("base_prediction"))
    prediction = _format_value(payload.get("prediction"))
    support = payload.get("support")
    support_text = f" from {int(support)} matching rows" if isinstance(support, (int, float)) else ""
    return [
        {
            "id": "prediction-1",
            "kind": "finite_count_posterior",
            "label": "Finite count evidence",
            "conditions": conditions,
            "condition_records": condition_records,
            "effect": f"changes class {base_prediction} to {prediction}{support_text}",
            "verified": verified,
        }
    ]


def _affine_evidence(
    payload: Mapping[str, Any],
    feature_names: Sequence[Any] | None,
    verified: bool,
) -> list[dict[str, Any]]:
    base_proof = payload.get("base_proof")
    evidence = _prediction_evidence(base_proof, feature_names, verified)
    try:
        classes = list(payload["classes"])
        predicted_index = classes.index(payload["prediction"])
        model_index = int(payload["class_order"][predicted_index])
        coefficients = payload["coefficients"]
        values = payload["input"]
        if len(classes) == 2:
            direction = 1.0 if model_index == 1 else -1.0
            contributions = [
                direction * float(value) * float(coefficient)
                for value, coefficient in zip(values, coefficients[0], strict=True)
            ]
        else:
            contributions = [
                float(value) * float(coefficient)
                for value, coefficient in zip(values, coefficients[model_index], strict=True)
            ]
        strongest = sorted(
            range(len(contributions)),
            key=lambda index: abs(contributions[index]),
            reverse=True,
        )[:5]
        terms = ", ".join(
            f"{_feature_name(index, feature_names)} {_format_value(contributions[index])}"
            for index in strongest
            if contributions[index] != 0.0
        )
        base_probability = float(payload["base_probability"][predicted_index])
        affine_probability = float(payload["affine_probability"][predicted_index])
        combined_probability = float(payload["combined_probability"][predicted_index])
        weight = float(payload["weight"])
        composition = payload.get("composition", "arithmetic")
        if composition == "prior_ratio":
            prior_probability = float(payload["prior"][predicted_index])
            effect = (
                f"changes class {_format_value(payload['base_prediction'])} to "
                f"{_format_value(payload['prediction'])}: normalized base * "
                f"(affine / training prior)^{weight:.3g} moves target-class probability "
                f"{base_probability:.6g} to {combined_probability:.6g} "
                f"(affine {affine_probability:.6g}, prior {prior_probability:.6g})"
            )
        else:
            effect = (
                f"changes class {_format_value(payload['base_prediction'])} to "
                f"{_format_value(payload['prediction'])}: arithmetic affine composition at "
                f"{100.0 * weight:.1f}% moves target-class probability "
                f"{base_probability:.6g} to {combined_probability:.6g} "
                f"(affine {affine_probability:.6g})"
            )
        if terms:
            effect += f"; largest logit terms: {terms}"
    except (KeyError, TypeError, ValueError, IndexError):
        effect = "recomputes the affine logits, probability composition, and final class"
    evidence.append(
        {
            "id": f"prediction-{len(evidence) + 1}",
            "kind": "affine_probability_decision",
            "label": "Global affine decision",
            "conditions": [],
            "effect": effect,
            "verified": verified,
        }
    )
    return evidence


def _prediction_evidence(
    payload: Any,
    feature_names: Sequence[Any] | None,
    verified: bool,
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    if payload.get("kind") == _AFFINE_DECISION_KIND:
        return _affine_evidence(payload, feature_names, verified)
    if payload.get("kind") in _POSTERIOR_KINDS:
        return _posterior_evidence(payload, verified)
    if "terms_shown" in payload:
        return _additive_evidence(payload, feature_names)
    node = payload.get("proof")
    if node is None:
        return []
    return [
        {
            "id": "prediction-1",
            "kind": "logical_derivation",
            "label": "Decision proof",
            "conditions": proof_conditions(node, feature_names),
            "condition_records": proof_condition_records(node, feature_names),
            "effect": "supports the prediction",
            "verified": bool(check_kernel_proof(node)),
        }
    ]


def _guarantee_kind(mode: str, payload: Any) -> str:
    if isinstance(payload, Mapping) and "certified_precision" in payload:
        return "calibration_region_precision_lower_bound"
    return "conformal_error_bound" if mode == "regression" else "calibration_region_precision_lower_bound"


def _guarantee_conclusion(mode: str, prediction: Any, guarantee: float | None, payload: Any) -> Any:
    if guarantee is None:
        return None
    kind = _guarantee_kind(mode, payload)
    result: dict[str, Any] = {
        "kind": kind,
        "value": float(guarantee),
        "scope": "exchangeable_calibration" if mode == "regression" else "calibration_region",
        "individual_guarantee": False,
    }
    if mode == "classification" and isinstance(payload, Mapping) and "region" in payload:
        result["region"] = _plain_scalar(payload["region"])
    if kind == "conformal_error_bound":
        try:
            center = float(prediction)
            result["interval"] = [center - guarantee, center + guarantee]
        except (TypeError, ValueError):
            pass
    return result


def _guarantee_evidence(
    mode: str,
    guarantee: float | None,
    payload: Any,
    verified: bool | None,
    feature_names: Sequence[Any] | None,
) -> list[dict[str, Any]]:
    if guarantee is None or payload is None:
        return []
    kind = _guarantee_kind(mode, payload)
    if kind == "conformal_error_bound" and isinstance(payload, Mapping):
        statement = (
            f"error bound {_format_value(guarantee)} = difficulty {_format_value(payload.get('sigma'))} "
            f"x calibration factor {_format_value(payload.get('q'))}"
        )
        nested = payload.get("sigma_proof")
        conditions: list[str] = []
        condition_records: list[dict[str, Any]] = []
        if isinstance(nested, Mapping):
            terms = nested.get("terms_shown", [])
            if isinstance(terms, (list, tuple)) and terms and isinstance(terms[0], Mapping):
                node = terms[0].get("proof")
                conditions = proof_conditions(node, feature_names)
                condition_records = proof_condition_records(node, feature_names)
    else:
        region = payload.get("region") if isinstance(payload, Mapping) else None
        statement = (
            f"calibration-region precision lower bound is {100.0 * guarantee:.1f}% for region {region}"
        )
        conditions = proof_conditions(payload, feature_names)
        condition_records = proof_condition_records(payload, feature_names)
    return [
        {
            "id": "guarantee-1",
            "kind": kind,
            "label": "Statistical reliability",
            "conditions": conditions,
            "condition_records": condition_records,
            "effect": statement,
            "verified": verified,
            "scope": "exchangeable_calibration" if mode == "regression" else "calibration_region",
            "individual_guarantee": False,
        }
    ]


def _proof_count(payload: Any) -> int:
    if not isinstance(payload, Mapping):
        return int(payload is not None)
    if payload.get("kind") in {"observed_target_attestation", "signed_target_attestation"}:
        return 1
    if payload.get("kind") in _POSTERIOR_KINDS:
        return 1
    if payload.get("kind") == _AFFINE_DECISION_KIND:
        return _proof_count(payload.get("base_proof")) + 1
    terms = payload.get("terms_shown")
    if isinstance(terms, (list, tuple)):
        return sum(isinstance(term, Mapping) and term.get("proof") is not None for term in terms)
    if "sigma_proof" in payload:
        return _proof_count(payload.get("sigma_proof"))
    return int(payload.get("proof") is not None)


def _verification_scope(payload: Any) -> tuple[str, dict[str, int] | None]:
    if not isinstance(payload, Mapping):
        return "included_evidence", None
    if payload.get("kind") in _POSTERIOR_KINDS:
        return "posterior_arithmetic", None
    if payload.get("kind") == _AFFINE_DECISION_KIND:
        return "affine_arithmetic_and_shown_region_memberships", None
    terms = payload.get("terms_shown")
    total = payload.get("n_stages")
    if isinstance(terms, (list, tuple)) and isinstance(total, int):
        coverage = {"shown": len(terms), "total": total}
        scope = "included_stages" if len(terms) >= total else "shown_region_memberships"
        return scope, coverage
    return "included_evidence", None


def _checker_name(payload: Any) -> str:
    if isinstance(payload, Mapping) and payload.get("kind") == _AFFINE_DECISION_KIND:
        return "Affine arithmetic + FOLKernel"
    return "FOLKernel"


def _verification_statement(scope: str, coverage: Mapping[str, int] | None, verified: bool) -> str:
    if not verified:
        return "At least one included machine-proof check failed."
    if scope == "shown_region_memberships" and coverage is not None:
        omitted = coverage["total"] - coverage["shown"]
        return (
            f"All {coverage['shown']} shown region memberships were re-derived; "
            f"{omitted} model stages are omitted from this compact response."
        )
    if scope == "posterior_arithmetic":
        return "The finite counts, posterior update, and resulting class were recomputed."
    if scope == "affine_arithmetic_and_shown_region_memberships":
        return (
            "The affine logits, sigmoid/softmax, probability blend, final class, and the incumbent's shown "
            "region memberships were recomputed."
        )
    return "All machine evidence included in this response was re-derived."


def _summary(
    mode: str,
    prediction: Any,
    payload: Any,
    guarantee: float | None,
    verified: bool,
    attestation: Mapping[str, Any] | None,
) -> str:
    prediction_text = _format_value(prediction)
    prefix = (
        f"Predicted class {prediction_text}." if mode == "classification" else f"Predicted {prediction_text}."
    )
    if isinstance(payload, Mapping) and payload.get("kind") in _POSTERIOR_KINDS:
        base = _format_value(payload.get("base_prediction"))
        detail = f" Finite count evidence changed the base prediction from {base}."
    elif isinstance(payload, Mapping) and payload.get("kind") == _AFFINE_DECISION_KIND:
        base = _format_value(payload.get("base_prediction"))
        detail = (
            f" OOF-authorized affine evidence changed the base prediction from {base}; "
            "its linear arithmetic and blend passed verification."
        )
    elif isinstance(payload, Mapping) and "terms_shown" in payload:
        shown = len(payload.get("terms_shown", []))
        total = payload.get("n_stages", shown)
        detail = f" {shown} shown region memberships passed verification out of {total} model stages."
    else:
        detail = " The included logical evidence passed verification." if verified else ""
    if guarantee is not None:
        if mode == "regression":
            detail += f" Exchangeable conformal error bound: +/- {_format_value(guarantee)}."
        else:
            detail += f" Calibration-region precision lower bound: {100.0 * guarantee:.1f}%."
    if not verified:
        detail += " Machine verification failed."
    elif attestation is not None:
        match = _targets_equal(prediction, attestation["value"])
        result = "confirms" if match else "refutes"
        if is_signed_attestation(attestation):
            detail += (
                f" A trusted Ed25519-signed target {result} this individual prediction "
                "in a post-outcome audit."
            )
        else:
            detail += (
                f" The supplied observed target {result} this individual prediction "
                "in a post-outcome audit; source authenticity remains external."
            )
    else:
        target = "label" if mode == "classification" else "target value"
        detail += (
            f" Model execution is verified; the individual {target} requires independent truth evidence."
        )
    return prefix + detail


def _targets_equal(left: Any, right: Any) -> bool:
    left, right = _plain_scalar(left), _plain_scalar(right)
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def _proof_claims(
    mode: str,
    prediction: Any,
    prediction_verified: bool,
    guarantee_claim: Any,
    guarantee_verified: bool | None,
    attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    target = "label" if mode == "classification" else "target value"
    if guarantee_claim is None:
        statistical = {
            "status": "unavailable",
            "claim": None,
            "statement": "No calibration claim is included in this response.",
        }
    else:
        statistical = {
            "status": "verified" if guarantee_verified else "failed",
            "claim": guarantee_claim,
            "statement": (
                "This is a calibration-population statement, not a deterministic claim "
                f"about this row's true {target}."
            ),
        }
    if attestation is None:
        individual = {
            "status": "not_certified",
            "verified": False,
            "reason": "independent_ground_truth_proof_unavailable",
            "statement": f"No independent proof of this row's true {target} is included.",
        }
    else:
        match = _targets_equal(prediction, attestation["value"])
        signed = is_signed_attestation(attestation)
        individual = {
            "status": "confirmed" if match else "refuted",
            "verified": match,
            "reason": "matches_observed_target" if match else "differs_from_observed_target",
            "scope": "post_outcome_audit",
            "observed_target": attestation["value"],
            "source": attestation["source"],
            "subject": attestation["subject"],
            "source_authentication": ("trusted_ed25519_signature" if signed else "external_to_tabpvn"),
            "statement": (
                f"The prediction {'matches' if match else 'does not match'} the supplied observed {target}; "
                + (
                    "TabPVN verified the trusted authority signature, binding, and equality."
                    if signed
                    else "TabPVN verifies the binding and equality, not the source's authenticity."
                )
            ),
        }
        if signed:
            individual.update(
                algorithm=attestation["algorithm"],
                key_id=attestation["key_id"],
            )
    return {
        "model_execution": {
            "status": "verified" if prediction_verified else "failed",
            "verified": bool(prediction_verified),
            "statement": "The independent verifier reproduced the model decision from the included evidence.",
        },
        "individual_correctness": individual,
        "statistical_reliability": statistical,
    }


def build_proof_artifact(
    prediction_proof: Any,
    *,
    mode: str,
    prediction: Any,
    prediction_verified: bool,
    guarantee: float | None = None,
    guarantee_proof: Any = None,
    guarantee_verified: bool | None = None,
    feature_names: Sequence[Any] | None = None,
    attestation: TargetAttestation | SignedTargetAttestation | None = None,
    trusted_attestation_keys: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Build the explicit audit artifact containing exact derivation data."""
    guarantee_ok = True if guarantee_proof is None else bool(guarantee_verified)
    verified = bool(prediction_verified and guarantee_ok)
    scope, coverage = _verification_scope(prediction_proof)
    truth_payload = attestation_payload(attestation, trusted_attestation_keys)
    guarantee_conclusion = _guarantee_conclusion(
        mode,
        prediction,
        guarantee,
        guarantee_proof,
    )
    conclusion: dict[str, Any] = {
        "mode": mode,
        "prediction": _plain_scalar(prediction),
        "guarantee": guarantee_conclusion,
    }
    if isinstance(prediction_proof, Mapping) and "score" in prediction_proof:
        conclusion["score"] = float(prediction_proof["score"])

    evidence = _prediction_evidence(prediction_proof, feature_names, prediction_verified)
    evidence += _guarantee_evidence(
        mode,
        guarantee,
        guarantee_proof,
        guarantee_verified,
        feature_names,
    )
    verification: dict[str, Any] = {
        "status": "verified" if verified else "failed",
        "verified": verified,
        "checker": _checker_name(prediction_proof),
        "claim": "model_execution_and_calibration_arithmetic",
        "scope": scope,
        "statement": _verification_statement(scope, coverage, verified),
        "proofs_checked": (
            _proof_count(prediction_proof) + _proof_count(guarantee_proof) + _proof_count(truth_payload)
        ),
        "does_not_establish": (
            "authority_observation_correctness_beyond_signature"
            if is_signed_attestation(truth_payload)
            else "attestation_source_authenticity"
            if truth_payload is not None
            else "individual_ground_truth_correctness"
        ),
    }
    if coverage is not None:
        verification["coverage"] = coverage
    if truth_payload is not None and is_signed_attestation(truth_payload):
        verification["trusted_attestation_key_id"] = truth_payload["key_id"]

    machine: dict[str, Any] = {"prediction": prediction_proof}
    if guarantee_proof is not None:
        machine["guarantee"] = guarantee_proof
    if truth_payload is not None:
        machine["truth"] = truth_payload
    response: dict[str, Any] = {
        "schema": PROOF_ARTIFACT_SCHEMA,
        "summary": _summary(
            mode,
            prediction,
            prediction_proof,
            guarantee,
            verified,
            truth_payload,
        ),
        "conclusion": conclusion,
        "claims": _proof_claims(
            mode,
            prediction,
            bool(prediction_verified),
            guarantee_conclusion,
            None if guarantee_proof is None else bool(guarantee_verified),
            truth_payload,
        ),
        "evidence": evidence,
        "verification": verification,
        "verified": verified,
        "machine_proof": machine,
    }
    if isinstance(prediction_proof, Mapping) and prediction_proof.get("kind") in (
        _POSTERIOR_KINDS | {_AFFINE_DECISION_KIND}
    ):
        response["kind"] = prediction_proof["kind"]
    return response


def _canonical_json_value(value: Any) -> Any:
    """Normalize proof data for a stable JSON audit reference."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _canonical_json_value(tolist())
    return _plain_scalar(value)


def proof_artifact_reference(artifact: Any) -> str:
    """Return the stable SHA-256 reference binding a reply to its audit artifact."""
    if not is_proof_artifact(artifact) or not isinstance(artifact.get("machine_proof"), Mapping):
        raise ValueError("expected a TabPVN proof artifact")
    encoded = json.dumps(
        _canonical_json_value(artifact["machine_proof"]),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _public_reliability(artifact: Mapping[str, Any]) -> dict[str, Any]:
    conclusion = artifact.get("conclusion", {})
    claims = artifact.get("claims", {})
    guarantee = conclusion.get("guarantee") if isinstance(conclusion, Mapping) else None
    statistical = claims.get("statistical_reliability", {}) if isinstance(claims, Mapping) else {}
    if not isinstance(guarantee, Mapping):
        return {"status": "unavailable"}
    kind = guarantee.get("kind")
    result: dict[str, Any] = {
        "status": statistical.get("status", "unavailable"),
        "type": "error_bound" if kind == "conformal_error_bound" else "precision_lower_bound",
        "value": float(guarantee["value"]),
        "applies_to": "validated_population"
        if kind == "conformal_error_bound"
        else "similar_validation_cases",
    }
    if kind == "conformal_error_bound" and isinstance(guarantee.get("interval"), (list, tuple)):
        result["interval"] = [float(value) for value in guarantee["interval"]]
    return result


def _public_outcome(artifact: Mapping[str, Any]) -> dict[str, Any]:
    claims = artifact.get("claims", {})
    individual = claims.get("individual_correctness", {}) if isinstance(claims, Mapping) else {}
    status = individual.get("status")
    if status not in {"confirmed", "refuted"}:
        return {"status": "not_observed"}
    source_verified = individual.get("source_authentication") == "trusted_ed25519_signature"
    outcome = {
        "status": status,
        "observed_value": _plain_scalar(individual.get("observed_target")),
        "source": individual.get("source"),
        "subject": individual.get("subject"),
        "source_verified": source_verified,
    }
    if source_verified and individual.get("key_id") is not None:
        outcome["key_id"] = individual["key_id"]
    return outcome


def _public_reasons(artifact: Mapping[str, Any], prediction: Any) -> list[dict[str, Any]]:
    descriptions = {
        "finite_count_posterior": "Comparable historical outcomes support this prediction.",
        "affine_probability_decision": "The overall feature pattern supports this prediction.",
        "decision_region": "The model baseline supports this prediction.",
        "logical_derivation": "The observed feature values support this prediction.",
    }
    reasons: list[dict[str, Any]] = []
    seen: set[str] = set()
    evidence = artifact.get("evidence", [])
    if not isinstance(evidence, (list, tuple)):
        return reasons
    for item in evidence:
        if not isinstance(item, Mapping) or str(item.get("id", "")).startswith("guarantee-"):
            continue
        raw_conditions = item.get("condition_records", ())
        conditions = [
            _canonical_json_value(condition) for condition in raw_conditions if isinstance(condition, Mapping)
        ]
        if conditions:
            key = json.dumps(conditions, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            if key in seen:
                continue
            reasons.append({"conditions": conditions, "supports": _plain_scalar(prediction)})
            seen.add(key)
            continue
        description = descriptions.get(str(item.get("kind")))
        key = f"description:{description}" if description else ""
        if description and key not in seen:
            reasons.append({"description": description, "supports": _plain_scalar(prediction)})
            seen.add(key)
    return reasons


def _public_summary(
    prediction: Any,
    mode: str,
    reliability: Mapping[str, Any],
    outcome: Mapping[str, Any],
    decision_status: str,
) -> str:
    parts = [f"Prediction: {_format_value(prediction)}."]
    parts.append(
        "Decision verification passed." if decision_status == "verified" else "Decision verification failed."
    )
    if reliability.get("status") == "verified":
        value = float(reliability["value"])
        if reliability.get("type") == "error_bound":
            parts.append(f"Validated error bound: +/- {_format_value(value)}.")
        else:
            parts.append(
                f"Estimated precision is at least {100.0 * value:.1f}% among similar validation cases."
            )
    status = outcome.get("status")
    if status == "confirmed":
        suffix = (
            " with a verified source."
            if outcome.get("source_verified")
            else "; the source is declared but not authenticated."
        )
        parts.append("The supplied outcome confirms this prediction" + suffix)
    elif status == "refuted":
        suffix = (
            " with a verified source."
            if outcome.get("source_verified")
            else "; the source is declared but not authenticated."
        )
        parts.append("The supplied outcome refutes this prediction" + suffix)
    else:
        target = "label" if mode == "classification" else "target"
        parts.append(f"No observed {target} was supplied.")
    return " ".join(parts)


def public_proof_response(artifact: Any) -> dict[str, Any]:
    """Project an audit artifact into the implementation-neutral public schema."""
    if not is_proof_artifact(artifact):
        raise ValueError("expected a TabPVN proof artifact")
    conclusion = artifact.get("conclusion")
    claims = artifact.get("claims")
    verification = artifact.get("verification")
    if not all(isinstance(value, Mapping) for value in (conclusion, claims, verification)):
        raise ValueError("malformed TabPVN proof artifact")
    mode = str(conclusion["mode"])
    prediction = _plain_scalar(conclusion["prediction"])
    reliability = _public_reliability(artifact)
    outcome = _public_outcome(artifact)
    model_execution = claims.get("model_execution", {})
    decision_status = model_execution.get("status", "failed")
    overall_status = "verified" if verification.get("verified") is True else "failed"
    return {
        "schema": PROOF_SCHEMA,
        "summary": _public_summary(prediction, mode, reliability, outcome, decision_status),
        "prediction": {"task": mode, "value": prediction},
        "reliability": reliability,
        "reasons": _public_reasons(artifact, prediction),
        "outcome": outcome,
        "verification": {
            "status": overall_status,
            "decision": decision_status,
            "reliability": reliability["status"],
            "audit_reference": proof_artifact_reference(artifact),
        },
    }


def build_proof_response(prediction_proof: Any, **kwargs: Any) -> dict[str, Any]:
    """Build an implementation-neutral response from raw prediction evidence."""
    return public_proof_response(build_proof_artifact(prediction_proof, **kwargs))


def machine_payloads(response: Any) -> tuple[Any, ...] | None:
    """Return exact machine payloads from a well-shaped audit artifact."""
    if not is_proof_artifact(response):
        return None
    machine = response.get("machine_proof")
    if not isinstance(machine, Mapping) or "prediction" not in machine:
        return None
    allowed = {"prediction", "guarantee", "truth"}
    if any(key not in allowed for key in machine):
        return None
    payloads = [machine["prediction"]]
    if "guarantee" in machine:
        payloads.append(machine["guarantee"])
    if "truth" in machine:
        payloads.append(machine["truth"])
    return tuple(payloads)


def proof_prediction(payload: Any) -> Any:
    """Read the prediction committed to by a supported machine payload."""
    if not isinstance(payload, Mapping):
        return None
    return _plain_scalar(payload.get("class", payload.get("prediction")))


def _proof_guarantee(payload: Any) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("certified_precision", payload.get("bound"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _payload_mode(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    if payload.get("kind") in _POSTERIOR_KINDS:
        return "classification"
    if payload.get("kind") == _AFFINE_DECISION_KIND:
        return "classification"
    if "class" in payload:
        return "classification"
    if "prediction" in payload:
        return "regression"
    return None


def response_matches_machine(
    response: Any,
    trusted_attestation_keys: Mapping[str, bytes] | None = None,
) -> bool:
    """Fail closed if a machine-derived artifact claim changed after construction."""
    payloads = machine_payloads(response)
    if payloads is None:
        return False
    machine = response["machine_proof"]
    prediction_payload = machine["prediction"]
    guarantee_payload = machine.get("guarantee")
    truth_payload = machine.get("truth")
    if truth_payload is not None and not verify_attestation_payload(
        truth_payload,
        trusted_attestation_keys,
    ):
        return False

    conclusion = response.get("conclusion")
    verification = response.get("verification")
    if not isinstance(conclusion, Mapping) or not isinstance(verification, Mapping):
        return False
    mode = _payload_mode(prediction_payload)
    prediction = proof_prediction(prediction_payload)
    if mode is None or prediction is None or conclusion.get("mode") != mode:
        return False

    if guarantee_payload is None:
        guarantee_value = None
        expected_guarantee = None
    else:
        guarantee_value = _proof_guarantee(guarantee_payload)
        if guarantee_value is None:
            return False
        expected_guarantee = _guarantee_conclusion(
            mode,
            prediction,
            guarantee_value,
            guarantee_payload,
        )
    expected_conclusion: dict[str, Any] = {
        "mode": mode,
        "prediction": prediction,
        "guarantee": expected_guarantee,
    }
    if isinstance(prediction_payload, Mapping) and "score" in prediction_payload:
        expected_conclusion["score"] = float(prediction_payload["score"])
    if conclusion != expected_conclusion:
        return False

    expected_claims = _proof_claims(
        mode,
        prediction,
        True,
        expected_guarantee,
        None if guarantee_payload is None else True,
        truth_payload,
    )
    scope, coverage = _verification_scope(prediction_payload)
    expected_verification = {
        "status": "verified",
        "verified": True,
        "checker": _checker_name(prediction_payload),
        "claim": "model_execution_and_calibration_arithmetic",
        "scope": scope,
        "statement": _verification_statement(scope, coverage, True),
        "proofs_checked": sum(_proof_count(payload) for payload in payloads),
        "does_not_establish": (
            "authority_observation_correctness_beyond_signature"
            if is_signed_attestation(truth_payload)
            else "attestation_source_authenticity"
            if truth_payload is not None
            else "individual_ground_truth_correctness"
        ),
    }
    if coverage is not None:
        expected_verification["coverage"] = coverage
    if truth_payload is not None and is_signed_attestation(truth_payload):
        expected_verification["trusted_attestation_key_id"] = truth_payload["key_id"]
    expected_summary = _summary(
        mode,
        prediction,
        prediction_payload,
        guarantee_value,
        True,
        truth_payload,
    )
    return bool(
        response.get("verified") is True
        and response.get("claims") == expected_claims
        and response.get("verification") == expected_verification
        and response.get("summary") == expected_summary
    )


def proof_response_matches_artifact(response: Any, artifact: Any) -> bool:
    """Check the public projection and SHA-256 reference against an artifact."""
    if not is_proof_response(response) or not is_proof_artifact(artifact):
        return False
    try:
        return response == public_proof_response(artifact)
    except (KeyError, TypeError, ValueError):
        return False


def _verify_sigma(payload: Mapping[str, Any], verify_nested: Callable[[Any], bool]) -> bool:
    try:
        sigma, q, bound = float(payload["sigma"]), float(payload["q"]), float(payload["bound"])
        return bool(
            all(math.isfinite(value) for value in (sigma, q, bound))
            and sigma >= 0.0
            and q >= 0.0
            and math.isclose(bound, sigma * q)
            and verify_nested(payload["sigma_proof"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _verify_additive(payload: Mapping[str, Any], verify_nested: Callable[[Any], bool]) -> bool:
    terms, stages = payload.get("terms_shown"), payload.get("n_stages")
    if not isinstance(terms, (list, tuple)) or not isinstance(stages, int):
        return False
    if len(terms) > stages or (stages > 0 and not terms):
        return False
    if any(not isinstance(term, Mapping) or not verify_nested(term.get("proof")) for term in terms):
        return False
    if len(terms) != stages or "base" not in payload or not all("contribution" in term for term in terms):
        return True
    try:
        reproduced = float(payload["base"]) + sum(float(term["contribution"]) for term in terms)
        return math.isclose(reproduced, float(payload["prediction"]))
    except (KeyError, TypeError, ValueError):
        return False


def verify_structured_payload(
    payload: Any,
    verify_nested: Callable[[Any], bool],
    trusted_attestation_keys: Mapping[str, bytes] | None = None,
) -> bool | None:
    """Verify a composite runtime payload, returning ``None`` for a raw Horn node."""
    if not isinstance(payload, Mapping):
        return None
    kind = payload.get("kind")
    if kind in {"observed_target_attestation", "signed_target_attestation"}:
        return verify_attestation_payload(payload, trusted_attestation_keys)
    if kind == _AFFINE_DECISION_KIND:
        from tabpvn.proposers.affine import AffineLogitRead

        return AffineLogitRead.verify_evidence(payload, verify_base=verify_nested)
    if "sigma_proof" in payload:
        return _verify_sigma(payload, verify_nested)
    if "terms_shown" in payload:
        return _verify_additive(payload, verify_nested)
    return None


__all__ = [
    "PROOF_ARTIFACT_SCHEMA",
    "PROOF_SCHEMA",
    "SignedTargetAttestation",
    "TargetAttestation",
    "build_proof_artifact",
    "build_proof_response",
    "is_proof_artifact",
    "is_proof_response",
    "machine_payloads",
    "proof_artifact_reference",
    "proof_condition_records",
    "proof_conditions",
    "proof_prediction",
    "proof_response_matches_artifact",
    "public_proof_response",
    "response_matches_machine",
    "verify_structured_payload",
]
