import copy
import json

import numpy as np
import pytest

from core.kernel_fol import FOLKernel
from core.kernel_fol import check_proof as check_kernel_proof
from tabpvn import (
    CertifiedAttention,
    SignedTargetAttestation,
    TabPVN,
    TargetAttestation,
    generate_attestation_keypair,
)
from tabpvn.proofs import (
    PROOF_ARTIFACT_SCHEMA,
    PROOF_SCHEMA,
    build_proof_artifact,
    build_proof_response,
    proof_condition_records,
    public_proof_response,
)


def _classification_payload():
    rule = (("reg2", "R"), [("feat", "R", 0, "V0"), ("cmp", ">", "V0", 100.0)])
    kernel = FOLKernel([rule])
    _facts, provenance = kernel.closure([("feat", 0, 0, 125.0)])
    node = kernel.proof(("reg2", 0), provenance)
    return {
        "class": "fraud",
        "score": 1.25,
        "n_stages": 2,
        "terms_shown": [
            {"stage": 0, "region": 2, "logit_contribution": 0.75, "proof": node},
            {"stage": 1, "region": 2, "logit_contribution": 0.50, "proof": node},
        ],
    }


def _classification_artifact(**kwargs):
    return build_proof_artifact(
        _classification_payload(),
        mode="classification",
        prediction="fraud",
        prediction_verified=True,
        feature_names=["amount"],
        **kwargs,
    )


def test_public_response_is_clean_and_bound_to_an_explicit_artifact():
    payload = _classification_payload()
    response = build_proof_response(
        payload,
        mode="classification",
        prediction="fraud",
        prediction_verified=TabPVN.check_proof(payload),
        feature_names=["amount"],
    )
    artifact = build_proof_artifact(
        payload,
        mode="classification",
        prediction="fraud",
        prediction_verified=True,
        feature_names=["amount"],
    )

    assert set(response) == {
        "schema",
        "summary",
        "prediction",
        "reliability",
        "reasons",
        "outcome",
        "verification",
    }
    assert response["schema"] == PROOF_SCHEMA
    assert response["prediction"] == {"task": "classification", "value": "fraud"}
    assert response["reliability"] == {"status": "unavailable"}
    assert response["reasons"] == [
        {
            "conditions": [
                {
                    "feature": "amount",
                    "operator": "gt",
                    "value": 100.0,
                    "observed": 125.0,
                }
            ],
            "supports": "fraud",
        }
    ]
    assert response["outcome"] == {"status": "not_observed"}
    assert response["verification"]["status"] == "verified"
    assert response["verification"]["audit_reference"].startswith("sha256:")

    encoded = json.dumps(response).lower()
    for internal_name in (
        "machine_proof",
        "model stages",
        "region",
        "logit",
        "folkernel",
        "checker",
        "proofs_checked",
        '"score"',
    ):
        assert internal_name not in encoded
    assert artifact["schema"] == PROOF_ARTIFACT_SCHEMA
    assert not TabPVN.check_proof(response)
    assert TabPVN.check_proof(
        response,
        base_facts=[("feat", 0, 0, 125.0)],
        artifact=artifact,
    )


def test_categorical_conditions_use_the_same_typed_public_contract():
    columns = (0, 1, 2)
    rule = (
        ("segment_rule", "R"),
        [("cat", "R", columns, "V0"), ("cmp", "in", "V0", (0, 2))],
    )
    kernel = FOLKernel([rule])
    _facts, provenance = kernel.closure([("cat", 0, columns, 2)])
    node = kernel.proof(("segment_rule", 0), provenance)

    assert proof_condition_records(
        node,
        ["segment=north", "segment=east", "segment=south"],
    ) == [
        {
            "feature": "segment",
            "operator": "in",
            "value": ["north", "south"],
            "observed": "south",
        }
    ]


def test_public_and_artifact_tampering_fail_closed():
    artifact = _classification_artifact()
    response = public_proof_response(artifact)

    changed_prediction = copy.deepcopy(response)
    changed_prediction["prediction"]["value"] = "legitimate"
    assert not TabPVN.check_proof(changed_prediction, artifact=artifact)

    changed_summary = copy.deepcopy(response)
    changed_summary["summary"] = "Guaranteed correct."
    assert not TabPVN.check_proof(changed_summary, artifact=artifact)

    changed_machine_data = copy.deepcopy(artifact)
    changed_machine_data["machine_proof"]["prediction"]["terms_shown"][0]["proof"] = ("malformed",)
    assert not TabPVN.check_proof(changed_machine_data)
    assert not TabPVN.check_proof(response, artifact=changed_machine_data)

    legacy_envelope = copy.deepcopy(response)
    legacy_envelope["schema"] = "tabpvn.proof/2"
    assert not TabPVN.check_proof(legacy_envelope, artifact=artifact)
    assert not check_kernel_proof(None)


def test_target_attestation_is_publicly_reported_without_internal_claim_fields():
    confirmed_artifact = _classification_artifact(
        attestation=TargetAttestation(
            value="fraud",
            source="audited holdout labels",
            subject="holdout-row:17",
        )
    )
    confirmed = public_proof_response(confirmed_artifact)

    assert confirmed["outcome"] == {
        "status": "confirmed",
        "observed_value": "fraud",
        "source": "audited holdout labels",
        "subject": "holdout-row:17",
        "source_verified": False,
    }
    assert TabPVN.check_proof(confirmed, artifact=confirmed_artifact)

    refuted_artifact = _classification_artifact(
        attestation=TargetAttestation(
            value="legitimate",
            source="audited holdout labels",
            subject="holdout-row:17",
        )
    )
    refuted = public_proof_response(refuted_artifact)
    assert refuted["outcome"]["status"] == "refuted"
    assert TabPVN.check_proof(refuted, artifact=refuted_artifact)

    forged = copy.deepcopy(refuted)
    forged["outcome"]["status"] = "confirmed"
    assert not TabPVN.check_proof(forged, artifact=refuted_artifact)

    with pytest.raises(ValueError, match="source"):
        TargetAttestation(value="fraud", source=" ", subject="holdout-row:17")


def test_signed_target_attestation_requires_its_trusted_ed25519_key():
    pytest.importorskip("cryptography")
    private_key, public_key = generate_attestation_keypair()
    signed = SignedTargetAttestation.sign(
        value="fraud",
        source="official holdout authority",
        subject="holdout-row:17",
        key_id="holdout-2026",
        private_key=private_key,
    )
    with pytest.raises(ValueError, match="matching trusted public key"):
        _classification_artifact(attestation=signed)

    artifact = _classification_artifact(
        attestation=signed,
        trusted_attestation_keys={"holdout-2026": public_key},
    )
    response = public_proof_response(artifact)
    assert response["outcome"]["status"] == "confirmed"
    assert response["outcome"]["source_verified"] is True
    assert response["outcome"]["key_id"] == "holdout-2026"
    assert not TabPVN.check_proof(artifact)
    assert TabPVN.check_proof(
        response,
        artifact=artifact,
        trusted_attestation_keys={"holdout-2026": public_key},
    )

    _wrong_private, wrong_public = generate_attestation_keypair()
    assert not TabPVN.check_proof(
        response,
        artifact=artifact,
        trusted_attestation_keys={"holdout-2026": wrong_public},
    )
    tampered = copy.deepcopy(artifact)
    tampered["machine_proof"]["truth"]["value"] = "legitimate"
    assert not TabPVN.check_proof(
        tampered,
        trusted_attestation_keys={"holdout-2026": public_key},
    )


def test_regression_public_response_exposes_bound_without_derivation_arithmetic():
    prediction_proof = {
        "base": 1.0,
        "terms_shown": [
            {"stage": 0, "region": 0, "contribution": 0.5, "proof": "root"},
        ],
        "n_stages": 1,
        "prediction": 1.5,
    }
    sigma_proof = {
        "base": 0.2,
        "terms_shown": [
            {"stage": 0, "region": 0, "contribution": 0.1, "proof": "root"},
        ],
        "n_stages": 1,
        "prediction": 0.3,
    }
    guarantee_proof = {"sigma": 0.3, "q": 2.0, "bound": 0.6, "sigma_proof": sigma_proof}
    artifact = build_proof_artifact(
        prediction_proof,
        mode="regression",
        prediction=1.5,
        prediction_verified=TabPVN.check_proof(prediction_proof),
        guarantee=0.6,
        guarantee_proof=guarantee_proof,
        guarantee_verified=TabPVN.check_proof(guarantee_proof),
    )
    response = public_proof_response(artifact)

    assert response["prediction"] == {"task": "regression", "value": 1.5}
    assert response["reliability"] == {
        "status": "verified",
        "type": "error_bound",
        "value": 0.6,
        "applies_to": "validated_population",
        "interval": [0.9, 2.1],
    }
    assert "difficulty" not in json.dumps(response).lower()
    assert TabPVN.check_proof(response, artifact=artifact)

    tampered = copy.deepcopy(response)
    tampered["reliability"]["value"] = 0.5
    assert not TabPVN.check_proof(tampered, artifact=artifact)


def test_tabpvn_public_proof_and_certificate_do_not_embed_artifacts():
    X = np.arange(48, dtype=float).reshape(24, 2)
    y = (X[:, 0] > 22).astype(int)
    model = TabPVN(boost={"rounds": 8, "depth": 2, "leaf": 2, "patience": 3, "lr": 0.1}).fit(X, y)

    response = model.proof(X, 4)
    artifact = model.proof_artifact(X, 4)
    raw = model.proof(X, 4, raw=True)
    attestation = TargetAttestation(
        value=model.predict(X[[4]])[0],
        source="unit-test observed target",
        subject="row:4",
    )
    attested = model.proof(X, 4, attestation=attestation)
    attested_artifact = model.proof_artifact(X, 4, attestation=attestation)
    certificate = model.certificate(X, 4, attestation=attestation)

    assert response["schema"] == PROOF_SCHEMA
    assert response["prediction"]["value"] == model.predict(X[[4]])[0]
    assert response["reasons"]
    assert "machine_proof" not in response
    assert TabPVN.check_proof(response, artifact=artifact)
    assert raw["terms_shown"]
    assert attested["outcome"]["status"] == "confirmed"
    assert TabPVN.check_proof(attested, artifact=attested_artifact)
    assert certificate["individual_correctness"]["status"] == "confirmed"
    assert certificate["proof"]["schema"] == PROOF_SCHEMA
    assert "machine_proof" not in certificate["proof"]
    assert TabPVN.check_proof(certificate["proof"], artifact=attested_artifact)
    with pytest.raises(ValueError, match="structured proof response"):
        model.proof(X, 4, raw=True, attestation=attestation)


def test_attention_uses_the_same_clean_response_and_explicit_artifact_boundary():
    X = np.arange(40, dtype=float).reshape(20, 2)
    classification_target = (X[:, 0] > 18).astype(int)
    classifier = CertifiedAttention(topk=3).fit(X, classification_target)
    classification = classifier.proof(X, 2)
    classification_artifact = classifier.proof_artifact(X, 2)

    assert classification["schema"] == PROOF_SCHEMA
    assert classification["reasons"] == [
        {
            "description": "Similar historical examples support this prediction.",
            "supports": 0,
        }
    ]
    assert CertifiedAttention.check_proof(classification, artifact=classification_artifact)

    regression_target = np.linspace(0.1, 3.7, len(X))
    regressor = CertifiedAttention(topk=3).fit(X, regression_target)
    regression = regressor.proof(X, 2)
    regression_artifact = regressor.proof_artifact(X, 2)

    assert regression["schema"] == PROOF_SCHEMA
    assert regression["prediction"]["task"] == "regression"
    assert CertifiedAttention.check_proof(regression, artifact=regression_artifact)

    tampered = copy.deepcopy(regression_artifact)
    tampered["machine_proof"]["prediction"]["weighted_sum"] += 1.0
    assert not CertifiedAttention.check_proof(tampered)
