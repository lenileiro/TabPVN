"""Post-outcome target attestations and optional Ed25519 authentication."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SIGNED_KIND = "signed_target_attestation"
_UNSIGNED_KIND = "observed_target_attestation"
_SCOPE = "post_outcome_audit"
_ALGORITHM = "ed25519"
_DOMAIN = b"tabpvn.target-attestation/v1\0"


def _plain_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            pass
    return value


def _validate_scalar(value: Any) -> str | int | float | bool:
    value = _plain_scalar(value)
    if not isinstance(value, (str, int, float, bool)) or (
        isinstance(value, float) and not math.isfinite(value)
    ):
        raise TypeError("attested target value must be a finite scalar")
    return value


def _validate_text(value: Any, name: str) -> str:
    value = value.strip() if isinstance(value, str) else ""
    if not value:
        raise ValueError(f"attestation {name} must be a non-empty string")
    return value


def _unsigned_fields(value: Any, source: str, subject: str) -> dict[str, Any]:
    return {
        "value": _validate_scalar(value),
        "source": _validate_text(source, "source"),
        "subject": _validate_text(subject, "subject"),
        "scope": _SCOPE,
    }


def _signed_message(payload: Mapping[str, Any]) -> bytes:
    unsigned = {
        key: payload[key] for key in ("algorithm", "key_id", "kind", "scope", "source", "subject", "value")
    }
    encoded = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _DOMAIN + encoded


def _crypto() -> tuple[Any, Any, Any, Any]:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as error:
        raise ImportError("signed target attestations require `uv sync --extra attest`") from error
    return InvalidSignature, serialization, Ed25519PrivateKey, Ed25519PublicKey


@dataclass(frozen=True)
class TargetAttestation:
    """A separately sourced, post-outcome target observation.

    The declared source is bound into the response but not authenticated. Use
    ``SignedTargetAttestation`` with a trusted public key for authentication.
    """

    value: str | int | float | bool
    source: str
    subject: str

    def __post_init__(self) -> None:
        fields = _unsigned_fields(self.value, self.source, self.subject)
        object.__setattr__(self, "value", fields["value"])
        object.__setattr__(self, "source", fields["source"])
        object.__setattr__(self, "subject", fields["subject"])

    def asdict(self) -> dict[str, Any]:
        """Return the bounded machine payload embedded in a proof response."""
        return {"kind": _UNSIGNED_KIND, **_unsigned_fields(self.value, self.source, self.subject)}


@dataclass(frozen=True)
class SignedTargetAttestation(TargetAttestation):
    """An Ed25519-signed target observation identified by a trusted key ID."""

    key_id: str
    signature: str
    algorithm: str = _ALGORITHM

    def __post_init__(self) -> None:
        super().__post_init__()
        key_id = _validate_text(self.key_id, "key_id")
        if self.algorithm != _ALGORITHM:
            raise ValueError(f"unsupported attestation algorithm: {self.algorithm!r}")
        if not isinstance(self.signature, str):
            raise ValueError("attestation signature must be a base64 string")
        try:
            signature = base64.b64decode(self.signature, validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("attestation signature must be valid base64") from error
        if len(signature) != 64:
            raise ValueError("an Ed25519 attestation signature must contain 64 bytes")
        if base64.b64encode(signature).decode("ascii") != self.signature:
            raise ValueError("attestation signature must use canonical base64")
        object.__setattr__(self, "key_id", key_id)

    @classmethod
    def sign(
        cls,
        *,
        value: str | int | float | bool,
        source: str,
        subject: str,
        key_id: str,
        private_key: bytes,
    ) -> SignedTargetAttestation:
        """Sign one observed target with a raw 32-byte Ed25519 private key."""
        _invalid, _serialization, private_type, _public_type = _crypto()
        fields = _unsigned_fields(value, source, subject)
        payload = {
            "kind": _SIGNED_KIND,
            "algorithm": _ALGORITHM,
            "key_id": _validate_text(key_id, "key_id"),
            **fields,
        }
        signer = private_type.from_private_bytes(bytes(private_key))
        signature = base64.b64encode(signer.sign(_signed_message(payload))).decode("ascii")
        return cls(
            value=fields["value"],
            source=fields["source"],
            subject=fields["subject"],
            key_id=payload["key_id"],
            signature=signature,
        )

    def asdict(self) -> dict[str, Any]:
        """Return the signed machine payload embedded in a proof response."""
        return {
            "kind": _SIGNED_KIND,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            **_unsigned_fields(self.value, self.source, self.subject),
            "signature": self.signature,
        }


def generate_attestation_keypair() -> tuple[bytes, bytes]:
    """Generate raw Ed25519 ``(private_key, public_key)`` bytes."""
    _invalid, serialization, private_type, _public_type = _crypto()
    private_key = private_type.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def verify_attestation_payload(
    payload: Any,
    trusted_keys: Mapping[str, bytes] | None = None,
) -> bool:
    """Verify attestation shape and, for signed payloads, its trusted Ed25519 signature."""
    if not isinstance(payload, Mapping):
        return False
    kind = payload.get("kind")
    if kind == _UNSIGNED_KIND:
        if set(payload) != {"kind", "value", "source", "subject", "scope"}:
            return False
        try:
            return payload == {
                "kind": _UNSIGNED_KIND,
                **_unsigned_fields(payload["value"], payload["source"], payload["subject"]),
            }
        except (KeyError, TypeError, ValueError):
            return False
    if kind != _SIGNED_KIND or set(payload) != {
        "algorithm",
        "key_id",
        "kind",
        "scope",
        "signature",
        "source",
        "subject",
        "value",
    }:
        return False
    if trusted_keys is None or payload.get("key_id") not in trusted_keys:
        return False
    try:
        invalid_signature, _serialization, _private_type, public_type = _crypto()
    except ImportError:
        return False
    try:
        expected = {
            "kind": _SIGNED_KIND,
            "algorithm": _ALGORITHM,
            "key_id": _validate_text(payload["key_id"], "key_id"),
            **_unsigned_fields(payload["value"], payload["source"], payload["subject"]),
            "signature": payload["signature"],
        }
        if payload != expected:
            return False
        signature = base64.b64decode(payload["signature"], validate=True)
        if (
            len(signature) != 64
            or not isinstance(payload["signature"], str)
            or base64.b64encode(signature).decode("ascii") != payload["signature"]
        ):
            return False
        verifier = public_type.from_public_bytes(bytes(trusted_keys[payload["key_id"]]))
        verifier.verify(signature, _signed_message(payload))
        return True
    except (KeyError, TypeError, ValueError, invalid_signature):
        return False


def attestation_payload(
    attestation: TargetAttestation | SignedTargetAttestation | None,
    trusted_keys: Mapping[str, bytes] | None = None,
) -> dict[str, Any] | None:
    """Normalize an attestation and fail closed on missing signature trust."""
    if attestation is None:
        return None
    if not isinstance(attestation, TargetAttestation):
        raise TypeError("attestation must be a TargetAttestation")
    payload = attestation.asdict()
    required_keys = trusted_keys if isinstance(attestation, SignedTargetAttestation) else None
    if not verify_attestation_payload(payload, required_keys):
        if isinstance(attestation, SignedTargetAttestation):
            raise ValueError("signed attestation requires a valid signature and matching trusted public key")
        raise ValueError("invalid target attestation")
    return payload


def is_signed_attestation(payload: Any) -> bool:
    """Return whether a machine payload declares signed target evidence."""
    return isinstance(payload, Mapping) and payload.get("kind") == _SIGNED_KIND


__all__ = [
    "SignedTargetAttestation",
    "TargetAttestation",
    "attestation_payload",
    "generate_attestation_keypair",
    "is_signed_attestation",
    "verify_attestation_payload",
]
