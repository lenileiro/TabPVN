"""Versioned, atomic persistence for fitted TabPVN models."""

from __future__ import annotations

import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from tabpvn._version import __version__

if TYPE_CHECKING:
    from tabpvn.base import TabPVN


_FORMAT = "tabpvn.model"
_FORMAT_VERSION = 1


def save_model(model: TabPVN, path: str | os.PathLike[str]) -> Path:
    """Atomically save a fitted model and its runtime format metadata."""
    from tabpvn.base import TabPVN

    if not isinstance(model, TabPVN):
        raise TypeError("save_model expects a TabPVN instance")
    if not model.__sklearn_is_fitted__():
        raise RuntimeError("model is not fitted; call fit(X, y) before save_model")

    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError(f"model directory does not exist: {destination.parent}")
    envelope = {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "package_version": __version__,
        "model": model,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(envelope, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def load_model(path: str | os.PathLike[str]) -> TabPVN:
    """Load a trusted model file written by :func:`save_model`.

    Pickle can execute code while loading. Never load a model from an untrusted
    source.
    """
    from tabpvn.base import TabPVN

    source = Path(path)
    with source.open("rb") as stream:
        envelope = pickle.load(stream)
    if not isinstance(envelope, dict) or envelope.get("format") != _FORMAT:
        raise ValueError("not a TabPVN model file")
    if envelope.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported TabPVN model format {envelope.get('format_version')!r}; "
            f"this runtime supports {_FORMAT_VERSION}"
        )
    model = envelope.get("model")
    if not isinstance(model, TabPVN):
        raise ValueError("TabPVN model file does not contain a TabPVN estimator")
    saved_version = envelope.get("package_version")
    if saved_version != __version__:
        warnings.warn(
            f"model was saved by TabPVN {saved_version!r} and is loading under {__version__!r}; "
            "verify predictions before deployment",
            RuntimeWarning,
            stacklevel=2,
        )
    return model
