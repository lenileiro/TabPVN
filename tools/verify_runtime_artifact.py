"""Reject release artifacts that omit runtime modules or include research code."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "benchmark/",
    "certifiedlm/",
    "model/",
    "tabpvn/experiments/",
    "tabpvn/tests/",
)
FORBIDDEN_FILES = {"tabpvn/fol_regression.py"}
REQUIRED_FILES = {
    "core/kernel_fol.py",
    "tabpvn/__init__.py",
    "tabpvn/adapters.py",
    "tabpvn/api.py",
    "tabpvn/base.py",
    "tabpvn/bayes.py",
    "tabpvn/model_io.py",
    "tabpvn/preprocessing.py",
    "tabpvn/pricing.py",
    "tabpvn/proofs.py",
    "tabpvn/relational.py",
}


def _artifact_names(path: Path) -> set[str]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            raw_names = archive.getnames()
        names = set()
        for name in raw_names:
            parts = name.split("/", 1)
            names.add(parts[1] if len(parts) == 2 else parts[0])
        return names
    raise SystemExit(f"unsupported release artifact: {path}")


def verify(path: Path) -> None:
    names = _artifact_names(path)
    leaked = sorted(
        name
        for name in names
        if name in FORBIDDEN_FILES or any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    )
    missing = sorted(REQUIRED_FILES - names)
    if leaked or missing:
        details = []
        if leaked:
            details.append(f"forbidden entries: {leaked}")
        if missing:
            details.append(f"missing runtime entries: {missing}")
        raise SystemExit(f"{path}: " + "; ".join(details))
    print(f"verified runtime artifact: {path} ({len(names)} files)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    verify(args.artifact)


if __name__ == "__main__":
    main()
