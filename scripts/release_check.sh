#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
artifact_dir=$(mktemp -d "${TMPDIR:-/tmp}/tabpvn-release.XXXXXX")
trap 'rm -rf "$artifact_dir"' EXIT HUP INT TERM

cd "$root"
uv run --locked --extra dev ruff check tabpvn core tools/verify_runtime_artifact.py
uv run --locked --extra dev ruff format --check tabpvn core tools/verify_runtime_artifact.py
uv run --locked --extra dev mypy
find core tabpvn -name '*.py' \
  ! -path 'tabpvn/experiments/*' \
  ! -path 'tabpvn/tests/*' \
  ! -path 'tabpvn/fol_regression.py' \
  -print0 | xargs -0 uv run --locked --extra dev vulture --min-confidence 90
uv run --locked --extra dev pytest -q
uv build --out-dir "$artifact_dir"
for artifact in "$artifact_dir"/*; do
  uv run --locked python tools/verify_runtime_artifact.py "$artifact"
done
