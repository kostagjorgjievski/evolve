#!/usr/bin/env bash
# Build (or refresh) a Python 3.11 environment for one fm-* dependency group
# straight from uv.lock, with every package hash-checked, installing through
# whatever index UV_INDEX_URL names (the campus mirror on the lab node).
#
# Why not `uv run --locked`: that compares the lock's recorded index with the
# configured one, and on the lab node (mirror configured) it reports the lock
# as out of date. Exporting the lock to a hashed requirements file keeps the
# exact locked versions and verifies the same SHA-256 hashes from any mirror.
#
# Usage: gpu-env.sh <dependency-group> <venv-dir>
set -euo pipefail
group="$1"; venv="$2"
export PATH="$HOME/.local/bin:$PATH"
req="$(mktemp)"
trap 'rm -f "$req"' EXIT
uv export --frozen --no-default-groups --group "$group" --no-emit-project \
  --format requirements-txt -o "$req" -q
[ -x "$venv/bin/python" ] || uv venv --python 3.11 "$venv"
uv pip install --python "$venv/bin/python" --require-hashes -r "$req"
uv pip install --python "$venv/bin/python" --no-deps --reinstall .
"$venv/bin/evolvets" --help >/dev/null
echo "environment ready: $venv ($group)"
