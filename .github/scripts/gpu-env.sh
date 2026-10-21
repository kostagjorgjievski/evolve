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
# A node without a system Python 3.11 (DGX Spark ships Ubuntu 24.04 / 3.12)
# uses uv's managed 3.11 instead.
if ! uv python find 3.11 >/dev/null 2>&1; then
  export UV_PYTHON_PREFERENCE=only-managed
  uv python install 3.11
fi
extra=(); nodeps=()
if [ "$(uname -m)" = aarch64 ]; then
  # The locked torch (2.6, CUDA 12.4) has no CUDA build for aarch64, and the
  # DGX Spark's GB10 (Blackwell) needs CUDA 13. On aarch64 torch comes from
  # PyTorch's CUDA 13 index, pinned by version (EVOLVETS_TORCH_SPEC); every
  # other package is still installed hash-checked from uv.lock. Each foundation
  # manifest records the library versions it was built with.
  # sympy and triton are pinned by torch itself, so they follow it here.
  extra=(--no-emit-package torch --no-emit-package sympy --no-emit-package triton)
  # The export is the full closure minus those three, which are installed
  # first, so the hashed install needs no resolution of its own.
  nodeps=(--no-deps)
fi
uv export --frozen --no-default-groups --group "$group" --no-emit-project \
  "${extra[@]}" --format requirements-txt -o "$req" -q
[ -x "$venv/bin/python" ] || uv venv --python 3.11 "$venv"
if [ "$(uname -m)" = aarch64 ]; then
  uv pip install --python "$venv/bin/python" \
    --index "${EVOLVETS_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}" \
    "${EVOLVETS_TORCH_SPEC:-torch==2.9.1}"
fi
uv pip install --python "$venv/bin/python" --require-hashes "${nodeps[@]}" -r "$req"
uv pip install --python "$venv/bin/python" --no-deps --reinstall .
"$venv/bin/evolvets" --help >/dev/null
"$venv/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())' || true
if [ "$(uname -m)" = aarch64 ]; then uv pip check --python "$venv/bin/python" || true; fi
echo "environment ready: $venv ($group)"
