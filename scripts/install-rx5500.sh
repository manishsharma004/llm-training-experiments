#!/usr/bin/env bash
# Replace the CPU PyTorch wheel in .venv with AMD's gfx1012 build.
# The RX 5500 / 5500M is Navi 14. Generic ROCm wheels omit that target.
# uv cannot resolve this index (package pages redirect across hosts and a
# PyPI package named "rocm" collides), so this script uses pip.
# Running `uv sync` afterwards restores the CPU wheel from uv.lock.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Create the environment first: uv sync --group dev" >&2
  exit 1
fi
"$PY" -m pip install \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  --extra-index-url https://pypi.org/simple \
  "torch[device-gfx1012]==2.14.0+rocm10.1.0"
"$PY" -m tinyllm.device
