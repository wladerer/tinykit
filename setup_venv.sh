#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
rm -rf .venv
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python --editable . --no-deps
echo ""
echo "Done. Activate with:  source ~/github/tinykit/.venv/bin/activate"
