#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
rm -rf .venv
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python --editable . --no-deps
# Heavy deps (pymatgen, ase, ...) and argcomplete come from the system env
# (home-manager python3.withPackages), so this venv is built with --no-deps.
# Add `argcomplete` to that closure to enable `tinykit` tab completion below.

# Register tab completion on venv activation (bash/zsh).
cat >> .venv/bin/activate <<'ACTIVATE'

# tinykit tab completion (added by setup_venv.sh)
if command -v register-python-argcomplete >/dev/null 2>&1; then
    eval "$(register-python-argcomplete tinykit)"
fi
ACTIVATE

echo ""
echo "Done. Activate with:  source ~/github/tinykit/.venv/bin/activate"
echo "Tab completion for 'tinykit' is registered automatically on activation."
