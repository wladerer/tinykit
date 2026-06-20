"""Load named INCAR presets from resources/incars.yaml."""

from pathlib import Path

import yaml

_INCARS_PATH = Path(__file__).parent / "resources" / "incars.yaml"


def available_presets() -> list[str]:
    """Return the names of all defined INCAR presets."""
    return sorted(yaml.safe_load(_INCARS_PATH.read_text()))


def load_incar_preset(name: str) -> dict:
    """Return a fresh, mutable dict of INCAR tags for the named preset.

    A new dict is returned on every call so callers may safely add or override
    tags (e.g. NELECT, DIPOL) without affecting other invocations.
    """
    presets = yaml.safe_load(_INCARS_PATH.read_text())
    if name not in presets:
        raise KeyError(
            f"Unknown INCAR preset '{name}'. Available: {sorted(presets)}"
        )
    return dict(presets[name])
