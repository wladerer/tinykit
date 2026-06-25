"""Named INCAR preset loading."""
import pytest

from tinykit.presets import available_presets, load_incar_preset


def test_expected_presets_available():
    names = available_presets()
    for expected in ("adsorb", "slab"):
        assert expected in names


def test_load_returns_dict():
    preset = load_incar_preset("slab")
    assert isinstance(preset, dict)
    assert preset  # non-empty


def test_loads_are_independent_copies():
    a = load_incar_preset("slab")
    a["NELECT"] = 999
    b = load_incar_preset("slab")
    assert "NELECT" not in b  # mutating one load must not affect the next


def test_unknown_preset_raises():
    with pytest.raises(Exception):
        load_incar_preset("does-not-exist")
