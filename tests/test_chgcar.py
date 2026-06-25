"""Charge-density isovalue heuristic, input detection, and isosurface assembly."""
import logging
import types

import numpy as np

from tinykit.chgcar import (
    resolve_isovalue, is_chgcar_input, build_isosurface_data,
)


def _args(**kw):
    base = dict(input="POSCAR", chgcar=None, input_is_chgcar=False,
                isovalue=None, color_scheme="blue-orange",
                iso_color="0.6,0.75,0.95", iso_color_negative="0.95,0.75,0.6",
                iso_transmittance=0.5, dual_phase=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_resolve_isovalue_explicit_passes_through_abs():
    d = np.linspace(0, 1, 50)
    assert resolve_isovalue(d, 0.05) == 0.05
    assert resolve_isovalue(d, -0.05) == 0.05  # sign is normalized away


def test_resolve_isovalue_positive_branch():
    d = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
    expected = min(d.mean() + 2 * d.std(), 0.5 * d.max())
    assert np.isclose(resolve_isovalue(d, None), abs(expected))


def test_resolve_isovalue_negative_branch():
    d = np.array([-1.0, -0.2, 0.0, 0.2, 1.0])
    expected = min(2 * d.std(), 0.3 * np.abs(d).max())
    assert np.isclose(resolve_isovalue(d, None), abs(expected))


def test_resolve_isovalue_too_large_warns(caplog):
    d = np.linspace(0, 1, 10)
    with caplog.at_level(logging.WARNING):
        v = resolve_isovalue(d, 5.0)
    assert v == 5.0
    assert any("exceeds" in r.message for r in caplog.records)


def test_is_chgcar_input():
    assert is_chgcar_input(_args(input="CHGCAR"))
    assert is_chgcar_input(_args(input="PARCHG"))
    assert is_chgcar_input(_args(input="POSCAR", chgcar="CHGCAR"))
    assert is_chgcar_input(_args(input="POSCAR", input_is_chgcar=True))
    assert not is_chgcar_input(_args(input="CONTCAR"))


def test_build_isosurface_single_vs_dual():
    d = np.linspace(0, 1, 27).reshape(3, 3, 3)
    single = build_isosurface_data(d, _args(isovalue=0.1, dual_phase=False))
    assert len(single) == 1
    assert single[0]["cut_off"] == 0.1
    assert len(single[0]["color"]) == 4  # transmittance 0.5 appended to RGB

    dual = build_isosurface_data(d, _args(isovalue=0.1, dual_phase=True))
    assert len(dual) == 2
    assert dual[1]["cut_off"] == -0.1


def test_build_isosurface_custom_colors_use_parse_rgb():
    d = np.linspace(0, 1, 27).reshape(3, 3, 3)
    surf = build_isosurface_data(
        d, _args(isovalue=0.1, color_scheme="custom",
                 iso_color="255,0,0", iso_transmittance=0.0))
    # "255,0,0" -> parse_rgb -> (1,0,0); transmittance 0 leaves a plain RGB triple
    assert surf[0]["color"] == (1.0, 0.0, 0.0)
