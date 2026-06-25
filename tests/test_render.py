"""Atom palette/radii resolution and color parsing (no actual POV-Ray render)."""
import numpy as np
from ase import Atoms

from tinykit.povray import (
    default_color_map, default_radius_map, parse_rgb, resolve_atom_styles,
)


def test_hydrogen_is_vesta_pink_not_blue():
    # Regression: H used to be dark blue (#274AB3) and vanish on white.
    colors = default_color_map()  # 0-255 RGB
    r, g, b = colors["H"]
    assert (r, g, b) == (255, 204, 204)
    assert r >= b  # pink/white, not blue-dominant


def test_resolve_atom_styles_applies_radius_scale():
    atoms = Atoms("CO", positions=[(0, 0, 0), (0, 0, 1.1)])
    radmap = default_radius_map()
    colors, radii = resolve_atom_styles(atoms, radius_scale=0.6)
    assert len(colors) == 2 and len(radii) == 2
    assert np.isclose(radii[0], radmap["C"] * 0.6)
    assert np.isclose(radii[1], radmap["O"] * 0.6)
    # colors are normalized to 0-1 for POV-Ray
    assert all(0.0 <= c <= 1.0 for rgb in colors for c in rgb)


def test_overrides_do_not_mutate_templates():
    atoms = Atoms("C", positions=[(0, 0, 0)])
    _, radii = resolve_atom_styles(atoms, overrides={"C": {"radius": 2.0}}, radius_scale=1.0)
    assert np.isclose(radii[0], 2.0)
    # the shipped template is untouched
    assert default_radius_map()["C"] != 2.0


def test_parse_rgb_hex_and_csv():
    assert np.allclose(parse_rgb("#ff0000"), (1.0, 0.0, 0.0))
    assert np.allclose(parse_rgb("255,0,0"), (1.0, 0.0, 0.0))      # 0-255 scale inferred
    assert np.allclose(parse_rgb("0.5,0.5,0.5"), (0.5, 0.5, 0.5))  # already 0-1
