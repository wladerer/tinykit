"""Miller-index parsing and selective-dynamics (layer freezing)."""
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab

from tinykit.slabgen import (
    parse_miller_index, apply_selective_dynamics, FreezingMode,
)


@pytest.mark.parametrize("text,expected", [
    ("111", (1, 1, 1)),
    ("201", (2, 0, 1)),
    ("-201", (-2, 0, 1)),
    ("2,0,1", (2, 0, 1)),
    ("-2,0,1", (-2, 0, 1)),
    ("1,-1,0", (1, -1, 0)),
    ("1 1 1", (1, 1, 1)),
])
def test_parse_miller_index(text, expected):
    assert parse_miller_index(text) == expected


# --- selective dynamics ------------------------------------------------------

RELAX = [True, True, True]
FREEZE = [False, False, False]


def _make_slab(n_layers=5):
    """A flat (001) slab with `n_layers` atoms stacked along z (normal = z),
    so layer projections are just the cartesian z coordinates 2, 4, ... ."""
    lat = Lattice.from_parameters(3.0, 3.0, 20.0, 90, 90, 90)
    zs = [0.1 * (i + 1) for i in range(n_layers)]
    species = ["Cu"] * n_layers
    coords = [[0.0, 0.0, z] for z in zs]
    ouc = Structure(lat, species, coords)
    return Slab(lat, species, coords, miller_index=(0, 0, 1),
                oriented_unit_cell=ouc, shift=0.0, scale_factor=np.eye(3),
                reorient_lattice=False)


def _sd_by_z(slab):
    return {round(s.coords[2], 1): s.properties.get("selective_dynamics") for s in slab}


def test_center_relaxes_top_and_bottom():
    sd = _sd_by_z(apply_selective_dynamics(_make_slab(), 1, FreezingMode.CENTER))
    assert sd[2.0] == RELAX and sd[10.0] == RELAX            # outer layers relax
    assert sd[4.0] == FREEZE and sd[6.0] == FREEZE and sd[8.0] == FREEZE


def test_bottom_freezes_lowest_n_layers():
    sd = _sd_by_z(apply_selective_dynamics(_make_slab(), 2, FreezingMode.BOTTOM))
    assert sd[2.0] == FREEZE and sd[4.0] == FREEZE           # bottom two frozen
    assert sd[6.0] == RELAX and sd[8.0] == RELAX and sd[10.0] == RELAX


def test_top_freezes_highest_n_layers():
    sd = _sd_by_z(apply_selective_dynamics(_make_slab(), 2, FreezingMode.TOP))
    assert sd[8.0] == FREEZE and sd[10.0] == FREEZE          # top two frozen
    assert sd[2.0] == RELAX and sd[4.0] == RELAX and sd[6.0] == RELAX


def test_center_too_many_layers_warns_and_leaves_slab_untouched():
    # CENTER needs nlayers > 2*n; 3 on a 5-layer slab is too many.
    with pytest.warns(UserWarning):
        out = apply_selective_dynamics(_make_slab(), 3, FreezingMode.CENTER)
    assert all(s.properties.get("selective_dynamics") is None for s in out)


def test_bottom_too_many_layers_warns_and_leaves_slab_untouched():
    # BOTTOM/TOP need n < nlayers; freezing all 5 of a 5-layer slab is rejected.
    with pytest.warns(UserWarning):
        out = apply_selective_dynamics(_make_slab(), 5, FreezingMode.BOTTOM)
    assert all(s.properties.get("selective_dynamics") is None for s in out)
