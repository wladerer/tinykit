"""Adsorbate resolution and the symmetry-reduction core (canonical keys,
distance pre-filter, duplicate detection)."""
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.structure import Molecule

from tinykit.adsorb import (
    create_single_atom_molecule, get_molecule,
    _check_site_distances, _compute_canonical_key, find_duplicate_structures,
)


# --- molecule resolution -----------------------------------------------------

def test_single_atom_from_element_symbol():
    mol = get_molecule("Au")
    assert isinstance(mol, Molecule)
    assert len(mol) == 1
    assert mol[0].specie.symbol == "Au"


def test_named_molecule_from_json():
    # a known polyatomic entry must round-trip with the right atoms; a bare
    # isinstance check would not catch a corrupted/renamed JSON entry
    mol = get_molecule("CO")
    assert isinstance(mol, Molecule)
    assert len(mol) == 2
    assert {s.specie.symbol for s in mol} == {"C", "O"}


def test_invalid_molecule_raises():
    with pytest.raises(ValueError):
        get_molecule("NotARealThing")


def test_create_single_atom_validates_element():
    with pytest.raises(ValueError):
        create_single_atom_molecule("Xx")


# --- symmetry reduction core -------------------------------------------------

def test_check_site_distances():
    coords = np.array([[0.0, 0, 0], [3.0, 0, 0]])
    assert _check_site_distances(coords, 2.0) is True   # 3.0 apart, all > 2.0
    assert _check_site_distances(coords, 4.0) is False  # closer than 4.0


def test_canonical_key_is_symmetry_invariant():
    identity = np.eye(3)
    # identity plus a +1/2 translation along x (a symmetry of the test cell)
    ops = [(identity, np.array([0.0, 0.0, 0.0])),
           (identity, np.array([0.5, 0.0, 0.0]))]
    a = np.array([[0.1, 0.1, 0.5]])
    b = np.array([[0.6, 0.1, 0.5]])  # a, shifted by the +1/2 x translation
    key_a = _compute_canonical_key((a, ops))
    key_b = _compute_canonical_key((b, ops))
    assert key_a == key_b
    # a placement not related by any op gets a different key
    c = np.array([[0.1, 0.7, 0.5]])
    assert _compute_canonical_key((c, ops)) != key_a


def _slab_with_halfx_symmetry():
    """A 2x1 square slab; the +1/2 x translation maps its two atoms onto each
    other, so adsorbates at x and x+1/2 are symmetry-equivalent."""
    lat = Lattice.from_parameters(3.0, 3.0, 18.0, 90, 90, 90)
    base = Structure(lat, ["Cu"], [[0.0, 0.0, 0.5]])
    return base * [2, 1, 1]  # 2 Cu at x=0 and x=0.5, a=6


def test_find_duplicate_structures_groups_symmetry_equivalent():
    slab = _slab_with_halfx_symmetry()
    n = len(slab)

    def with_ads(fx, fy):
        s = slab.copy()
        s.append("H", [fx, fy, 0.6])
        return s

    s1 = with_ads(0.1, 0.0)
    s2 = with_ads(0.6, 0.0)   # s1 shifted by +1/2 x -> equivalent to s1
    s3 = with_ads(0.1, 0.5)   # different y, not related -> distinct

    groups = find_duplicate_structures([s1, s2, s3], slab, n)
    assert any(set(g) == {0, 1} for g in groups)        # s1, s2 grouped
    assert all(2 not in g for g in groups)              # s3 stands alone
