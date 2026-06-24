"""Adsorbate molecule resolution."""
import pytest
from pymatgen.core.structure import Molecule

from tinykit.adsorb import create_single_atom_molecule, get_molecule, molecules


def test_single_atom_from_element_symbol():
    mol = get_molecule("Au")
    assert isinstance(mol, Molecule)
    assert len(mol) == 1
    assert mol[0].specie.symbol == "Au"


def test_named_molecule_from_json():
    # pick whatever the bundled JSON actually ships
    name = next(iter(molecules))
    mol = get_molecule(name)
    assert isinstance(mol, Molecule)


def test_invalid_molecule_raises():
    with pytest.raises(ValueError):
        get_molecule("NotARealThing")


def test_create_single_atom_validates_element():
    with pytest.raises(ValueError):
        create_single_atom_molecule("Xx")
