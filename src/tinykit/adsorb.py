import os
import json
import numpy as np
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar, Poscar
from pymatgen.io.vasp.sets import VaspInput
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
import argparse

molecules_json = Path(__file__).parent / "molecules.json"
molecules = json.loads(molecules_json.read_text())
molecules = {key: Molecule.from_dict(value) for key, value in molecules.items()}

kpoints = Kpoints.gamma_automatic((5,5,1), shift=(0,0,0))

incar_dict = {
    "ALGO": "Normal",
    "EDIFF": 1e-06,
    "EDIFFG": -0.01,
    "ENCUT": 500,
    "GGA": "Pe",
    "IBRION": 2,
    "ISIF": 2,
    "ISMEAR": 0,
    "ISYM": 2,
    "IVDW": 4,
    "IWAVPR": 1,
    "KPAR": 16,
    "LASPH": True,
    "LCHARG": True,
    "LMAXMIX": 6,
    "LORBIT": 11,
    "LREAL": "Auto",
    "LVHAR": True,
    "LWAVE": False,
    "NELM": 60,
    "NSW": 100,
    "POTIM": 0.4,
    "PREC": "Accurate",
    "SIGMA": 0.02,
    "NCORE": 64,
}

def structure_from_pubchem(cid: int, output: str = None):
    """Gets a structure from PubChem"""
    from pymatgen.core import Molecule
    import pubchempy as pcp
    compound = pcp.Compound.from_cid(cid, record_type='3d')

    #convert json data to xyz
    compound_dict = compound.to_dict(properties=['atoms'])
    symbols_and_coordinates = [(atom['element'], atom['x'], atom['y'], atom['z']) for atom in compound_dict['atoms']]
    #create a molecule
    species = [symbol for symbol, *_ in symbols_and_coordinates]
    coords = np.array([coord for *, *coord in symbols_and_coordinates])
    molecule = Molecule(species, coords)

    if not output:
        print(molecule.to(fmt="xyz"))
    else:
        molecule.to(fmt="xyz", filename=output)

def create_single_atom_molecule(element: str) -> Molecule:
    """Creates a single atom molecule from an element symbol"""
    from pymatgen.core import Element

    # Validate element symbol
    try:
        Element(element)
    except ValueError:
        raise ValueError(f"Invalid element symbol: {element}")

    # Create molecule with single atom at origin
    species = [element]
    coords = [[0.0, 0.0, 0.0]]
    molecule = Molecule(species, coords)

    return molecule

def get_molecule(molecule_input: str) -> Molecule:
    """Get molecule from either the JSON file or create a single atom molecule"""
    if molecule_input in molecules:
        return molecules[molecule_input]
    else:
        # Try to create a single atom molecule
        try:
            return create_single_atom_molecule(molecule_input)
        except ValueError as e:
            available_molecules = list(molecules.keys())
            raise ValueError(f"Invalid molecule '{molecule_input}'. "
                           f"Available molecules from JSON: {available_molecules}. "
                           f"Or provide a valid element symbol (e.g., 'Ag', 'Au', 'Pt').")

def adsorb(structure: Structure, molecule: Molecule, supercell: list[int] = [2,2,1], **find_args) -> list[Structure]:
    """generates a list of adsorbed structures"""
    finder = AdsorbateSiteFinder(structure)
    adsorbed_structures = finder.generate_adsorption_structures(molecule, repeat=supercell, find_args=find_args)
    return adsorbed_structures

def write_directories(structures: list[Structure], directory: str) -> None:
    """writes each structure to its own directory"""
    for index, structure in enumerate(structures):
        path = Path(directory) / f"adsorb_{index}"
        path.mkdir(parents=True, exist_ok=True)

        poscar = Poscar(structure)
        potcar = Potcar(symbols=poscar.site_symbols, functional="PBE")
        incar = Incar.from_dict(incar_dict)

        input_set = VaspInput(incar=incar, kpoints=kpoints, poscar=poscar, potcar=potcar)
        input_set.write_input(path)
    return None

def main():
    #create an argparser to get structure from file and molecule from command line
    parser = argparse.ArgumentParser(description="Generate adsorbed structures")
    parser.add_argument("structure", type=str, help="Path to structure file")
    parser.add_argument("molecule", type=str,
                       help="Molecule to adsorb (from JSON file) or element symbol for single atom (e.g., 'Ag', 'Au', 'Pt')")
    parser.add_argument("--supercell", type=int, nargs=3, default=[2,2,1], help="Supercell to generate")
    parser.add_argument("-d", "--distance", type=float, default=1.8,
                       help="Distance between adsorbate center of mass and surface site")
    args = parser.parse_args()

    structure = Structure.from_file(args.structure)
    try:
        molecule = get_molecule(args.molecule)
    except ValueError as e:
        print(f"Error: {e}")
        return

    #generate adsorbed structures
    adsorbed_structures = adsorb(structure, molecule, args.supercell, distance=args.distance)
    write_directories(adsorbed_structures, f"adsorbed_{args.molecule}")
    print(f"Generated {len(adsorbed_structures)} adsorbed structures for {args.molecule}")

if __name__ == "__main__":
    main()
