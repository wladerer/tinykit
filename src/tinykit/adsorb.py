import os
import json
import numpy as np
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar, Poscar
from pymatgen.io.vasp.sets import VaspInput
from pymatgen.core.structure import Structure, Molecule
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from itertools import combinations
import argparse
import random

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

def adsorb(structure: Structure, molecule: Molecule, supercell: list[int,int,int] = None, **find_args) -> list[Structure]:
    """generates a list of adsorbed structures"""
    finder = AdsorbateSiteFinder(structure)
    adsorbed_structures = finder.generate_adsorption_structures(molecule, repeat=supercell, find_args=find_args)

    return adsorbed_structures

def adsorb_sampling(
    slab: Structure,
    molecule: Molecule,
    multiplicity: int,
    distance: float = 1.5,
    supercell=None
):
    if supercell is not None:
        slab = slab.copy()
        slab.make_supercell(supercell)

    finder = AdsorbateSiteFinder(slab)
    sites = finder.find_adsorption_sites()

    if "all_positions" in sites:
        site_coords = np.array(sites["all_positions"])
    else:
        site_coords = np.array(sites["all"])

    combos = combinations(range(len(site_coords)), multiplicity)
    final_structures = []

    for combo in combos:
        coords = site_coords[list(combo)]

        # distance filtering
        ok = True
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                if np.linalg.norm(coords[i] - coords[j]) < distance:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        # build structure with multiple adsorbates
        struct = slab.copy()

        for idx in combo:
            asf = AdsorbateSiteFinder(struct)  # IMPORTANT: rebuild each time
            struct = asf.add_adsorbate(molecule, site_coords[idx])

        final_structures.append(struct)

    return final_structures


def _check_site_distances(coords: list[np.ndarray], min_distance: float) -> bool:
    """
    Check if all sites are at least min_distance apart.
    
    Args:
        coords: List of coordinate arrays
        min_distance: Minimum allowed distance between sites
    
    Returns:
        bool: True if all distances are acceptable, False otherwise
    """
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < min_distance:
                return False
    return True


def write_directories(structures: list[Structure], directory: str, reference_incar_path: str = None) -> None:
    """writes each structure to its own directory"""
    for index, structure in enumerate(structures):
        path = Path(directory) / f"adsorb_{index}"
        path.mkdir(parents=True, exist_ok=True)

        poscar = Poscar(structure)
        potcar = Potcar(symbols=poscar.site_symbols, functional="PBE")

        if reference_incar_path:
            try:
                # Convert to Path object and resolve
                ref_path = Path(reference_incar_path)
                resolved_path = ref_path.resolve(strict=True)
                incar = Incar.from_file(resolved_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Reference INCAR file not found: {reference_incar_path}")
        else:
            incar = Incar.from_dict(incar_dict)

        input_set = VaspInput(incar=incar, kpoints=kpoints, poscar=poscar, potcar=potcar)
        input_set.write_input(path)

def main():
    #create an argparser to get structure from file and molecule from command line
    parser = argparse.ArgumentParser(description="Generate adsorbed structures")
    parser.add_argument("structure", type=str, help="Path to structure file")
    parser.add_argument("molecule", type=str,
                       help="Molecule to adsorb (from JSON file) or element symbol for single atom (e.g., 'Ag', 'Au', 'Pt')")
    parser.add_argument("--supercell", type=int, nargs=3, default=[1,1,1], help="Supercell to generate")
    parser.add_argument("-d", "--distance", type=float, default=1.8,
                       help="Distance between adsorbate center of mass and surface site")
    parser.add_argument("--incar", type=str, default=None, help="Path to reference INCAR file")
    parser.add_argument("--multiple", type=int, default=None, help="Number of adsorbates to add")
    parser.add_argument("--min-distance", type=float, default=2.0, 
                       help="Minimum distance between adsorbates in multiple adsorption")
    args = parser.parse_args()

    structure = Structure.from_file(args.structure)
    try:
        molecule = get_molecule(args.molecule)
    except ValueError as e:
        print(f"Error: {e}")
        return

    #generate adsorbed structures
    if args.multiple:
        # Use comprehensive sampling for multiple adsorptions
        adsorbed_structures = adsorb_sampling(
            structure, 
            molecule, 
            args.multiple, 
            distance=args.min_distance,
            supercell=args.supercell,
        )
        
        directory_name = f"adsorbed_{args.molecule}_x{args.multiple}"
        write_directories(adsorbed_structures, directory_name, reference_incar_path=args.incar)
        
        print(f"Generated {len(adsorbed_structures)} structures with {args.multiple} {args.molecule} adsorbates")
        
    else:
        # Use original single adsorption method
        adsorbed_structures = adsorb(
            structure, 
            molecule, 
            args.supercell, 
            distance=args.distance
        )
        write_directories(adsorbed_structures, f"adsorbed_{args.molecule}", reference_incar_path=args.incar)
        print(f"Generated {len(adsorbed_structures)} adsorbed structures for {args.molecule}")

if __name__ == "__main__":
    main()
