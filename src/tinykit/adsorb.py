import os
import json
import numpy as np
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar, Poscar
from pymatgen.io.vasp.sets import VaspInput
from pymatgen.core.structure import Structure, Molecule
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

def adsorb_sampling(structure: Structure, molecule: Molecule, multiplicity: int, 
                   distance: float = 2.0, positions=('ontop', 'bridge', 'hollow'),
                   max_samples: int = None, random_seed: int = None):
    """
    Generate all unique combinations of adsorption sites for a given multiplicity.
    Optionally sample a random subset if the total number is too large.
    
    Args:
        structure: The slab structure
        molecule: The molecule to adsorb
        multiplicity: Number of adsorbates to place simultaneously
        distance: Minimum distance between adsorbates
        positions: Types of adsorption sites to consider
        max_samples: Maximum number of structures to generate (None for all)
        random_seed: Random seed for reproducible sampling (None for random)
    
    Returns:
        list: All unique structures with adsorbates placed (or random sample)
    """
    # Set random seed for reproducibility if specified
    if random_seed is not None:
        random.seed(random_seed)
    
    finder = AdsorbateSiteFinder(structure)
    sites_dict = finder.find_adsorption_sites(
        distance=distance, 
        put_inside=True, 
        symm_reduce=0.01, 
        near_reduce=0.01, 
        positions=positions, 
        no_obtuse_hollow=True
    )
    
    # Flatten all sites into a single list with site type information
    all_sites = []
    for site_type, site_coords in sites_dict.items():
        for coord in site_coords:
            all_sites.append({
                'coord': coord,
                'type': site_type,
                'id': len(all_sites)  # Unique identifier for each site
            })
    
    print(f"Found {len(all_sites)} total adsorption sites")
    for site_type in positions:
        count = len(sites_dict.get(site_type, []))
        print(f"  {site_type}: {count} sites")
    
    # Generate all unique combinations for the given multiplicity
    if multiplicity > len(all_sites):
        print(f"Warning: Multiplicity ({multiplicity}) exceeds number of available sites ({len(all_sites)})")
        return []
    
    # Get all combinations without repetition
    site_combinations = list(combinations(all_sites, multiplicity))
    total_combinations = len(site_combinations)
    print(f"Total possible combinations for multiplicity {multiplicity}: {total_combinations}")
    
    # Sample random subset if max_samples is specified and we have more combinations than requested
    if max_samples is not None and total_combinations > max_samples:
        site_combinations = random.sample(site_combinations, max_samples)
        print(f"Randomly sampled {max_samples} combinations out of {total_combinations}")
    else:
        print(f"Using all {total_combinations} combinations")
    
    # Create structures for each combination
    structures = []
    for i, site_combo in enumerate(site_combinations):
        try:
            # Start with a fresh copy of the original structure for each combination
            new_structure = structure.copy()
            
            # Check if sites are too close to each other (optional distance check)
            if multiplicity > 1:
                coords = [site['coord'] for site in site_combo]
                if not _check_site_distances(coords, min_distance=distance):
                    continue  # Skip this combination if sites are too close
            
            # Create a fresh finder for this combination
            combination_finder = AdsorbateSiteFinder(new_structure)
            
            # Add adsorbates to all sites in this combination
            for site in site_combo:
                new_structure = combination_finder.add_adsorbate(
                    molecule=molecule,
                    ads_coord=site['coord'],
                    repeat=None,
                    translate=True,
                    reorient=True
                )
                # Update finder with the new structure for the next adsorbate in this combination
                combination_finder = AdsorbateSiteFinder(new_structure)
            
            # Store structure with metadata
            structure_info = {
                'structure': new_structure,
                'sites': site_combo,
                'combination_id': i,
                'site_types': [site['type'] for site in site_combo],
                'site_coords': [site['coord'] for site in site_combo]
            }
            structures.append(structure_info)
            
        except Exception as e:
            print(f"Warning: Failed to create structure for combination {i}: {str(e)}")
            continue
    
    print(f"Successfully created {len(structures)} structures")
    return structures


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

def extract_structures_only(structure_info_list: list) -> list[Structure]:
    """
    Extract just the Structure objects from the detailed structure info list.
    
    Args:
        structure_info_list: List of dictionaries containing structure info
    
    Returns:
        list: List of Structure objects only
    """
    return [info['structure'] for info in structure_info_list]

def filter_by_site_type(structure_info_list: list, allowed_types: list[str]) -> list:
    """
    Filter structures to only include those with specified site types.
    
    Args:
        structure_info_list: List of structure info dictionaries
        allowed_types: List of site types to keep (e.g., ['ontop', 'bridge'])
    
    Returns:
        list: Filtered structure info list
    """
    filtered = []
    for info in structure_info_list:
        if all(site_type in allowed_types for site_type in info['site_types']):
            filtered.append(info)
    return filtered


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
    parser.add_argument("--supercell", type=int, nargs=3, default=[2,2,1], help="Supercell to generate")
    parser.add_argument("-d", "--distance", type=float, default=1.8,
                       help="Distance between adsorbate center of mass and surface site")
    parser.add_argument("--incar", type=str, default=None, help="Path to reference INCAR file")
    parser.add_argument("--multiple", type=int, default=None, help="Number of adsorbates to add")
    parser.add_argument("--min-distance", type=float, default=2.0, 
                       help="Minimum distance between adsorbates in multiple adsorption")
    parser.add_argument("--max-samples", type=int, default=None,
                          help="Maximum number of structures to generate for multiple adsorption (None for all)")
    parser.add_argument("--random-seed", type=int, default=None,
                            help="Random seed for reproducible sampling (None for random)")
    parser.add_argument("--sites", type=str, nargs='+', default=['ontop', 'bridge', 'hollow'],
                       choices=['ontop', 'bridge', 'hollow'],
                       help="Types of adsorption sites to consider (e.g., --sites ontop bridge)")
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
        structure_info_list = adsorb_sampling(
            structure, 
            molecule, 
            args.multiple, 
            max_samples=args.max_samples,
            random_seed=args.random_seed,
            distance=args.min_distance,
            positions=tuple(args.sites)  # Use the specified site types
        )
        
        # Optional: Further filter by site types if needed
        # structure_info_list = filter_by_site_type(structure_info_list, args.sites)
        
        # Extract just the structures for writing directories
        adsorbed_structures = extract_structures_only(structure_info_list)
        
        # Write structures to directories
        directory_name = f"adsorbed_{args.molecule}_x{args.multiple}"
        write_directories(adsorbed_structures, directory_name, reference_incar_path=args.incar)
        
        print(f"Generated {len(adsorbed_structures)} structures with {args.multiple} {args.molecule} adsorbates")
        
        # Print summary of site types used
        site_type_summary = {}
        for info in structure_info_list:
            site_types_key = tuple(sorted(info['site_types']))
            site_type_summary[site_types_key] = site_type_summary.get(site_types_key, 0) + 1
        
        print("\nSite type combinations:")
        for site_combo, count in site_type_summary.items():
            print(f"  {site_combo}: {count} structures")
    
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
