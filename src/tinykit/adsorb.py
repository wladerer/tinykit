import os
import json
import numpy as np
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar, Poscar
from pymatgen.io.vasp.sets import VaspInput
from pymatgen.core.structure import Structure, Molecule
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
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

    try:
        Element(element)
    except ValueError:
        raise ValueError(f"Invalid element symbol: {element}")

    return Molecule([element], [[0.0, 0.0, 0.0]])

def get_molecule(molecule_input: str) -> Molecule:
    """Get molecule from either the JSON file or create a single atom molecule"""
    if molecule_input in molecules:
        return molecules[molecule_input]
    else:
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


# ---------------------------------------------------------------------------
# Site labeling
# ---------------------------------------------------------------------------

_TYPE_PREFIX = {"ontop": "T", "bridge": "B", "hollow": "H"}

def label_adsorption_sites(slab: Structure) -> list[dict]:
    """
    Return labeled adsorption sites for a slab.

    Each entry is a dict with:
      'coord'  – Cartesian coordinates (np.ndarray)
      'type'   – 'ontop', 'bridge', or 'hollow'
      'label'  – short string like 'T0', 'B2', 'H1'
    """
    finder = AdsorbateSiteFinder(slab)
    sites = finder.find_adsorption_sites()

    labeled = []
    for site_type, prefix in _TYPE_PREFIX.items():
        for idx, coord in enumerate(sites.get(site_type, [])):
            labeled.append({
                "coord": np.array(coord),
                "type": site_type,
                "label": f"{prefix}{idx}",
            })

    return labeled


# ---------------------------------------------------------------------------
# Symmetry reduction
# ---------------------------------------------------------------------------

def symmetry_reduce_combos(
    slab: Structure,
    site_coords: np.ndarray,
    combos: list[tuple],
    symprec: float = 0.1,
) -> list[tuple]:
    """
    Return a symmetry-reduced subset of combos.

    Two combos are considered equivalent if a symmetry operation of the clean
    slab maps one set of site positions onto the other (modulo lattice
    translation in the surface plane).

    Args:
        slab:        Clean slab structure (no adsorbates).
        site_coords: Cartesian coords array of shape (N_sites, 3).
        combos:      List of index tuples into site_coords.
        symprec:     Tolerance passed to SpacegroupAnalyzer.

    Returns:
        Subset of combos with one representative per symmetry class.
    """
    analyzer = SpacegroupAnalyzer(slab, symprec=symprec)
    sym_ops = analyzer.get_symmetry_operations(cartesian=False)  # fractional

    lattice = slab.lattice
    frac_coords = np.array([lattice.get_fractional_coords(c) for c in site_coords])

    seen: set[tuple] = set()
    unique: list[tuple] = []

    for combo in combos:
        coords_frac = frac_coords[list(combo)]  # (k, 3)

        # Compute canonical key = lexicographic minimum over all sym-op images.
        canonical = None
        for op in sym_ops:
            # Apply rotation + translation in fractional space.
            transformed = (op.rotation_matrix @ coords_frac.T).T + op.translation_vector
            # Reduce xy periodically (surface plane); leave z alone.
            transformed[:, :2] %= 1.0
            key = tuple(map(tuple, np.round(np.sort(transformed, axis=0), decimals=3)))
            if canonical is None or key < canonical:
                canonical = key

        if canonical not in seen:
            seen.add(canonical)
            unique.append(combo)

    return unique


# ---------------------------------------------------------------------------
# Multi-adsorbate sampling
# ---------------------------------------------------------------------------

def adsorb_sampling(
    slab: Structure,
    molecule: Molecule,
    multiplicity: int,
    min_distance: float = 2.0,
    supercell: list[int] = None,
    symmetry_reduce: bool = True,
    symprec: float = 0.1,
) -> tuple[list[Structure], list[str]]:
    """
    Generate all (symmetry-reduced) structures with `multiplicity` adsorbates.

    Site coordinates are determined once from the clean (optionally supercelled)
    slab and then used for all combinations — the AdsorbateSiteFinder is NOT
    rebuilt between placements, which avoids drifting site positions.

    Returns:
        (structures, combo_labels) where combo_labels[i] is a string like
        'H0+T2' describing which sites were used in structures[i].
    """
    if supercell is not None:
        slab = slab.copy()
        slab.make_supercell(supercell)

    labeled_sites = label_adsorption_sites(slab)
    if not labeled_sites:
        return [], []

    site_coords = np.array([s["coord"] for s in labeled_sites])
    site_labels = [s["label"] for s in labeled_sites]

    # All combinations of `multiplicity` distinct sites.
    all_combos = list(combinations(range(len(site_coords)), multiplicity))

    # Distance filter.
    valid_combos = [
        combo for combo in all_combos
        if _check_site_distances(site_coords[list(combo)], min_distance)
    ]

    # Symmetry reduction.
    if symmetry_reduce and len(valid_combos) > 1:
        valid_combos = symmetry_reduce_combos(slab, site_coords, valid_combos, symprec=symprec)

    # Build structures.
    structures = []
    combo_labels = []

    for combo in valid_combos:
        struct = slab.copy()
        asf = AdsorbateSiteFinder(struct)

        for idx in combo:
            struct = asf.add_adsorbate(molecule, site_coords[idx])
            # Rebuild ASF on updated struct so surface detection stays correct,
            # but keep using the original site_coords (from the clean slab).
            asf = AdsorbateSiteFinder(struct)

        structures.append(struct)
        label = "+".join(site_labels[i] for i in sorted(combo))
        combo_labels.append(label)

    return structures, combo_labels


def _check_site_distances(coords: np.ndarray, min_distance: float) -> bool:
    """Return True if all pairwise distances between coords exceed min_distance."""
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if np.linalg.norm(coords[i] - coords[j]) < min_distance:
                return False
    return True


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_directories(
    structures: list[Structure],
    directory: str,
    names: list[str] = None,
    reference_incar_path: str = None,
) -> None:
    """
    Write each structure to its own subdirectory.

    Args:
        structures:          List of structures to write.
        directory:           Parent directory path.
        names:               Per-structure subdirectory names. Falls back to
                             'adsorb_{index}' if not provided.
        reference_incar_path: Path to a reference INCAR; uses built-in defaults
                             if None.
    """
    for index, structure in enumerate(structures):
        subdir_name = names[index] if (names and index < len(names)) else f"adsorb_{index}"
        path = Path(directory) / subdir_name
        path.mkdir(parents=True, exist_ok=True)

        poscar = Poscar(structure)
        potcar = Potcar(symbols=poscar.site_symbols, functional="PBE")

        if reference_incar_path:
            ref_path = Path(reference_incar_path).resolve(strict=True)
            incar = Incar.from_file(ref_path)
        else:
            incar = Incar.from_dict(incar_dict)

        input_set = VaspInput(incar=incar, kpoints=kpoints, poscar=poscar, potcar=potcar)
        input_set.write_input(path)

def main():
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
    parser.add_argument("--no-symmetry-reduce", action="store_true",
                       help="Skip symmetry reduction for multiple adsorbate sampling")
    args = parser.parse_args()

    structure = Structure.from_file(args.structure)
    try:
        molecule = get_molecule(args.molecule)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.multiple:
        structures, combo_labels = adsorb_sampling(
            structure,
            molecule,
            args.multiple,
            min_distance=args.min_distance,
            supercell=args.supercell if args.supercell != [1,1,1] else None,
            symmetry_reduce=not args.no_symmetry_reduce,
        )

        mol_tag = args.molecule
        parent_dir = f"adsorbed_{mol_tag}_x{args.multiple}"
        # Subdirectory names encode the molecule, multiplicity, and site combo.
        names = [f"{mol_tag}_x{args.multiple}_{label}" for label in combo_labels]

        write_directories(structures, parent_dir, names=names, reference_incar_path=args.incar)
        print(f"Generated {len(structures)} structures with {args.multiple} {args.molecule} adsorbates")
        for name in names:
            print(f"  {parent_dir}/{name}")

    else:
        adsorbed_structures = adsorb(
            structure,
            molecule,
            args.supercell,
            distance=args.distance,
        )
        write_directories(adsorbed_structures, f"adsorbed_{args.molecule}", reference_incar_path=args.incar)
        print(f"Generated {len(adsorbed_structures)} adsorbed structures for {args.molecule}")

if __name__ == "__main__":
    main()
