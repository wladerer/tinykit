import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar, Poscar
from pymatgen.io.vasp.sets import VaspInput
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from itertools import combinations
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


def create_single_atom_molecule(element: str) -> Molecule:
    from pymatgen.core import Element
    try:
        Element(element)
    except ValueError:
        raise ValueError(f"Invalid element symbol: {element}")
    return Molecule([element], [[0.0, 0.0, 0.0]])


def get_molecule(molecule_input: str) -> Molecule:
    if molecule_input in molecules:
        return molecules[molecule_input]
    try:
        return create_single_atom_molecule(molecule_input)
    except ValueError:
        available = list(molecules.keys())
        raise ValueError(
            f"Invalid molecule '{molecule_input}'. "
            f"Available molecules from JSON: {available}. "
            f"Or provide a valid element symbol (e.g., 'Ag', 'Au', 'Pt')."
        )


def adsorb(structure: Structure, molecule: Molecule, supercell: list[int] = None, **find_args) -> list[Structure]:
    """Single-adsorbate: generate all symmetry-unique adsorbed structures."""
    finder = AdsorbateSiteFinder(structure)
    return finder.generate_adsorption_structures(molecule, repeat=supercell, find_args=find_args)


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
# Parallel workers (module-level for pickling)
# ---------------------------------------------------------------------------

def _compute_canonical_key(args: tuple) -> tuple:
    """
    Compute the canonical symmetry key for one combo of fractional site coords.

    Receives plain numpy data (not SymmOp objects) so pickling is robust.
    args = (coords_frac, [(rotation_matrix, translation_vector), ...])
    """
    coords_frac, sym_ops_data = args
    canonical = None
    for rot, trans in sym_ops_data:
        transformed = (rot @ coords_frac.T).T + trans
        transformed[:, :2] %= 1.0  # periodic in surface plane only
        key = tuple(map(tuple, np.round(np.sort(transformed, axis=0), decimals=3)))
        if canonical is None or key < canonical:
            canonical = key
    return canonical


def _build_structure_for_combo(args: tuple) -> Structure | None:
    """
    Build a slab+adsorbates structure for one site combo and check for atomic
    overlaps between different adsorbate molecules.

    Returns None if any inter-adsorbate atom pair is closer than min_atom_distance.
    args = (slab, molecule, site_coords, combo, min_atom_distance)
    """
    slab, molecule, site_coords, combo, min_atom_distance = args

    struct = slab.copy()
    asf = AdsorbateSiteFinder(struct)

    adsorbate_ranges: list[tuple[int, int]] = []

    for idx in combo:
        start = len(struct)
        struct = asf.add_adsorbate(molecule, site_coords[idx])
        adsorbate_ranges.append((start, len(struct)))
        asf = AdsorbateSiteFinder(struct)

    # Check for atomic overlaps between distinct adsorbate molecules.
    for i, (s1, e1) in enumerate(adsorbate_ranges):
        for j, (s2, e2) in enumerate(adsorbate_ranges):
            if j <= i:
                continue
            for a1 in range(s1, e1):
                for a2 in range(s2, e2):
                    if struct.get_distance(a1, a2) < min_atom_distance:
                        return None

    return struct


# ---------------------------------------------------------------------------
# Symmetry reduction
# ---------------------------------------------------------------------------

def symmetry_reduce_combos(
    slab: Structure,
    site_coords: np.ndarray,
    combos: list[tuple],
    symprec: float = 0.1,
    n_workers: int = 1,
) -> list[tuple]:
    """
    Return a symmetry-reduced subset of combos.

    Two combos are equivalent if a symmetry operation of the clean slab maps
    one set of site positions onto the other (modulo xy lattice translation).
    Canonical key computation is parallelized across workers.
    """
    analyzer = SpacegroupAnalyzer(slab, symprec=symprec)
    sym_ops = analyzer.get_symmetry_operations(cartesian=False)
    # Extract plain arrays so workers don't need to pickle SymmOp objects.
    sym_ops_data = [(op.rotation_matrix, op.translation_vector) for op in sym_ops]

    lattice = slab.lattice
    frac_coords = np.array([lattice.get_fractional_coords(c) for c in site_coords])

    combo_args = [(frac_coords[list(c)], sym_ops_data) for c in combos]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            keys = list(ex.map(_compute_canonical_key, combo_args))
    else:
        keys = [_compute_canonical_key(a) for a in combo_args]

    seen: set[tuple] = set()
    unique: list[tuple] = []
    for combo, key in zip(combos, keys):
        if key not in seen:
            seen.add(key)
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
    min_atom_distance: float = 1.5,
    supercell: list[int] = None,
    symmetry_reduce: bool = True,
    symprec: float = 0.1,
    n_workers: int = 1,
) -> tuple[list[Structure], list[str]]:
    """
    Generate all (symmetry-reduced) structures with `multiplicity` adsorbates.

    Pipeline:
      1. Find and label adsorption sites on the clean slab.
      2. Generate all C(N_sites, multiplicity) combos.
      3. Filter by minimum site-to-site distance (fast pre-filter).
      4. Symmetry-reduce the remaining combos (parallelized).
      5. Build structures and reject any with inter-adsorbate atomic overlaps
         (parallelized).

    Args:
        min_distance:      Minimum Cartesian distance between adsorption site
                           anchor points (Å). Fast pre-filter.
        min_atom_distance: Minimum allowed distance between atoms of different
                           adsorbate molecules (Å). Catches overlapping
                           polyatomic adsorbates that pass the site filter.
        n_workers:         Number of parallel worker processes.

    Returns:
        (structures, combo_labels) — combo_labels[i] is like 'H0+T2'.
    """
    if supercell is not None:
        slab = slab.copy()
        slab.make_supercell(supercell)

    labeled_sites = label_adsorption_sites(slab)
    if not labeled_sites:
        return [], []

    site_coords = np.array([s["coord"] for s in labeled_sites])
    site_labels = [s["label"] for s in labeled_sites]

    all_combos = list(combinations(range(len(site_coords)), multiplicity))

    # Fast site-distance pre-filter.
    valid_combos = [
        c for c in all_combos
        if _check_site_distances(site_coords[list(c)], min_distance)
    ]

    # Symmetry reduction.
    if symmetry_reduce and len(valid_combos) > 1:
        valid_combos = symmetry_reduce_combos(
            slab, site_coords, valid_combos, symprec=symprec, n_workers=n_workers
        )

    # Build structures in parallel, filtering atomic overlaps.
    build_args = [
        (slab, molecule, site_coords, combo, min_atom_distance)
        for combo in valid_combos
    ]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_build_structure_for_combo, build_args))
    else:
        results = [_build_structure_for_combo(a) for a in build_args]

    structures = []
    combo_labels = []
    for combo, struct in zip(valid_combos, results):
        if struct is not None:
            structures.append(struct)
            combo_labels.append("+".join(site_labels[i] for i in sorted(combo)))

    return structures, combo_labels


def _check_site_distances(coords: np.ndarray, min_distance: float) -> bool:
    """Return True if all pairwise distances between site coords exceed min_distance."""
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
    """Write each structure to its own subdirectory with VASP input files."""
    for index, structure in enumerate(structures):
        subdir_name = names[index] if (names and index < len(names)) else f"adsorb_{index}"
        path = Path(directory) / subdir_name
        path.mkdir(parents=True, exist_ok=True)

        poscar = Poscar(structure)
        potcar = Potcar(symbols=poscar.site_symbols, functional="PBE")

        if reference_incar_path:
            incar = Incar.from_file(Path(reference_incar_path).resolve(strict=True))
        else:
            incar = Incar.from_dict(incar_dict)

        VaspInput(incar=incar, kpoints=kpoints, poscar=poscar, potcar=potcar).write_input(path)


def main():
    parser = argparse.ArgumentParser(description="Generate adsorbed structures")
    parser.add_argument("structure", type=str, help="Path to structure file")
    parser.add_argument("molecule", type=str,
                        help="Molecule to adsorb (from JSON) or element symbol (e.g. 'CO', 'Au')")
    parser.add_argument("--supercell", type=int, nargs=3, default=[1,1,1])
    parser.add_argument("-d", "--distance", type=float, default=1.8,
                        help="Adsorbate height above surface site (single-adsorbate mode, Å)")
    parser.add_argument("--incar", type=str, default=None, help="Path to reference INCAR file")
    parser.add_argument("--multiple", type=int, default=None,
                        help="Number of adsorbates to place simultaneously")
    parser.add_argument("--min-distance", type=float, default=2.0,
                        help="Minimum distance between adsorption site anchors (Å)")
    parser.add_argument("--min-atom-distance", type=float, default=1.5,
                        help="Minimum allowed distance between atoms of different adsorbates (Å)")
    parser.add_argument("--no-symmetry-reduce", action="store_true",
                        help="Skip symmetry reduction")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
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
            min_atom_distance=args.min_atom_distance,
            supercell=args.supercell if args.supercell != [1,1,1] else None,
            symmetry_reduce=not args.no_symmetry_reduce,
            n_workers=args.jobs,
        )

        mol_tag = args.molecule
        parent_dir = f"adsorbed_{mol_tag}_x{args.multiple}"
        names = [f"{mol_tag}_x{args.multiple}_{label}" for label in combo_labels]

        write_directories(structures, parent_dir, names=names, reference_incar_path=args.incar)
        print(f"Generated {len(structures)} structures with {args.multiple} {args.molecule} adsorbates")
        for name in names:
            print(f"  {parent_dir}/{name}")

    else:
        adsorbed_structures = adsorb(structure, molecule, args.supercell, distance=args.distance)
        write_directories(adsorbed_structures, f"adsorbed_{args.molecule}", reference_incar_path=args.incar)
        print(f"Generated {len(adsorbed_structures)} adsorbed structures for {args.molecule}")


if __name__ == "__main__":
    main()
