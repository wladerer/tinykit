import argparse
import warnings
import json
import logging
import numpy as np
from enum import Enum
from pathlib import Path
from datetime import datetime

from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab, SlabGenerator, generate_all_slabs


# Configure logging
def setup_logging(verbose: bool = False):
    """
    Set up logging configuration.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Show level in console when verbose for clarity
    if verbose:
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    else:
        console_formatter = logging.Formatter('%(message)s')

    logger = logging.getLogger()
    
    return logger


class FreezingMode(Enum):
    """Modes for applying selective dynamics to slabs."""
    CENTER = "center"  # Relax top and bottom, freeze center
    BOTTOM = "bottom"  # Freeze bottom layers, relax top
    TOP = "top"        # Freeze top layers, relax bottom


kpoints = Kpoints.gamma_automatic((1, 1, 1), shift=(0, 0, 0))

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


def apply_selective_dynamics(
    slab: Slab,
    n_layers: int,
    mode: FreezingMode = FreezingMode.CENTER,
) -> Slab:
    """
    Apply selective dynamics to a slab structure with different freezing strategies.
    
    Args:
        slab: Pymatgen Slab object
        n_layers: Number of layers to relax/freeze (interpretation depends on mode)
            - CENTER: Number of layers to relax at top and bottom
            - BOTTOM: Number of bottom layers to freeze
            - TOP: Number of top layers to freeze
        mode: Freezing strategy (default: CENTER)
        
    Returns:
        Slab with selective dynamics applied
    """
    logger = logging.getLogger()
    
    # Get surface normal
    normal = slab.normal
    
    # Project atomic positions onto surface normal
    positions = np.array([site.coords for site in slab.sites])
    projections = np.dot(positions, normal)
    
    # Identify distinct layers (decimals=0 for coarse grouping)
    unique_projections = np.unique(np.round(projections, decimals=0))
    unique_projections.sort()
    nlayers = len(unique_projections)
    
    logger.debug(f"  Detected {nlayers} distinct layers in slab")
    logger.debug(f"  Applying {mode.value} mode with n_layers={n_layers}")
    
    # Validate layer count based on mode
    if mode == FreezingMode.CENTER:
        if nlayers <= 2 * n_layers:
            msg = (
                f"Slab has {nlayers} layers but {2*n_layers} layers "
                f"requested for relaxation in {mode.value} mode. "
                f"Skipping selective dynamics."
            )
            logger.warning(f"  {msg}")
            warnings.warn(msg)
            return slab
    else:  # BOTTOM or TOP
        if n_layers >= nlayers:
            msg = (
                f"Slab has {nlayers} layers but {n_layers} layers "
                f"requested for freezing in {mode.value} mode. "
                f"Skipping selective dynamics."
            )
            logger.warning(f"  {msg}")
            warnings.warn(msg)
            return slab
    
    # Count atoms that will be relaxed/frozen
    relaxed_count = 0
    frozen_count = 0
    
    # Apply selective dynamics based on mode
    if mode == FreezingMode.CENTER:
        bottom_threshold = unique_projections[n_layers - 1]
        top_threshold = unique_projections[nlayers - n_layers]
        
        logger.debug(f"  Bottom threshold: {bottom_threshold:.2f}, Top threshold: {top_threshold:.2f}")
        
        for site in slab.sites:
            proj = np.round(np.dot(site.coords, normal), decimals=0)
            # Relax top and bottom layers
            if proj <= bottom_threshold or proj >= top_threshold:
                site.properties["selective_dynamics"] = [True, True, True]
                relaxed_count += 1
            else:
                site.properties["selective_dynamics"] = [False, False, False]
                frozen_count += 1
    
    elif mode == FreezingMode.BOTTOM:
        freeze_threshold = unique_projections[n_layers - 1]
        logger.debug(f"  Freeze threshold: {freeze_threshold:.2f}")
        
        for site in slab.sites:
            proj = np.round(np.dot(site.coords, normal), decimals=0)
            # Freeze bottom layers, relax upper layers
            if proj <= freeze_threshold:
                site.properties["selective_dynamics"] = [False, False, False]
                frozen_count += 1
            else:
                site.properties["selective_dynamics"] = [True, True, True]
                relaxed_count += 1
    
    elif mode == FreezingMode.TOP:
        freeze_threshold = unique_projections[nlayers - n_layers]
        logger.debug(f"  Freeze threshold: {freeze_threshold:.2f}")
        
        for site in slab.sites:
            proj = np.round(np.dot(site.coords, normal), decimals=0)
            # Freeze top layers, relax lower layers
            if proj >= freeze_threshold:
                site.properties["selective_dynamics"] = [False, False, False]
                frozen_count += 1
            else:
                site.properties["selective_dynamics"] = [True, True, True]
                relaxed_count += 1
    
    logger.debug(f"  Relaxed atoms: {relaxed_count}, Frozen atoms: {frozen_count}")
    
    return slab


def generate_slabs_from_miller(
    structure: Structure,
    miller_index: tuple,
    min_slab_size: float,
    min_vacuum_size: float,
    symmetrize: bool = False,
    lll_reduce: bool = True,
    center_slab: bool = True,
    in_unit_planes: bool = False,
    primitive: bool = True,
) -> list[Slab]:
    """
    Generate slabs for a specific Miller index.
    
    Args:
        structure: Input structure
        miller_index: Specific Miller index tuple, e.g., (2, 0, 1) or (-2, 0, 1)
        min_slab_size: Minimum slab thickness (Angstroms or unit planes depending on in_unit_planes)
        min_vacuum_size: Minimum vacuum thickness in Angstroms (always)
        symmetrize: Whether to make top and bottom surfaces equivalent
        lll_reduce: Whether to perform LLL reduction
        center_slab: Whether to center the slab
        in_unit_planes: Whether to use lattice planes for slab thickness (vacuum always in Angstroms)
        primitive: Whether to reduce to primitive cell
        
    Returns:
        List of Slab objects
    """
    logger = logging.getLogger()
    logger.debug(f"  Creating SlabGenerator for {miller_index}, symmetrize={symmetrize}")
    
    # Create SlabGenerator - it always uses Angstroms for vacuum
    # The in_unit_planes parameter only affects min_slab_size interpretation
    slabgen = SlabGenerator(
        initial_structure=structure,
        miller_index=miller_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        lll_reduce=lll_reduce,
        in_unit_planes=in_unit_planes,
        center_slab=center_slab,
        primitive=primitive,
    )
    
    slabs = slabgen.get_slabs(symmetrize=symmetrize)
    logger.debug(f"  Generated {len(slabs)} slabs")
    
    return slabs


def write_slab_directories(
    slabs: list[Slab],
    directory: str,
    min_slab_size: float,
    layers_to_relax: int = None,
    freeze_mode: FreezingMode = FreezingMode.CENTER,
    overwrite: bool = True,
) -> int:
    """
    Write VASP input files for slabs to directory structure.
    
    Args:
        slabs: List of Slab objects
        directory: Parent directory for output
        min_slab_size: Minimum slab size (for directory naming)
        layers_to_relax: Number of layers (interpretation depends on freeze_mode)
        freeze_mode: Freezing mode for selective dynamics
        overwrite: Whether to overwrite existing directories
        
    Returns:
        Number of slabs successfully written
    """
    logger = logging.getLogger()
    written_count = 0
    skipped_duplicate = 0
    skipped_existing = 0
    failed_count = 0
    root = Path(directory)
    
    # Track unique structures using a hash of atomic positions
    seen_structures = {}

    for termination_index, slab in enumerate(slabs):
        logger.debug(f"\nProcessing termination {termination_index}:")
        logger.debug(f"  Miller index: {slab.miller_index}")
        logger.debug(f"  Formula: {slab.composition.reduced_formula}")
        logger.debug(f"  Num atoms: {len(slab)}")
        logger.debug(f"  Is symmetric: {slab.is_symmetric()}")
        
        # Apply selective dynamics BEFORE sorting to preserve layer structure
        if layers_to_relax is not None:
            logger.debug(f"  Applying selective dynamics...")
            slab = apply_selective_dynamics(slab, layers_to_relax, mode=freeze_mode)
        
        # Now sort by species for consistent POSCAR formatting
        # Note: Coordinates don't change, just the order in the file
        slab = slab.get_sorted_structure(key=lambda s: s.species_string)

        # Create a unique identifier for this slab based on structure
        # Use fractional coordinates and species to identify duplicates
        structure_hash = _get_structure_hash(slab)
        
        # Check if we've seen this structure before
        if structure_hash in seen_structures:
            original_term = seen_structures[structure_hash]
            msg = f"Skipping duplicate: term_{termination_index} is identical to term_{original_term}"
            logger.info(f"  {msg}")
            warnings.warn(msg)
            skipped_duplicate += 1
            continue
        
        seen_structures[structure_hash] = termination_index

        sym_dir = "sym" if slab.is_symmetric() else "asym"
        hkl = "".join(str(i) for i in slab.miller_index)
        size_str = f"{min_slab_size:.2f}"

        path = root / hkl / size_str / sym_dir / f"term_{termination_index}"
        
        # Check for existing directory
        if path.exists() and not overwrite:
            msg = f"Directory {path} already exists. Skipping."
            logger.info(f"  {msg}")
            warnings.warn(msg)
            skipped_existing += 1
            continue

        logger.debug(f"  Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)

        try:
            logger.debug(f"  Writing VASP input files...")
            poscar = Poscar(slab)
            potcar = Potcar(symbols=poscar.site_symbols, functional="PBE")
            incar = Incar.from_dict(incar_dict)

            input_set = VaspInput(
                incar=incar,
                kpoints=kpoints,
                poscar=poscar,
                potcar=potcar,
            )
            input_set.write_input(path)

            with open(path / "slab.json", "w") as f:
                json.dump(slab.as_dict(), f, indent=2)
                
            logger.info(f"  ✓ Written term_{termination_index} to {path.relative_to(root)}")
            written_count += 1

        except Exception as e:
            msg = f"Failed to write files to {path}: {e}"
            logger.error(f"  {msg}")
            warnings.warn(msg)
            failed_count += 1
            continue

    # Summary
    logger.info(f"\nWrite summary:")
    logger.info(f"  Successfully written: {written_count}")
    if skipped_duplicate > 0:
        logger.info(f"  Skipped (duplicates): {skipped_duplicate}")
    if skipped_existing > 0:
        logger.info(f"  Skipped (existing): {skipped_existing}")
    if failed_count > 0:
        logger.info(f"  Failed: {failed_count}")

    return written_count


def _get_structure_hash(structure: Structure, decimals: int = 4) -> str:
    """
    Generate a hash string for a structure based on its atomic positions and species.
    
    Args:
        structure: Structure to hash
        decimals: Decimal precision for coordinate comparison
        
    Returns:
        Hash string uniquely identifying the structure
    """
    logger = logging.getLogger()
    
    # Get fractional coordinates and species
    frac_coords = structure.frac_coords
    species = [site.species_string for site in structure]
    
    # Round coordinates to avoid floating point issues
    rounded_coords = np.round(frac_coords, decimals=decimals)
    
    # Sort by species then coordinates for consistent ordering
    sorted_indices = np.lexsort((
        rounded_coords[:, 2],  # z
        rounded_coords[:, 1],  # y
        rounded_coords[:, 0],  # x
        species,               # species
    ))
    
    # Create hash from sorted positions and species
    hash_parts = []
    for idx in sorted_indices:
        coord = rounded_coords[idx]
        spec = species[idx]
        hash_parts.append(f"{spec}_{coord[0]:.{decimals}f}_{coord[1]:.{decimals}f}_{coord[2]:.{decimals}f}")
    
    # Include lattice parameters in hash
    lattice = structure.lattice
    lattice_str = (
        f"{lattice.a:.{decimals}f}_{lattice.b:.{decimals}f}_{lattice.c:.{decimals}f}_"
        f"{lattice.alpha:.{decimals}f}_{lattice.beta:.{decimals}f}_{lattice.gamma:.{decimals}f}"
    )
    
    hash_string = lattice_str + "_" + "_".join(hash_parts)
    logger.debug(f"  Structure hash: {hash_string[:80]}..." if len(hash_string) > 80 else f"  Structure hash: {hash_string}")
    
    return hash_string


def parse_miller_index(miller_str: str) -> tuple:
    """
    Parse a Miller index string into a tuple.
    
    Args:
        miller_str: String like '201' or '-201' or '2,0,1' or '-2,0,1'
        
    Returns:
        Tuple of integers, e.g., (2, 0, 1) or (-2, 0, 1)
    """
    miller_str = miller_str.replace(' ', '')
    
    if ',' in miller_str:
        return tuple(int(x) for x in miller_str.split(','))
    else:
        result = []
        i = 0
        while i < len(miller_str):
            if miller_str[i] == '-':
                result.append(-int(miller_str[i+1]))
                i += 2
            else:
                result.append(int(miller_str[i]))
                i += 1
        return tuple(result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and plot slabs for surface analysis.",
        epilog="Example: python slabgen.py POSCAR --hkl 111 --layers 2 --freeze-mode bottom"
    )
    
    # Mutually exclusive group for Miller index specification
    miller_group = parser.add_mutually_exclusive_group(required=True)
    miller_group.add_argument(
        '--hkl',
        type=str,
        help='Specific Miller index (e.g., "201" or "-201" or "2,0,1")'
    )
    miller_group.add_argument(
        '-m', '--max-hkl',
        type=int,
        help='Max Miller index for automatic generation'
    )
    
    parser.add_argument(
        'structure',
        type=str,
        help='Path to the structure file'
    )
    parser.add_argument(
        '-t', '--thicknesses',
        type=float,
        nargs='+',
        default=[12],
        help='Slab thicknesses (Angstroms or unit planes with -u, default: [12])'
    )
    parser.add_argument(
        '--vacuum',
        type=float,
        default=15.0,
        help='Vacuum thickness in Angstroms (always in Angstroms, default: 15)'
    )
    parser.add_argument(
        '--layers',
        type=int,
        default=3,
        help='Number of layers (interpretation depends on freeze-mode, default: 3)'
    )
    parser.add_argument(
        '--freeze-mode',
        type=str,
        choices=['center', 'bottom', 'top'],
        default='center',
        help='Freezing mode: center (relax top/bottom), bottom (freeze bottom), top (freeze top). Default: center'
    )
    parser.add_argument(
        '-u', '--unit-planes',
        action='store_true',
        default=False,
        help='Use unit planes for slab thickness (vacuum always in Angstroms)'
    )
    parser.add_argument(
        '-d', '--directory',
        default='GeneratedSlabs',
        help='Parent directory of all slabs (default: GeneratedSlabs)',
        type=str
    )
    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Do not overwrite existing directories (default: overwrite enabled)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: no log file, console only)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = setup_logging(args.verbose)
    # Load structure
    try:
        structure = Structure.from_file(args.structure)
    except Exception as e:
        print(f"Error: Could not load structure from {args.structure}: {e}")
        return 1

    # Convert freeze mode to enum
    freeze_mode = FreezingMode(args.freeze_mode)
    
    total_generated_slabs = 0
    
    for thickness in args.thicknesses:
        if args.hkl is not None:
            miller_index = parse_miller_index(args.hkl)
            print(f"Generating slabs for Miller index {miller_index}")
            
            slabs_asym = generate_slabs_from_miller(
                structure=structure,
                miller_index=miller_index,
                min_slab_size=thickness,
                min_vacuum_size=args.vacuum,
                in_unit_planes=args.unit_planes,
                symmetrize=False,
                lll_reduce=True,
                center_slab=True,
                primitive=True,
            )
            
            slabs_sym = generate_slabs_from_miller(
                structure=structure,
                miller_index=miller_index,
                min_slab_size=thickness,
                min_vacuum_size=args.vacuum,
                in_unit_planes=args.unit_planes,
                symmetrize=True,
                lll_reduce=True,
                center_slab=True,
                primitive=True,
            )
            
            slabs = slabs_asym + slabs_sym
            
        else:
            slabs_asym = generate_all_slabs(
                structure,
                max_index=args.max_hkl,
                min_slab_size=thickness,
                min_vacuum_size=args.vacuum,
                in_unit_planes=args.unit_planes,
                symmetrize=False,
                lll_reduce=True,
                center_slab=True,
                primitive=True,
            )
        
            slabs_sym = generate_all_slabs(
                structure,
                max_index=args.max_hkl,
                min_slab_size=thickness,
                min_vacuum_size=args.vacuum,
                in_unit_planes=args.unit_planes,
                symmetrize=True,
                lll_reduce=True,
                center_slab=True,
                primitive=True,
            )
        
            slabs = slabs_asym + slabs_sym
        
        total_generated_slabs += len(slabs)
        written_count = write_slab_directories(
            slabs,
            args.directory,
            min_slab_size=thickness,
            layers_to_relax=args.layers if args.layers > 0 else None,
            freeze_mode=freeze_mode,
            overwrite=not args.no_overwrite,  # Invert the flag
        )
        
        print(f"Thickness {thickness}: Generated {len(slabs)} slabs, wrote {written_count}")
            
    print(f"\nTotal: {total_generated_slabs} slabs generated")
    return 0


if __name__ == "__main__":
    exit(main())
