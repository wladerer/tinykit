import argparse
import warnings
import json
import numpy as np

from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab, SlabGenerator, generate_all_slabs
from pathlib import Path

kpoints = Kpoints.gamma_automatic((1,1,1), shift=(0,0,0))

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


def apply_selective_dynamics(slab: Slab, layers_to_relax: int) -> Slab:
    """
    Apply selective dynamics to a slab structure, relaxing only the top and 
    bottom layers while fixing the center.
    
    Args:
        slab: Pymatgen Slab object
        layers_to_relax: Number of layers to relax at top and bottom
        
    Returns:
        Slab with selective dynamics applied
    """

    normal = slab.normal
    positions = np.array([site.coords for site in slab.sites])
    projections = np.dot(positions, normal)
    
    unique_projections = np.unique(np.round(projections, decimals=0))
    unique_projections.sort()
    
    nlayers = len(unique_projections)
    
    if nlayers <= 2 * layers_to_relax:
        warnings.warn(
            f"Slab has {nlayers} layers but {2*layers_to_relax} layers "
            f"requested for relaxation. Skipping selective dynamics."
        )
        return slab
    
    bottom_threshold = unique_projections[layers_to_relax - 1]
    top_threshold = unique_projections[nlayers - layers_to_relax]
    
    for site in slab.sites:
        projection = np.round(np.dot(site.coords, normal), decimals=0)
        
        if projection <= bottom_threshold or projection >= top_threshold:
            site.properties["selective_dynamics"] = [True, True, True]
        else:
            site.properties["selective_dynamics"] = [False, False, False]
    
    return slab


def generate_slabs_from_miller(
    structure: Structure,
    miller_index: tuple,
    min_slab_size: float,
    min_vacuum_size: float,
    symmetrize: bool = False,
    lll_reduce: bool = True,
    center_slab: bool = True,
    primitive: bool = True,
) -> list[Slab]:
    """
    Generate slabs for a specific Miller index.
    
    Args:
        structure: Input structure
        miller_index: Specific Miller index tuple, e.g., (2, 0, 1) or (-2, 0, 1)
        min_slab_size: Minimum slab thickness in Angstroms
        min_vacuum_size: Minimum vacuum thickness in Angstroms
        symmetrize: Whether to make top and bottom surfaces equivalent
        lll_reduce: Whether to perform LLL reduction
        center_slab: Whether to center the slab
        primitive: Whether to reduce to primitive cell
        
    Returns:
        List of Slab objects
    """
    slabgen = SlabGenerator(
        initial_structure=structure,
        miller_index=miller_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        lll_reduce=lll_reduce,
        center_slab=center_slab,
        primitive=primitive,
    )
    
    slabs = slabgen.get_slabs(symmetrize=symmetrize)
    
    return slabs


def write_slab_directories(
    slabs: list[Slab], 
    directory: str, 
    min_slab_size: float,
    layers_to_relax: int = None,
) -> None:

    seen = set()
    root = Path(directory)

    for termination_index, slab in enumerate(slabs):

        slab = slab.get_sorted_structure(key=lambda s: s.species_string)
        
        # Apply selective dynamics if requested
        if layers_to_relax is not None:
            slab = apply_selective_dynamics(slab, layers_to_relax)

        sym_dir = "sym" if slab.is_symmetric() else "asym"

        hkl = "".join(str(i) for i in slab.miller_index)
        size_str = f"{min_slab_size:.2f}"

        path = (
            Path(directory)
            / hkl
            / size_str
            / sym_dir
            / f"term_{termination_index}"
        )
        path.mkdir(parents=True, exist_ok=True)

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


def parse_miller_index(miller_str: str) -> tuple:
    """
    Parse a Miller index string into a tuple.
    
    Args:
        miller_str: String like '201' or '-201' or '2,0,1' or '-2,0,1'
        
    Returns:
        Tuple of integers, e.g., (2, 0, 1) or (-2, 0, 1)
    """
    # Remove spaces and handle comma-separated values
    miller_str = miller_str.replace(' ', '')
    
    if ',' in miller_str:
        # Handle comma-separated: '2,0,1' or '-2,0,1'
        return tuple(int(x) for x in miller_str.split(','))
    else:
        # Handle concatenated: '201' or '-201'
        result = []
        i = 0
        while i < len(miller_str):
            if miller_str[i] == '-':
                # Negative number
                result.append(-int(miller_str[i+1]))
                i += 2
            else:
                # Positive number
                result.append(int(miller_str[i]))
                i += 1
        return tuple(result)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and plot slabs for surface analysis.")
    
    # Command-line arguments
    parser.add_argument('structure', type=str,
                        help='Path to the structure file')
    parser.add_argument('--hkl', type=str, default=None,
                        help='Specific Miller index (e.g., "201" or "-201" or "2,0,1")')
    parser.add_argument('-m', '--max-hkl', type=int, default=None,
                        help='Max Miller index for automatic generation (default: None)')
    parser.add_argument('-t','--thicknesses', type=float, nargs='+', default=[12],
                        help='Slab thicknesses to generate (default: [12])')
    parser.add_argument('--vacuum', type=float, default=15.0,
                        help='Vacuum thicknesses to add (default: 15)')
    parser.add_argument('--layers_to_relax', type=int, default=3,
                        help='Number of layers to relax (default: 3)')
    parser.add_argument('--symmetrize', action='store_true', default=False, 
                        help='Force top and bottom surface to be equivalent (does not preserve stoichiometry)')
    parser.add_argument('-d', "--directory",default='GeneratedSlabs', 
                        help='parent directory of all slabs',type=str)
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Validate that either hkl or max_hkl is provided
    if args.hkl is None and args.max_hkl is None:
        print("Error: Must specify either --max-hkl (-m) for automatic generation or --hkl for specific plane")
        return
    
    if args.hkl is not None and args.max_hkl is not None:
        print("Error: Cannot specify both --hkl and --max-hkl. Choose one.")
        return

    structure = Structure.from_file(args.structure)
    total_generated_slabs = 0
    
    for thickness in args.thicknesses:
        if args.hkl is not None:
            # Generate slabs for specific Miller index
            miller_index = parse_miller_index(args.hkl)
            print(f"Generating slabs for Miller index {miller_index}")
            
            # Generate both symmetric and asymmetric slabs
            slabs_asym = generate_slabs_from_miller(
                structure=structure,
                miller_index=miller_index,
                min_slab_size=thickness,
                min_vacuum_size=args.vacuum,
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
                symmetrize=True,
                lll_reduce=True,
                center_slab=True,
                primitive=True,
            )
            
            slabs = slabs_asym + slabs_sym
            
        else:
            # Original behavior: generate all slabs up to max_index
            slabs_asym = generate_all_slabs(
                structure,
                max_index=args.max_hkl,
                min_slab_size=thickness,
                min_vacuum_size=args.vacuum,
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
                symmetrize=True,
                lll_reduce=True,
                center_slab=True,
                primitive=True,
            )
        
            slabs = slabs_asym + slabs_sym
        
        total_generated_slabs += len(slabs)
        write_slab_directories(
            slabs, 
            args.directory, 
            min_slab_size=thickness,
            layers_to_relax=args.layers_to_relax if args.layers_to_relax > 0 else None
        )
            
    print(f"Generated {total_generated_slabs} slabs")


if __name__ == "__main__":
    main()
