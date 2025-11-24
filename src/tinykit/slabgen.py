import argparse
import warnings
import json
import numpy as np

from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab, generate_all_slabs
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
    
    unique_projections = np.unique(np.round(projections, decimals=2))
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
        projection = np.round(np.dot(site.coords, normal), decimals=2)
        
        if projection <= bottom_threshold or projection >= top_threshold:
            site.properties["selective_dynamics"] = [True, True, True]
        else:
            site.properties["selective_dynamics"] = [False, False, False]
    
    return slab

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


# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and plot slabs for surface analysis.")
    
    # Command-line arguments
    parser.add_argument('structure', type=str,
                        help='Path to the structure file')
    parser.add_argument('--hkl', type=int, default=1,
                        help='Max Miller index for the surface (default: 1)')
    parser.add_argument('-t','--thicknesses', type=float, nargs='+', default=[12],
                        help='Slab thicknesses to generate (default: [12])')
    parser.add_argument('--vacuum', type=float, default=15.0,
                        help='Vacuum thicknesses to add (default: [15])')
    parser.add_argument('--layers_to_relax', type=int, default=3,
                        help='Number of layers to relax (default: 3)')
    parser.add_argument('--symmetrize', action='store_true', default=False, help='Force top and bottom surface to be equivalent (does not preserve stoichiometry)')
    parser.add_argument('-d', "--directory",default='GeneratedSlabs', help='parent directory of all slabs',type=str)
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Generate slabs without tasker
    structure = Structure.from_file(args.structure)
    total_generated_slabs = 0
    for thickness in args.thicknesses:

        slabs_asym = generate_all_slabs(
            structure,
            max_index=args.hkl,
            min_slab_size=thickness,
            min_vacuum_size=args.vacuum,
            symmetrize=False,
            lll_reduce=True,
            center_slab=True,
            primitive=True,
        )
    
        slabs_sym = generate_all_slabs(
            structure,
            max_index=args.hkl,
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
