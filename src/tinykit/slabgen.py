import argparse
import warnings
import json
import numpy as np
from surfaxe.generation import generate_slabs

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




def write_slab_directories(
    slabs: list[Slab], 
    directory: str, 
    min_slab_size: float,
) -> None:

    seen = set()
    root = Path(directory)

    for termination_index, slab in enumerate(slabs):

        slab = slab.get_sorted_structure(key=lambda s: s.species_string)

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
#        slabs = remove_duplicates(slabs)  # your existing hash-based system
    
        total_generated_slabs += len(slabs)
        write_slab_directories(slabs, args.directory, min_slab_size=thickness)
            

    print(f"Generated {total_generated_slabs} slabs")
if __name__ == "__main__":
    main()
