import argparse
import warnings
from surfaxe.generation import generate_slabs

from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from pathlib import Path

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
    "KPAR": 8,
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

# Filter warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="POTCAR data")
warnings.filterwarnings("ignore", message="Overriding the POTCAR functional")

def generate_miller_planes(hkl_max: int) -> list[tuple[int]]:
    """generates a list of miller planes for slab generation"""
    
    miller_planes = []
    for h in range(hkl_max+1):
        for k in range(hkl_max+1):
            for l in range(hkl_max+1):
                if h == 0 and k == 0 and l == 0:
                    continue
                miller_planes.append((h, k, l))
    
    return miller_planes

def write_directories(structures: list[Structure], directory: str) -> None:
    """writes each structure to its own directory"""

    for index, structure in enumerate(structures):
        path = Path(directory) / f"slab_{index}"
        path.mkdir(parents=True, exist_ok=True)

        #create poscar from structure
        poscar = Poscar(structure)
        potcar = Potcar(symbols=poscar.site_symbols, functional="PBE")
        incar = Incar.from_dict(incar_dict)
        
        input_set = VaspInput(incar=incar, kpoints=kpoints, poscar=poscar, potcar=potcar)
        input_set.write_input(path)

    return None

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and plot slabs for surface analysis.")
    
    # Command-line arguments
    parser.add_argument('structure', type=str,
                        help='Path to the structure file')
    parser.add_argument('--hkl', type=int, default=1,
                        help='Miller indices for the surface (default: max index 1)')
    parser.add_argument('-t','--thicknesses', type=float, nargs='+', default=[12],
                        help='Slab thicknesses to generate (default: [12])')
    parser.add_argument('--vacuums', type=float, nargs='+', default=[15],
                        help='Vacuum thicknesses to add (default: [15])')
    parser.add_argument('--layers_to_relax', type=int, default=3,
                        help='Number of layers to relax (default: 3)')
    parser.add_argument('--config_dict', type=str, default='pe_relax',
                        help='Configuration dictionary for slab generation (default: pe_relax)')
    parser.add_argument('-a', '--allow-asymmetric', action='store_true')
    parser.add_argument('--no-tasker', action='store_true',
                        help='Uses pymatgen to generate slabs without checking for suitability (dipole, symmetry, etc)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    if args.no_tasker:
        # Generate slabs without tasker
        structure = Structure.from_file(args.structure)
        slabs_dict = {}
        for hkl in generate_miller_planes(args.hkl):
            slabgen = SlabGenerator(
                structure,
                hkl,
                min_slab_size=min(args.thicknesses),
                min_vacuum_size=15,
                center_slab=True,
                reorient_lattice=True,
                lll_reduce=True,
            )
            slabs = slabgen.get_slabs()
            #hkl to string
            hkl_str = f"{hkl[0]}{hkl[1]}{hkl[2]}"
            slabs_dict[hkl_str] = slabs

            write_directories(slabs, f"GeneratedSlabs/{hkl_str}") 

    else:
        # Generate slabs

        slabs_dict = generate_slabs(
            structure=args.structure,
            hkl=args.hkl,
            thicknesses=args.thicknesses,
            vacuums=args.vacuums,
            make_input_files=True,
            layers_to_relax=args.layers_to_relax,
            save_slabs=True,
            config_dict=args.config_dict,
            is_symmetric=False if args.allow_asymmetric else True
        )


if __name__ == "__main__":
    main()
