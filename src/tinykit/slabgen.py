import argparse
import warnings
from surfaxe.generation import generate_slabs

# Filter warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="POTCAR data")
warnings.filterwarnings("ignore", message="Overriding the POTCAR functional")


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
        from pymatgen.core import Structure
        from pymatgen.analysis.adsorption import SlabGenerator
        structure = Structure.from_file(args.structure)
        slabgen = SlabGenerator(structure, hkl=args.hkl, min_slab_size=min(args.thicknesses), min_vacuum_size=15)

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
