import argparse
import warnings
from surfaxe.generation import generate_slabs

# Filter warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="POTCAR data")
warnings.filterwarnings("ignore", message="Overriding the POTCAR functional")

def parse_hkl(value):
    # If a single integer is passed, convert to a tuple of that value repeated three times
    try:
        return (int(value), int(value), int(value))
    except ValueError:
        # If a list of integers is passed, convert to tuple
        hkl_list = [int(i) for i in value.split(',')]
        if len(hkl_list) != 3:
            raise argparse.ArgumentTypeError("hkl must be either a single integer or a list of three integers.")
        return tuple(hkl_list)


# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and plot slabs for surface analysis.")
    
    # Command-line arguments
    parser.add_argument('structure', type=str,
                        help='Path to the structure file')
    parser.add_argument('--hkl', type=parse_hkl, default=1,
                        help='Miller indices for the surface (default: max index 1)')
    parser.add_argument('--thicknesses', type=float, nargs='+', default=[12],
                        help='Slab thicknesses to generate (default: [12])')
    parser.add_argument('--vacuums', type=float, nargs='+', default=[15],
                        help='Vacuum thicknesses to add (default: [15])')
    parser.add_argument('--make_input_files', action='store_true',
                        help='If set, make input files for VASP')
    parser.add_argument('--layers_to_relax', type=int, default=3,
                        help='Number of layers to relax (default: 3)')
    parser.add_argument('--save_slabs', action='store_true',
                        help='If set, save generated slab structures')
    parser.add_argument('--config_dict', type=str, default='pe_relax',
                        help='Configuration dictionary for slab generation (default: pe_relax)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Generate slabs
    slabs_dict = generate_slabs(
        structure=args.structure,
        hkl=args.hkl,
        thicknesses=args.thicknesses,
        vacuums=args.vacuums,
        make_input_files=args.make_input_files,
        layers_to_relax=args.layers_to_relax,
        save_slabs=args.save_slabs,
        config_dict=args.config_dict,
        is_symmetric=True
    )


if __name__ == "__main__":
    main()
