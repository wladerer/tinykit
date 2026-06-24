"""Convert a trajectory of structures into batched VASP input directories."""

import argparse
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pathlib import Path

from ase.io import read

from tinykit.vaspio import write_vasp_input


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a directory name."""
    return "".join(c if c.isalnum() else "" for c in name)


def freeze_atoms(structure: Structure, z_limit: float) -> Structure:
    """Freeze atoms in the structure below a specific z-coordinate."""
    for site in structure.sites:
        if site.coords[2] < z_limit:
            site.properties['selective_dynamics'] = [False, False, False]
        else:
            site.properties['selective_dynamics'] = [True, True, True]
    return structure


# Define argument parser
def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(
        description="Generate VASP input files from a trajectory of structures."
    )

    # Command-line arguments
    parser.add_argument('structures', type=str,
                        help='Path to the structures file (traj, extxyz, XDATCAR, etc.)')
    parser.add_argument('-k', '--kpoints', type=str, default='KPOINTS',
                        help='Path to the kpoints file (default: KPOINTS)')
    parser.add_argument('-i', '--incar', type=str, default='INCAR',
                        help='Path to the incar file (default: INCAR)')
    parser.add_argument('-o', '--output', type=str, default='./kamino',
                        help='Output directory for VASP inputs (default: ./kamino)')
    parser.add_argument('--freeze', type=float, default=None,
                        help='Freeze atoms below the specified z-coordinate')
    parser.add_argument('--functional', default='PBE',
                        help='POTCAR functional family (default: PBE)')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false',
                        help='Skip directories that already exist instead of overwriting')
    parser.set_defaults(overwrite=True)
    return parser


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = build_parser().parse_args(args)

    # Load structures from file
    try: 
        ase_structures = read(args.structures, index=':')
        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in ase_structures]
    except Exception as e:
        print(f"Error reading structures: {e}")
        return

    # Load incar and kpoints files
    try:
        incar = Incar.from_file(args.incar)
    except Exception as e:
        print(f"Error reading INCAR file: {e}")
        return

    try:
        kpoints = Kpoints.from_file(args.kpoints)
    except Exception as e:
        print(f"Error reading KPOINTS file: {e}")
        return


    if args.freeze is not None:
        # Freeze atoms in the structures
        for i, structure in enumerate(structures):
            structures[i] = freeze_atoms(structure, args.freeze)

    # Write VASP input files to output directory, one per structure.
    written = 0
    for i, structure in enumerate(structures):
        # Get a more descriptive name based on the chemical formula
        formula = sanitize_filename(structure.composition.reduced_formula)
        output_dir = Path(args.output) / f"{formula}_{i+1}"
        path = write_vasp_input(structure, output_dir, incar, kpoints, sort_structure=True,
                                potcar_functional=args.functional, overwrite=args.overwrite)
        if path is not None:
            written += 1

    print(f"Wrote {written} VASP input directories to: {args.output}")

    
if __name__ == "__main__":
    main()



