"""Convert a trajectory of structures into batched VASP input directories."""

import argparse
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure

from ase.io import read

from tinykit.vaspio import write_many
from tinykit.cli import add_potcar_args, add_overwrite_args


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
    parser.add_argument('-o', '--output', type=str, default='deployed',
                        help='Output directory for VASP inputs (default: deployed)')
    parser.add_argument('--freeze', type=float, default=None,
                        help='Freeze atoms below the specified z-coordinate')
    add_potcar_args(parser)
    add_overwrite_args(parser)
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

    # One directory per structure, named by formula. Sort sites for clean POSCARs.
    jobs = [
        (f"{sanitize_filename(s.composition.reduced_formula)}_{i+1}", s, incar)
        for i, s in enumerate(structures)
    ]
    written = write_many(jobs, args.output, kpoints, sort_structure=True,
                         potcar_functional=args.functional, overwrite=args.overwrite)

    print(f"Wrote {written} VASP input directories to: {args.output}")

    
if __name__ == "__main__":
    main()



