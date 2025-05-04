import argparse
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pathlib import Path

from ase.io import read 


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a directory name."""
    return "".join(c if c.isalnum() else "" for c in name)


def assemble_vasp_inputs(structures: list[Structure], incar: Incar, kpoints: Kpoints) -> list[VaspInput]:
    """Assemble VASP input files from structures, incar, kpoints, and potcar."""
    inputs = []
    for structure in structures:
        poscar = Poscar(structure, sort_structure=True)
        potcar = Potcar(symbols=poscar.site_symbols, functional="PBE")
        input_set = VaspInput(incar=incar, kpoints=kpoints, poscar=poscar, potcar=potcar)
        inputs.append(input_set)

    return inputs

def freeze_atoms(structure: Structure, fraction: float = 0.75) -> Structure:
    """Freeze a fraction of atoms in the structure."""
    sites = structure.sites
    sites_sorted = sorted(sites, key=lambda x: x.coords[2])
    num_atoms_to_freeze = int(len(sites) * fraction)
    for site in sites_sorted[:num_atoms_to_freeze]:
        site.properties['selective_dynamics'] = [False, False, False]
    for site in sites_sorted[num_atoms_to_freeze:]:
        site.properties['selective_dynamics'] = [True, True, True]
    return structure


# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate VASP input files from a list of structures.")
    
    # Command-line arguments
    parser.add_argument('structures', type=str,
                        help='Path to the structures file (extxyz, XDATCAR, etc.)')
    parser.add_argument('-k', '--kpoints', type=str, default='KPOINTS',
                        help='Path to the kpoints file (default: KPOINTS)')
    parser.add_argument('-i', '--incar', type=str, default='INCAR',
                        help='Path to the incar file (default: INCAR)')
    parser.add_argument('-o', '--output', type=str, default='./kamino',
                        help='Output directory for VASP inputs (default: ./kamino)')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze bottom 75% of atoms in the structure')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

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

    # Generate VASP inputs
    inputs = assemble_vasp_inputs(structures, incar, kpoints)

    # Write VASP input files to output directory
    for i, (structure, input_set) in enumerate(zip(structures, inputs)):
        # Get a more descriptive name based on the chemical formula
        formula = sanitize_filename(structure.composition.reduced_formula)
        output_dir = Path(args.output) / f"{formula}_{i+1}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_set.write_input(output_dir)

    print(f"VASP inputs written to: {args.output}")

    
if __name__ == "__main__":
    main()



