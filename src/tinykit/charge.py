import argparse

from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints, Potcar
from pymatgen.io.vasp.sets import VaspInput

import numpy as np

base_incar_dict = {
    "ALGO": "Normal",
    "EDIFF": 1e-06,
    "EDIFFG": -0.01,
    "ENCUT": 500,
    "GGA": "Pe",
    "IBRION": 2,
    "ISMEAR": 0,
    "ISYM": 2,
    "IVDW": 4,
    "IWAVPR": 1,
    "KPAR": 16,
    "LASPH": True,
    "LCHARG": True,
    "LMAXMIX": 6,
    "LORBIT": 11,
    "LREAL": "Auto",
    "LVHAR": True,
    "LWAVE": False,
    "NELM": 100,
    "NEDOS": 3001,
    "PREC": "Accurate",
    "SIGMA": 0.02,
    "NCORE": 64,
}

def write_directories(structure: Structure, nelects: list[float], kpoints: Kpoints, directory: str) -> None:
    """writes each structure to its own directory"""

    for nelect in nelects:

        path = Path(directory) / f"NELECT_{nelect:.2f}"
        path.mkdir(parents=True, exist_ok=True)

        #create poscar from structure
        poscar = Poscar(structure)
        site_symbols = poscar.site_symbols
        #update K to K_pv
        site_symbols = [site_symbol.replace("K", "K_pv") for site_symbol in site_symbols]
        potcar = Potcar(symbols=poscar.site_symbols, functional="PBE")
        #update incar_dict with NELECT
        incar_dict = base_incar_dict.copy()
        incar_dict["NELECT"] = nelect
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
    parser.add_argument('--step', type=float, default=0.1,)
    parser.add_argument('--start', type=float, default=0.1,)
    parser.add_argument('--stop', type=float, default=1.1)
    parser.add_argument('--kpoints', type=int, nargs=3, default=[4, 4, 1])
    parser.add_argument('--dipole', action='store_true')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Generate charged slabs
    structure = Structure.from_file(args.structure)
    kpoints = Kpoints(kpts=args.kpoints)
    nelects = np.arange(args.start, args.stop, args.step)

    if args.dipole:
        base_incar_dict["IDIPOL"] = 3
        base_incar_dict["LDIPOL"] = True

        #get center of mass of the structure    
        weights = [site.species.weight for site in structure.sites]
        center_of_mass = np.average(structure.frac_coords, weights=weights, axis=0)
        base_incar_dict["DIPOL"] = f"{center_of_mass[0]:.2f} {center_of_mass[1]:.2f} {center_of_mass[2]:.2f}"

    write_directories(structure, nelects, kpoints, "ChargedInputs")

if __name__ == "__main__":
    main()
