"""Generate VASP inputs for charged slabs over a range of NELECT values."""

import argparse

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

import numpy as np

from tinykit.vaspio import write_vasp_input
from tinykit.cli import (
    add_incar_args, resolve_incar, add_potcar_args,
    add_kpoints_args, gamma_kpoints, add_overwrite_args,
)


def write_directories(
    structure: Structure,
    nelects: list[float],
    kpoints,
    directory: str,
    base_incar,
    functional: str = "PBE",
    overwrite: bool = True,
) -> int:
    """Write a VASP input directory for each NELECT value. Returns count written."""
    base_incar = dict(base_incar)
    # Use the K_pv POTCAR for potassium.
    site_symbols = [s.replace("K", "K_pv") for s in Poscar(structure).site_symbols]

    written = 0
    for nelect in nelects:
        incar = dict(base_incar)
        incar["NELECT"] = round(nelect, 6)
        path = write_vasp_input(
            structure,
            f"{directory}/NELECT_{nelect:.2f}",
            incar,
            kpoints,
            potcar_symbols=site_symbols,
            potcar_functional=functional,
            overwrite=overwrite,
        )
        if path is not None:
            written += 1
    return written


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(
        description="Generate VASP inputs for a charged slab over a range of NELECT values."
    )

    parser.add_argument('structure', type=str,
                        help='Path to the structure file')
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--start', type=float, default=0.1)
    parser.add_argument('--stop', type=float, default=1.1)
    parser.add_argument('--dipole', action='store_true',
                        help='Add a dipole correction referenced at the center of mass')
    parser.add_argument('-o', '--output', default='ChargedInputs',
                        help='Output directory (default: ChargedInputs)')
    add_incar_args(parser, "charge")
    add_kpoints_args(parser, (5, 5, 1))
    add_potcar_args(parser)
    add_overwrite_args(parser)
    return parser


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = build_parser().parse_args(args)

    structure = Structure.from_file(args.structure)
    kpoints = gamma_kpoints(args)
    nelects = np.arange(args.start, args.stop, args.step)

    base_incar = dict(resolve_incar(args))
    if args.dipole:
        base_incar["IDIPOL"] = 3
        # Place the dipole reference at the center of mass.
        weights = [site.species.weight for site in structure.sites]
        center_of_mass = np.average(structure.frac_coords, weights=weights, axis=0)
        base_incar["DIPOL"] = f"{center_of_mass[0]:.2f} {center_of_mass[1]:.2f} {center_of_mass[2]:.2f}"

    written = write_directories(
        structure, nelects, kpoints, args.output, base_incar,
        functional=args.functional, overwrite=args.overwrite,
    )
    print(f"Wrote {written} charged-slab directories to {args.output}/")


if __name__ == "__main__":
    main()
