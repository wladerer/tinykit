#!/usr/bin/env python3
"""Extract magnetic moments from vasprun.xml and write to CIF."""

import argparse
from pathlib import Path

import numpy as np
from pymatgen.io.vasp import Outcar, Vasprun
from pymatgen.io.cif import CifWriter


def get_collinear_magmoms(outcar):
    """Extract collinear (scalar) per-atom magmoms from an Outcar.

    Collinear moments live in the OUTCAR (not the vasprun), one total per atom.
    """
    return [site_mag["tot"] for site_mag in outcar.magnetization]


def _sibling_outcar(path):
    """Resolve the OUTCAR for a given path (the path itself, or its sibling)."""
    p = Path(path)
    return p if p.name == "OUTCAR" else p.parent / "OUTCAR"


def get_noncollinear_magmoms(vasprun):
    """Extract non-collinear (vector) magmoms by summing projected magnetisation."""
    # projected_magnetisation shape: (nkpoints, nbands, natoms, norbitals, 3)
    proj_mag = vasprun.projected_magnetisation
    return np.sum(proj_mag, axis=(0, 1, 3))  # -> (natoms, 3)


def get_moment_vectors(path, collinear=False):
    """Return an (natoms, 3) array of Cartesian magnetic-moment vectors.

    Collinear (scalar) moments are read from the OUTCAR (`path`, or the OUTCAR
    beside it) and placed along the +z axis by sign, so they render as up/down
    arrows. Non-collinear moments are read from the vasprun's projected
    magnetisation and returned as-is. This is the form consumed by
    `viz --moments`.
    """
    if collinear:
        magmoms = get_collinear_magmoms(Outcar(str(_sibling_outcar(path))))
        return np.array([[0.0, 0.0, float(m)] for m in magmoms])
    vasprun = Vasprun(path, parse_projected_eigen=True)
    return np.asarray(get_noncollinear_magmoms(vasprun), dtype=float)


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(
        description="Extract magnetic moments from vasprun.xml and write to CIF."
    )
    parser.add_argument(
        "vasprun",
        nargs="?",
        default="vasprun.xml",
        help="Path to vasprun.xml (default: vasprun.xml)",
    )
    parser.add_argument(
        "-o", "--output",
        default="magmoms.cif",
        help="Output CIF filename (default: magmoms.cif)",
    )
    parser.add_argument(
        "--collinear",
        action="store_true",
        help="Use collinear (scalar) magmoms instead of non-collinear (vector)",
    )
    return parser


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = build_parser().parse_args(args)

    if args.collinear:
        structure = Vasprun(args.vasprun).final_structure.copy()
        magmoms = get_collinear_magmoms(Outcar(str(_sibling_outcar(args.vasprun))))
        for site, mag in zip(structure, magmoms):
            site.properties["magmom"] = mag
    else:
        vasprun = Vasprun(args.vasprun, parse_projected_eigen=True)
        structure = vasprun.final_structure.copy()
        magmoms = get_noncollinear_magmoms(vasprun)
        for site, mag in zip(structure, magmoms):
            site.properties["magmom"] = mag.tolist()

    CifWriter(structure, write_magmoms=True).write_file(args.output)
    print(f"CIF with {'collinear' if args.collinear else 'non-collinear'} magnetic moments written to {args.output}")


if __name__ == "__main__":
    main()
