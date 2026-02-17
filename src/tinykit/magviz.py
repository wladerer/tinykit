#!/usr/bin/env python3
"""Extract magnetic moments from vasprun.xml and write to CIF."""

import argparse
import numpy as np
from pymatgen.io.vasp import Vasprun
from pymatgen.io.cif import CifWriter


def get_collinear_magmoms(vasprun):
    """Extract collinear (scalar) magmoms from the final ionic step."""
    mag = vasprun.magnetization  # tuple of dicts, one per site
    # Each dict has orbital contributions; 'tot' is the total moment
    return [site_mag["tot"] for site_mag in mag[-1]]  # last ionic step


def get_noncollinear_magmoms(vasprun):
    """Extract non-collinear (vector) magmoms by summing projected magnetisation."""
    # projected_magnetisation shape: (nkpoints, nbands, natoms, norbitals, 3)
    proj_mag = vasprun.projected_magnetisation
    return np.sum(proj_mag, axis=(0, 1, 3))  # -> (natoms, 3)


def main():
    parser = argparse.ArgumentParser(
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
    args = parser.parse_args()

    parse_projected = not args.collinear
    vasprun = Vasprun(args.vasprun, parse_projected_eigen=parse_projected)
    structure = vasprun.final_structure.copy()

    if args.collinear:
        magmoms = get_collinear_magmoms(vasprun)
        for site, mag in zip(structure, magmoms):
            site.properties["magmom"] = mag
    else:
        magmoms = get_noncollinear_magmoms(vasprun)
        for site, mag in zip(structure, magmoms):
            site.properties["magmom"] = mag.tolist()

    CifWriter(structure, write_magmoms=True).write_file(args.output)
    print(f"CIF with {'collinear' if args.collinear else 'non-collinear'} magnetic moments written to {args.output}")


if __name__ == "__main__":
    main()
