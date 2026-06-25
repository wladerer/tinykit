"""Extract per-atom magnetic moments from VASP output for `viz --moments`.

A support module, not a subcommand: it reads the physical moments from the
OUTCAR (collinear scalars or non-collinear vectors) so `viz` can draw them.
"""

from pathlib import Path

import numpy as np
from pymatgen.io.vasp import Outcar, Vasprun


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
    """Extract non-collinear (vector) magmoms by summing projected magnetisation.

    NOTE: this sums over all bands/k-points of the projected magnetisation, so
    the magnitudes are not the physical local moments (only useful for
    orientation). Prefer :func:`read_outcar_moment_vectors` when an OUTCAR is
    available; this remains as a vasprun-only fallback.
    """
    # projected_magnetisation shape: (nkpoints, nbands, natoms, norbitals, 3)
    proj_mag = vasprun.projected_magnetisation
    return np.sum(proj_mag, axis=(0, 1, 3))  # -> (natoms, 3)


def _read_outcar_component(lines, component):
    """Per-ion 'tot' moments from the LAST 'magnetization (<component>)' block.

    Returns a list of floats (one per ion), or None if the block is absent
    (e.g. a collinear OUTCAR only has the 'x' block).
    """
    header = f"magnetization ({component})"
    starts = [i for i, line in enumerate(lines) if line.strip().startswith(header)]
    if not starts:
        return None
    totals, started = [], False
    for line in lines[starts[-1] + 1:]:
        parts = line.split()
        if parts and parts[0].isdigit() and len(parts) >= 2:
            totals.append(float(parts[-1]))  # 'tot' is the last column
            started = True
        elif started:
            break  # trailing '----' / 'tot' summary ends the per-ion table
    return totals


def read_outcar_moment_vectors(outcar_path):
    """Return (natoms, 3) physical moment vectors from a non-collinear OUTCAR.

    Reads the per-ion total from the final 'magnetization (x/y/z)' tables.
    Returns None if the OUTCAR is missing or has no y/z blocks (i.e. it is a
    collinear run rather than a non-collinear one).
    """
    outcar_path = Path(outcar_path)
    if not outcar_path.exists():
        return None
    lines = outcar_path.read_text().splitlines()
    components = [_read_outcar_component(lines, c) for c in ("x", "y", "z")]
    if any(c is None for c in components) or not all(components):
        return None
    return np.array(components, dtype=float).T  # columns x,y,z -> (natoms, 3)


def get_moment_vectors(path, collinear=False):
    """Return an (natoms, 3) array of Cartesian magnetic-moment vectors.

    Both modes read the physical per-atom moments from the OUTCAR (`path`, or
    the OUTCAR beside it):

    - collinear: the scalar total placed along +z by sign (up/down arrows);
    - non-collinear: the (x, y, z) totals from the OUTCAR magnetization tables.

    If a non-collinear OUTCAR is unavailable, fall back to summing the vasprun's
    projected magnetisation (orientation only; magnitudes are not physical).
    This is the form consumed by `viz --moments`.
    """
    outcar = _sibling_outcar(path)
    if collinear:
        magmoms = get_collinear_magmoms(Outcar(str(outcar)))
        return np.array([[0.0, 0.0, float(m)] for m in magmoms])

    vectors = read_outcar_moment_vectors(outcar)
    if vectors is not None:
        return vectors
    # An OUTCAR with no x/y/z vector tables is a collinear run; say so plainly
    # rather than blindly trying to parse it as a vasprun.xml.
    if outcar.exists():
        raise ValueError(
            f"{outcar} has no non-collinear magnetization (x/y/z) tables; it "
            f"looks like a collinear run. Re-render with --collinear.")
    vasprun = Vasprun(path, parse_projected_eigen=True)
    return np.asarray(get_noncollinear_magmoms(vasprun), dtype=float)
