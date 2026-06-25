"""Charge-density loading and isosurface assembly for `viz`.

Pulls the CHGCAR/PARCHG handling out of `viz.main`: load the density, tile and
interpolate it to a supercell, resolve an isovalue, and build the
`isosurface_data` list ASE's POV-Ray writer consumes.
"""

import numpy as np
from ase.io import read
from ase.calculators.vasp.vasp_auxiliary import VaspChargeDensity

from tinykit.povray import parse_rgb
from tinykit.cli import get_logger

logger = get_logger(__name__)

# Named dual-phase palettes, (positive, negative).
COLOR_SCHEMES = {
    "blue-orange": ((0.6, 0.75, 0.95), (0.95, 0.75, 0.6)),
    "purple-yellow": ((0.8, 0.7, 0.95), (0.95, 0.95, 0.7)),
    "teal-coral": ((0.65, 0.9, 0.9), (0.95, 0.75, 0.75)),
    "green-magenta": ((0.7, 0.9, 0.75), (0.95, 0.7, 0.9)),
}


def is_chgcar_input(args) -> bool:
    """Whether this invocation should be treated as a charge-density render."""
    return bool(args.chgcar) or args.input_is_chgcar or "CHG" in args.input.upper()


def load_density(input_path, chgcar_path=None):
    """Return (atoms, density).

    With a separate `chgcar_path`, the structure is read from `input_path` and
    the density from `chgcar_path`. Otherwise `input_path` is itself a
    CHGCAR/PARCHG providing both.
    """
    if chgcar_path:
        atoms = read(input_path, index=-1)
        density = VaspChargeDensity(chgcar_path).chg[-1]
    else:
        vcd = VaspChargeDensity(input_path)
        atoms, density = vcd.atoms[-1], vcd.chg[-1]
    return atoms, density


def prepare_density(density, supercell, mesh_refinement):
    """Tile the density to the supercell and optionally interpolate it finer."""
    if supercell != [1, 1, 1]:
        logger.info(f"Tiling density grid {supercell[0]}x{supercell[1]}x{supercell[2]}")
        density = np.tile(density, supercell)
    if mesh_refinement > 1:
        from scipy.ndimage import zoom
        logger.warning(
            f"--mesh-refinement {mesh_refinement} interpolates the grid "
            f"{mesh_refinement ** 3}x, which multiplies render time")
        density = zoom(density, mesh_refinement, order=3)
    return density


def resolve_isovalue(density, isovalue):
    """Return a positive isovalue: the given one, or a simple data-driven default.

    The default is a single rule: density centred near zero (wavefunctions,
    difference densities) keys off the spread; an all-positive density keys off
    the mean plus spread. No iterative re-adjustment; pass `--isovalue` to
    control it exactly.
    """
    abs_max = float(np.abs(density).max())
    if isovalue is not None:
        value = abs(isovalue)
        if value >= abs_max:
            logger.warning(
                f"--isovalue {value:.3e} exceeds the data max {abs_max:.3e}; "
                f"the surface may be empty")
        return value

    if density.min() < 0:
        value = min(2 * density.std(), 0.3 * abs_max)
    else:
        value = min(density.mean() + 2 * density.std(), 0.5 * density.max())
    value = abs(value)
    logger.info(f"No --isovalue given, using {value:.3e} (pass --isovalue to control)")
    return value


def _isosurface(density, isovalue, color, transmittance):
    if transmittance > 0:
        color = tuple(color) + (transmittance,)
    return {"density_grid": density, "cut_off": isovalue,
            "color": color, "material": "ase3"}


def build_isosurface_data(density, args):
    """Build the `isosurface_data` list for ASE's POV writer from parsed args."""
    isovalue = resolve_isovalue(density, args.isovalue)
    has_negative = density.min() < 0

    if args.color_scheme != "custom":
        pos_color, neg_color = COLOR_SCHEMES[args.color_scheme]
    else:
        pos_color = parse_rgb(args.iso_color)
        neg_color = parse_rgb(args.iso_color_negative)

    surfaces = [_isosurface(density, isovalue, pos_color, args.iso_transmittance)]
    if args.dual_phase or (has_negative and isovalue <= abs(density.min())):
        surfaces.append(_isosurface(density, -isovalue, neg_color, args.iso_transmittance))
        logger.info(f"Isosurface +/-{isovalue:.3e}, transmittance {args.iso_transmittance}")
    else:
        logger.info(f"Isosurface {isovalue:.3e}, transmittance {args.iso_transmittance}")
        if has_negative:
            logger.info("Data has negative values; pass --dual-phase to render both phases")
    return surfaces
