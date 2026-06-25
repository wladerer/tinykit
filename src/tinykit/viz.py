#!/usr/bin/env python
"""Render structures (slabs or bulk) and charge-density isosurfaces with POV-Ray."""
import argparse
import os

import numpy as np
import yaml
from ase.io import read

from tinykit.povray import (
    resolve_atom_styles, render_structure, render_structure_with_bonds,
    render_structure_with_moments, add_render_args, povray_settings_from_args,
    parse_rgb,
)
from tinykit.chgcar import (
    is_chgcar_input, load_density, prepare_density, build_isosurface_data,
)
from tinykit.magviz import get_moment_vectors
from tinykit.cli import get_logger

logger = get_logger(__name__)


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(
        description='Render structures (slabs or bulk) and charge-density isosurfaces.')
    parser.add_argument('input', help='Input VASP file (POSCAR, CONTCAR, vasprun.xml, or CHGCAR)')
    parser.add_argument('-c', '--colors', '--styles', dest='styles', default=None,
                        help='YAML file overriding per-element color and/or radius, '
                             'e.g. "Fe: [255,100,0]" or "C: {radius: 0.75}"')
    parser.add_argument('-o', '--output', help='Output file name', default='structure.png')
    parser.add_argument('--rotation', help='Rotation of the structure', default=[0, 0, 0], nargs=3, type=float)
    parser.add_argument('--supercell', help='Supercell dimensions', default=[1, 1, 1], nargs=3, type=int)
    parser.add_argument('-v', '--verbose', help='Enable verbose output', action='store_true')

    # Dashed-bond annotations between explicit atom-index pairs
    bonds = parser.add_argument_group('dashed bonds')
    bonds.add_argument('--bond', metavar=('I', 'J'), nargs=2, type=int, action='append',
                       dest='bonds', default=None,
                       help='Draw a dashed line between zero-based atom indices I and J '
                            '(repeatable). Indices refer to the structure after --supercell.')
    bonds.add_argument('--bond-color', default='0.3,0.3,0.3',
                       help='Dashed-line color as a hex string or comma-separated RGB '
                            '0-255 (default: grey)')
    bonds.add_argument('--bond-radius', type=float, default=0.10,
                       help='Dashed-line thickness on the atom-radius scale (default: 0.10)')
    bonds.add_argument('--dash-length', type=float, default=0.30,
                       help='Length of each dash (default: 0.30)')
    bonds.add_argument('--gap-length', type=float, default=0.22,
                       help='Gap between dashes (default: 0.22)')

    # Magnetic-moment arrows
    moments = parser.add_argument_group('magnetic moments')
    moments.add_argument('--moments', nargs='?', const='vasprun.xml', default=None,
                         metavar='VASPRUN',
                         help='Draw magnetic-moment arrows from a vasprun.xml '
                              '(default file: vasprun.xml). Atom order/count must '
                              'match the input structure.')
    moments.add_argument('--collinear', action='store_true',
                         help='Treat moments as collinear scalars along z '
                              '(default: non-collinear vectors)')
    moments.add_argument('--moment-length', type=float, default=2.8,
                         help='Arrow length (default: 2.8)')
    moments.add_argument('--moment-threshold', type=float, default=0.1,
                         help='Skip arrows for |moment| below this (default: 0.1), '
                              'hiding near-zero residual moments')
    moments.add_argument('--moment-by-magnitude', action='store_true',
                         help='Scale arrow length by |moment| (relative to the '
                              'largest), instead of a fixed length')

    # Charge-density isosurfaces (see tinykit.chgcar)
    iso = parser.add_argument_group('charge density')
    iso.add_argument('--chgcar', default=None,
                     help='Separate CHGCAR file (structure still comes from the input)')
    iso.add_argument('--input-is-chgcar', action='store_true',
                     help='Force treating the input file as CHGCAR format')
    iso.add_argument('--isovalue', type=float, default=None,
                     help='Isosurface value (positive; both +/- rendered with --dual-phase). '
                          'Omit to auto-pick from the data.')
    iso.add_argument('--iso-color', default='0.6,0.75,0.95',
                     help='Positive-phase color, comma-separated RGB 0-1 (default: pale blue)')
    iso.add_argument('--iso-color-negative', default='0.95,0.75,0.6',
                     help='Negative-phase color, comma-separated RGB 0-1 (default: pale orange)')
    iso.add_argument('--iso-transmittance', type=float, default=0.5,
                     help='Isosurface transparency, 0=opaque to 1=transparent (default: 0.5)')
    iso.add_argument('--dual-phase', action='store_true',
                     help='Render both positive and negative isosurfaces')
    iso.add_argument('--color-scheme',
                     choices=['blue-orange', 'purple-yellow', 'teal-coral', 'green-magenta', 'custom'],
                     default='blue-orange',
                     help='Dual-phase palette ("custom" uses --iso-color/--iso-color-negative)')
    iso.add_argument('--mesh-refinement', type=int, default=1, choices=[1, 2, 3, 4],
                     help='Grid interpolation factor for a smoother mesh '
                          '(1=none, 2=8x points, 3=27x; multiplies render time)')

    add_render_args(parser, default_height=1100)
    return parser


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = build_parser().parse_args(args)

    if args.verbose:
        get_logger(__name__, verbose=True)
        logger.debug("Verbose mode enabled")

    if not os.path.isfile(args.input):
        logger.error(f"File '{args.input}' does not exist.")
        return

    density = None
    if is_chgcar_input(args):
        if args.chgcar and not os.path.isfile(args.chgcar):
            logger.error(f"CHGCAR file '{args.chgcar}' does not exist.")
            return
        slab, density = load_density(args.input, args.chgcar)
        density = prepare_density(density, args.supercell, args.mesh_refinement)
    else:
        slab = read(args.input, index=-1)

    slab = slab * args.supercell

    overrides = None
    if args.styles:
        with open(args.styles) as fh:
            overrides = yaml.safe_load(fh)
    colors, radii = resolve_atom_styles(slab, overrides=overrides, radius_scale=args.radius_scale)

    povray_settings = povray_settings_from_args(args, extra={
        'point_lights': [],
        'area_light': [(2., 3., 125.),  # location
                       'White',         # color
                       .95, .8, 5, 4],  # width, height, Nlamps_x, Nlamps_y
    })

    isosurface_data = build_isosurface_data(density, args) if density is not None else None

    if args.moments is not None:
        moments = get_moment_vectors(args.moments, collinear=args.collinear)
        ncells = int(np.prod(args.supercell))
        if ncells > 1:
            # ase tiles atoms in block-repeat order, so repeat the moment block.
            moments = np.tile(moments, (ncells, 1))
        if len(moments) != len(slab):
            raise ValueError(
                f"Moment count ({len(moments)}) does not match atom count "
                f"({len(slab)}); ensure --moments matches the input structure "
                f"and --supercell.")
        if isosurface_data or args.bonds:
            logger.warning("--moments takes precedence; --isovalue/--bond ignored in this render.")
        logger.info(f"Drawing magnetic-moment arrows from {args.moments}"
                    f"{' (collinear)' if args.collinear else ''}")
        render_structure_with_moments(
            slab, moments, args.output, rotation=args.rotation,
            colors=colors, radii=radii, povray_settings=povray_settings,
            cleanup=not args.keep_pov, length=args.moment_length,
            min_moment=args.moment_threshold,
            scale_by_magnitude=args.moment_by_magnitude,
        )
    elif args.bonds:
        n = len(slab)
        for i, j in args.bonds:
            if not (0 <= i < n and 0 <= j < n):
                logger.warning(f"Bond ({i}, {j}) skipped: index out of range (0-{n - 1})")
        logger.info(f"Drawing {len(args.bonds)} dashed bond(s)")
        render_structure_with_bonds(
            slab, args.bonds, args.output, rotation=args.rotation,
            colors=colors, radii=radii, povray_settings=povray_settings,
            isosurface_data=isosurface_data, cleanup=not args.keep_pov,
            bond_color=parse_rgb(args.bond_color), bond_radius=args.bond_radius,
            dash_length=args.dash_length, gap_length=args.gap_length,
        )
    else:
        render_structure(
            slab, args.output, rotation=args.rotation,
            colors=colors, radii=radii, povray_settings=povray_settings,
            isosurface_data=isosurface_data, cleanup=not args.keep_pov,
        )


if __name__ == "__main__":
    main()
