#!/usr/bin/env python
"""Render bulk structures with POV-Ray."""
import argparse
import logging
import os

import yaml
from ase.io import read

from tinykit.povray import (
    resolve_atom_styles, render_structure, add_render_args, povray_settings_from_args,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description='Render bulk structures with POV-Ray.')
    parser.add_argument('input', help='Input structure file (POSCAR, CONTCAR, vasprun.xml, ...)')
    parser.add_argument('-o', '--output', help='Output image file name', default='structure.png')
    parser.add_argument('-c', '--colors', '--styles', dest='styles', default=None,
                        help='YAML file overriding per-element color and/or radius, '
                             'e.g. "Fe: [255,100,0]" or "C: {radius: 0.75}"')
    parser.add_argument('--rotation', help='Rotation as three angles (x y z, degrees)',
                        default=[-52, -48, -30], nargs=3, type=float)
    parser.add_argument('--supercell', help='Supercell dimensions', default=[1, 1, 1], nargs=3, type=int)
    parser.add_argument('-v', '--verbose', help='Enable verbose output', action='store_true')
    add_render_args(parser, default_height=900)
    return parser


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = build_parser().parse_args(args)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not os.path.isfile(args.input):
        logger.error(f"File '{args.input}' does not exist.")
        return

    try:
        atoms = read(args.input, index=-1)
    except Exception as e:
        logger.error(f"Could not read structure from '{args.input}': {e}")
        return

    atoms = atoms * args.supercell
    logger.debug(f"Loaded {len(atoms)} atoms ({atoms.get_chemical_formula()}) after supercell {args.supercell}")

    overrides = None
    if args.styles:
        with open(args.styles, 'r') as yaml_file:
            overrides = yaml.safe_load(yaml_file)
        logger.debug(f"Loaded style overrides for: {sorted(overrides)}")

    colors, radii = resolve_atom_styles(atoms, overrides=overrides, radius_offset=-0.4)

    povray_settings = povray_settings_from_args(args)

    image_path = render_structure(
        atoms, args.output, rotation=args.rotation,
        colors=colors, radii=radii, povray_settings=povray_settings,
        cleanup=not args.keep_pov,
    )
    logger.info(f"Wrote {image_path}")


if __name__ == "__main__":
    main()
