#!/usr/bin/env python
"""Render structures and charge-density isosurfaces with POV-Ray."""
import argparse
import numpy as np
import yaml
import logging
from ase.io import read
from ase.calculators.vasp.vasp_auxiliary import VaspChargeDensity
import os

from tinykit.povray import (
    resolve_atom_styles, render_structure, render_structure_with_bonds,
    add_render_args, povray_settings_from_args, hex_to_rgb,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_chgcar_data(chgcar_file):
    """Load charge density data from CHGCAR file"""
    chgcar = VaspChargeDensity(chgcar_file)
    density = chgcar.chg[-1]  # Last frame is usually the total density
    atoms = chgcar.atoms[-1]  # VaspChargeDensity already provides Atoms objects
    return atoms, density


def tile_density_grid(density, supercell):
    """
    Tile density grid to match supercell dimensions
    
    Parameters:
    -----------
    density : numpy array
        3D charge density data
    supercell : list or tuple
        Supercell dimensions [nx, ny, nz]
    
    Returns:
    --------
    numpy array : Tiled density grid
    """
    if supercell == [1, 1, 1]:
        return density
    
    logger.info(f"Tiling density grid {supercell[0]}x{supercell[1]}x{supercell[2]}...")
    logger.info(f"Original grid shape: {density.shape}")
    
    tiled = np.tile(density, supercell)

    logger.info(f"Tiled grid shape: {tiled.shape}")
    
    return tiled


def interpolate_density_grid(density, factor=2):
    """
    Interpolate density grid to higher resolution for smoother isosurfaces
    
    Parameters:
    -----------
    density : numpy array
        3D charge density data
    factor : int
        Multiplication factor for grid dimensions (2 = 8x more points, 3 = 27x more points)
    
    Returns:
    --------
    numpy array : Interpolated density grid
    """
    from scipy.ndimage import zoom
    
    logger.info(f"Interpolating density grid from {density.shape} by factor of {factor}...")
    interpolated = zoom(density, factor, order=3)  # Cubic interpolation
    logger.info(f"New grid shape: {interpolated.shape}")
    
    return interpolated


def create_isosurface(density, atoms, isovalue, color=(0.5, 0.5, 1.0), transmittance=0.0):
    """
    Create isosurface data dictionary for POV-Ray rendering
    
    Parameters:
    -----------
    density : numpy array
        3D charge density data
    atoms : ASE Atoms object
        Structure containing cell information
    isovalue : float
        Isosurface value threshold
    color : tuple
        RGB color values (0-1 scale)
    transmittance : float
        Transparency value (0=opaque, 1=transparent)
    
    Returns:
    --------
    dict : Dictionary with isosurface parameters for write_pov
    """
    # Add transmittance to color tuple if provided
    if transmittance > 0:
        color = tuple(color) + (transmittance,)
    
    return {
        'density_grid': density,
        'cut_off': isovalue,
        'color': color,
        'material': 'ase3'  # Use default ASE material
    }


def parse_rgb(spec):
    """Parse a color given as a hex string ('#rrggbb') or comma-separated RGB
    into a 0-1 tuple. Comma values are treated as 0-255 if any exceeds 1."""
    spec = spec.strip()
    if spec.startswith('#') or (len(spec) == 6 and ',' not in spec):
        return tuple(c / 255 for c in hex_to_rgb(spec))
    vals = [float(x) for x in spec.split(',')]
    if any(v > 1 for v in vals):
        vals = [v / 255 for v in vals]
    return tuple(vals)


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(
        description='Visualize VASP structures with custom colors and charge density.')
    parser.add_argument('input', help='Input VASP file (CONTCAR, vasprun.xml, or CHGCAR)')
    parser.add_argument('-c', '--colors', '--styles', dest='styles', default=None,
                        help='YAML file overriding per-element color and/or radius, '
                             'e.g. "Fe: [255,100,0]" or "C: {radius: 0.75}"')
    parser.add_argument('-o', '--output', help='Output file name', default='slab.png')
    parser.add_argument('--rotation', help='Rotation of the slab', default=[0,0,0], nargs=3, type=float)
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

    # CHGCAR-specific arguments
    parser.add_argument('--chgcar', help='Path to CHGCAR file for charge density visualization', default=None)
    parser.add_argument('--input-is-chgcar', help='Treat input file as CHGCAR format', action='store_true')
    parser.add_argument('--isovalue', help='Isosurface value for charge density (use positive value, both +/- will be rendered if --dual-phase)', type=float, default=None)
    parser.add_argument('--iso-color', help='Isosurface color as comma-separated RGB (0-1). Default is pale blue for positive phase.', 
                       default='0.6,0.75,0.95')  # Pale blue - desaturated, high value
    parser.add_argument('--iso-color-negative', help='Negative isosurface color as comma-separated RGB (0-1). Default is pale orange (complementary to blue).', 
                       default='0.95,0.75,0.6')  # Pale orange - complementary to blue
    parser.add_argument('--iso-transmittance', help='Isosurface transparency (0=opaque, 1=transparent). Higher values = more transparent.', 
                       type=float, default=0.5)  # More transparent by default
    parser.add_argument('--dual-phase', help='Render both positive and negative isosurfaces', action='store_true')
    parser.add_argument('--color-scheme', help='Preset color scheme for dual-phase rendering', 
                       choices=['blue-orange', 'purple-yellow', 'teal-coral', 'green-magenta', 'custom'],
                       default='blue-orange')
    parser.add_argument('--mesh-refinement', help='Grid interpolation factor for smoother mesh (1=no interpolation, 2=8x points, 3=27x points)',
                       type=int, default=1, choices=[1, 2, 3, 4])

    add_render_args(parser, default_height=1100)
    return parser


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = build_parser().parse_args(args)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    if not os.path.isfile(args.input):
        logger.error(f"File '{args.input}' does not exist.")
        return

    is_chgcar_input = ('CHGCAR' in args.input.upper() or 
                       'CHG' in args.input.upper() or 
                       args.input_is_chgcar)
    
    if is_chgcar_input or args.chgcar:
        if args.chgcar:
            # Separate structure and charge files
            chgcar_file = args.chgcar
            if not os.path.isfile(chgcar_file):
                logger.error(f"CHGCAR file '{chgcar_file}' does not exist.")
                return
            logger.info(f"Loading structure from {args.input}...")
            slab = read(args.input, index=-1)
            logger.info(f"Loading charge density from {chgcar_file}...")
            _, density = load_chgcar_data(chgcar_file)
        else:
            # Input file is the CHGCAR
            logger.info(f"Loading structure and charge density from {args.input}...")
            slab, density = load_chgcar_data(args.input)
        
        if density is not None and args.supercell != [1, 1, 1]:
            density = tile_density_grid(density, args.supercell)
        
        if density is not None and args.mesh_refinement > 1:
            density = interpolate_density_grid(density, args.mesh_refinement)
    else:
        slab = read(args.input, index=-1)
        density = None

    # Apply supercell to atoms structure
    slab = slab * args.supercell
    
    overrides = None
    if args.styles:
        with open(args.styles, 'r') as yaml_file:
            overrides = yaml.safe_load(yaml_file)

    colors, radii = resolve_atom_styles(slab, overrides=overrides, radius_scale=args.radius_scale)

    povray_settings = povray_settings_from_args(args, extra={
        'point_lights': [],
        'area_light': [(2., 3., 125.),  # location
                       'White',         # color
                       .95, .8, 5, 4],  # width, height, Nlamps_x, Nlamps_y
    })

    # Add isosurface if charge density is loaded
    isosurface_data = None
    if density is not None:
        logger.info("Charge density statistics:")
        logger.info(f"  Min: {density.min():.6e}")
        logger.info(f"  Max: {density.max():.6e}")
        logger.info(f"  Mean: {density.mean():.6e}")
        logger.info(f"  Std: {density.std():.6e}")
        
        # Check if data has negative values (indicates wavefunctions or density difference)
        has_negative = density.min() < 0
        if has_negative:
            logger.info("  Data contains negative values (wavefunction or density difference)")
        
        if args.isovalue is None:
       
            if has_negative:
      
                abs_max = max(abs(density.min()), abs(density.max()))
                suggested = min(2 * density.std(), 0.3 * abs_max)
            else:
                
                suggested = min(density.mean() + 2 * density.std(), 0.5 * density.max())
            args.isovalue = abs(suggested)  # Always use positive value
            logger.info(f"Auto-detected isovalue: {args.isovalue:.6e}")
        else:
            args.isovalue = abs(args.isovalue)  # Ensure positive
            logger.info(f"Using isovalue: {args.isovalue:.6e}")
        
        abs_max = max(abs(density.min()), abs(density.max()))
        if args.isovalue >= abs_max:
            logger.warning(f"Isovalue {args.isovalue:.6e} is too large (max absolute value: {abs_max:.6e})")
            # Use the better strategy: mean + 1*std (or just std for data centered near zero)
            if has_negative and abs(density.mean()) < density.std():
                # Data is roughly symmetric around zero (like wavefunctions)
                args.isovalue = density.std()
                logger.warning(f"Adjusting to 1*std = {args.isovalue:.6e}")
            else:
                # Data has a bias
                args.isovalue = abs(density.mean()) + density.std()
                logger.warning(f"Adjusting to |mean| + 1*std = {args.isovalue:.6e}")
        
        # Parse colors - apply color scheme presets if not using custom
        if args.color_scheme != 'custom':
        
            color_schemes = {
                'blue-orange': {
                    'positive': (0.6, 0.75, 0.95),   # Pale blue
                    'negative': (0.95, 0.75, 0.6)    # Pale orange
                },
                'purple-yellow': {
                    'positive': (0.8, 0.7, 0.95),    # Pale purple
                    'negative': (0.95, 0.95, 0.7)    # Pale yellow
                },
                'teal-coral': {
                    'positive': (0.65, 0.9, 0.9),    # Pale teal
                    'negative': (0.95, 0.75, 0.75)   # Pale coral
                },
                'green-magenta': {
                    'positive': (0.7, 0.9, 0.75),    # Pale green
                    'negative': (0.95, 0.7, 0.9)     # Pale magenta
                }
            }
            scheme = color_schemes[args.color_scheme]
            iso_color_pos = scheme['positive']
            iso_color_neg = scheme['negative']
            logger.info(f"Using '{args.color_scheme}' color scheme")
        else:
       
            iso_color_pos = tuple(float(x) for x in args.iso_color.split(','))
            iso_color_neg = tuple(float(x) for x in args.iso_color_negative.split(','))
            logger.info(f"Using custom colors")
        
        isosurface_data = []
        isosurface_data.append(create_isosurface(
            density, 
            slab, 
            args.isovalue,
            color=iso_color_pos,
            transmittance=args.iso_transmittance
        ))
        logger.info(f"Adding positive isosurface: value={args.isovalue:.6e}, color={iso_color_pos}, transmittance={args.iso_transmittance}")
        
        if args.dual_phase or (has_negative and args.isovalue <= abs(density.min())):
            isosurface_data.append(create_isosurface(
                density, 
                slab, 
                -args.isovalue,  # Negative isovalue
                color=iso_color_neg,
                transmittance=args.iso_transmittance
            ))
            logger.info(f"Adding negative isosurface: value={-args.isovalue:.6e}, color={iso_color_neg}, transmittance={args.iso_transmittance}")
        elif has_negative and not args.dual_phase:
            logger.info(f"Data has negative values but --dual-phase not set. Use --dual-phase to render both phases.")


    if args.bonds:
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
