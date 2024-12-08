#!/usr/bin/env python
import argparse
import numpy as np
import yaml
from ase.io import read, write
import os

# Default color and radius mappings
default_atom_type_to_color_map = {
    'Pt': (208, 208, 224),
    'Sn': (102, 128, 128),
    'H': (85, 173, 211),
    'C': (166,84,77),
}

default_atom_type_to_radius_map = {
    'Pt': 1.5,
    'Sn': 1.5,
    'H': 0.8,
    'C': 1.0,
}

# Convert colors to 0 to 1 scale
def normalize_colors(atom_colors):
    return {k: tuple(vv / 255 for vv in v) for k, v in atom_colors.items()}

def array_to_rotation_string(array):
    return f"{array[0]}x,{array[1]}y,{array[2]}z"

def update_image_extension(string):
    #if doesnt end with .pov add .pov
    #also if it does not have an extension add .pov
    if not string.endswith('.pov') and '.' not in string:
        return string + '.pov'

    if not string.endswith('.pov'):
        return string.replace('.png', '.pov')

    return string

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Visualize VASP structures with custom colors.')
    parser.add_argument('input', help='Input VASP file (CONTCAR or vasprun.xml)')
    parser.add_argument('-c', '--colors', help='YAML file specifying custom colors for atoms', default=None)
    parser.add_argument('-o', '--output', help='Output file name', default='slab.png')
#    parser.add_argument('--rotation', help='Rotation of the slab', default=[2.87, 3.43 ,-0.0229], nargs=3, type=float)
    parser.add_argument('--rotation', help='Rotation of the slab', default=[0,0,0], nargs=3, type=float)
    parser.add_argument('--supercell', help='Supercell dimensions', default=[1, 1, 1], nargs=3, type=int)

    args = parser.parse_args()

    # Determine input file type
    if not os.path.isfile(args.input):
        print(f"Error: File '{args.input}' does not exist.")
        return

    if args.input.endswith('CONTCAR'):
        slab = read(args.input, index=-1)
    elif args.input.endswith('vasprun.xml'):
        slab = read(args.input, index=-1)
    else:
        print("Error: Input file must be a CONTCAR or vasprun.xml.")
        return

    slab = slab * args.supercell
    # Load custom colors from YAML if specified
    atom_type_to_color_map = default_atom_type_to_color_map.copy()
    if args.colors:
        with open(args.colors, 'r') as yaml_file:
            custom_colors = yaml.safe_load(yaml_file)
            atom_type_to_color_map.update(custom_colors)

    #normalize colors
    atom_type_to_color_map = normalize_colors(atom_type_to_color_map)
    colors = [atom_type_to_color_map[a.symbol] for a in slab]
    radii = np.array([default_atom_type_to_radius_map[a.symbol] for a in slab]) - 0.4

    # hatoms = [a for a in slab if a.symbol == 'H']

    # # Get x and y dimensions of the slab
    # x = slab.cell[0]
    # y = slab.cell[1]

    # closest_h = min(hatoms, key=lambda a: a.position[0]**2 + a.position[1]**2)
    # slab.translate(-closest_h.position)

    povray_settings = {
        'canvas_width': None,  # Width of canvas in pixels
        'canvas_height': 1100,  # Height of canvas in pixels
        'camera_dist': 20.,  # Distance from camera to front atom
        'celllinewidth': 0.0,  # Thickness of cell lines
    }

    args.output = update_image_extension(args.output)
    rotation = array_to_rotation_string(args.rotation)
    renderer = write(args.output, slab, format='pov', colors=colors, rotation=rotation, radii=radii, povray_settings=povray_settings).render()

if __name__ == "__main__":
    main()

