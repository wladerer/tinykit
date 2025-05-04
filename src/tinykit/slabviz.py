#!/usr/bin/env python
import argparse
import numpy as np
import yaml
import json
from ase.io import read, write
import os

import pathlib
path = pathlib.Path(__file__).parent.absolute()

atom_template_path = path / 'resources' / 'atom_templates.json'
with open(atom_template_path, 'r') as f:
    atom_template = json.load(f)


def hex_to_rgb(hex_string):
    return tuple(int(hex_string.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

default_atom_type_to_color_map = {k: hex_to_rgb(v['color']) for k, v in atom_template.items()}
default_atom_type_to_radius_map = {k: v['radius'] for k, v in atom_template.items()}

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

    slab = read(args.input, index=-1)

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

    povray_settings = {
        'canvas_width': None,  # Width of canvas in pixels
        'canvas_height': 1100,  # Height of canvas in pixels
        'camera_dist': 20.,  # Distance from camera to front atom
        'celllinewidth': 0.0,  # Thickness of cell lines
        'camera_type'  : 'orthographic', # perspective, ultra_wide_angle
	    'point_lights' : [], #[(18,20,40), 'White'],[(60,20,40),'White'],             # [[loc1, color1], [loc2, color2],...]
	    'area_light'   : [(2., 3., 125.), # location
	                      'White',       # color
	                      .95, .8, 5, 4], # width, height, Nlamps_x, Nlamps_y
    }

    args.output = update_image_extension(args.output)
    rotation = array_to_rotation_string(args.rotation)
    renderer = write(args.output, slab, format='pov', colors=colors, rotation=rotation, radii=radii, povray_settings=povray_settings).render()

    #remove .ini and .pov files 
    os.remove(args.output.replace('.pov', '.ini'))
    os.remove(args.output.replace('.pov', '.pov'))

if __name__ == "__main__":
    main()

