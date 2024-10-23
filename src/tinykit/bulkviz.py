#!/usr/bin/env python
import argparse
import numpy as np
import yaml
from ase.io import read, write
import os




def array_to_rotation_string(array):
    return f"{array[0]}x,{array[1]}y,{array[2]}z"

def update_image_extension(string):
    #if doesnt end with .pov add .pov
    #also if it does not have an extension add .pov
    if not string.endswith('.pov') and '.' not in string:
        return f'{string}.pov'

    if not string.endswith('.pov'):
        return string.replace('.png', '.pov')

    return string

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Visualize VASP structures with custom colors.')
    parser.add_argument('input', help='Input VASP file (POSCAR or vasprun.xml)')
    parser.add_argument('-o', '--output', help='Output file name', default='structure.png')
    parser.add_argument('--rotation', help='Rotation of the slab', default=[-52, -48 ,-30], nargs=3, type=float)
    parser.add_argument('--supercell', help='Supercell dimensions', default=[1, 1, 1], nargs=3, type=int)

    args = parser.parse_args()

    # Determine input file type
    if not os.path.isfile(args.input):
        print(f"Error: File '{args.input}' does not exist.")
        return

    if args.input.endswith('CONTCAR'):
        slab = read(args.input, index=-1)
    elif args.input_file.endswith('vasprun.xml'):
        slab = read(args.input, index=-1)
    else:
        print("Error: Input file must be a CONTCAR or vasprun.xml.")
        return

    slab = slab * args.supercell

    povray_settings = {
        'canvas_width': None,  # Width of canvas in pixels
        'canvas_height': 900,  # Height of canvas in pixels
        'camera_dist': 20.,  # Distance from camera to front atom
        'celllinewidth': 0.0,  # Thickness of cell lines
    }

    args.output = update_image_extension(args.output)
    rotation = array_to_rotation_string(args.rotation)
    renderer = write(args.output, slab, format='pov', rotation=rotation,  povray_settings=povray_settings).render()

if __name__ == "__main__":
    main()

