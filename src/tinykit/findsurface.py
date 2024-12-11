from pymatgen.core import Structure 

from pyprocar.io import Parser 
from pyprocar.plotter import EBSPlot
from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
import matplotlib.pyplot as plt

import numpy as np 

import argparse


def get_surface_and_bulk(structure: Structure, natoms: int = 8) -> tuple[list]:
    """returns a list of surface sites"""
    #add index property to each site
    for i, site in enumerate(structure.sites):
        site.index = i

    #sort structure sites by z-coordinate
    surface_sites_sorted = sorted(structure.sites, key=lambda x: x.coords[2], reverse=True)[:natoms]
    bulk_sites_sorted = sorted(structure.sites, key=lambda x: x.coords[2], reverse=True)[natoms:]

    return surface_sites_sorted, bulk_sites_sorted



def surface_projection_from_sites(ebs: EBSPlot, surface_sites: list) -> list:
    """returns a list of surface projections"""
    surface_projections = ebs.ebs_sum(atoms=surface_sites)

    return surface_projections


def analyze_surface_projections(ebs, structure: Structure, natoms: int = 8, top_n: int = 15):
    surface_sites, bulk_sites = get_surface_and_bulk(structure, natoms)
    surface_projections = surface_projection_from_sites(ebs, [site.index for site in surface_sites])
    bulk_projection = surface_projection_from_sites(ebs, [site.index for site in bulk_sites])
    # get the top bands with the largest surface to bulk ratio
    total_surface_projections = np.sum(surface_projections, axis=(0, 2))
    total_bulk_projections = np.sum(bulk_projection, axis=(0, 2))
    surface_projection_ratio = total_surface_projections / total_bulk_projections

    top_bands = np.argsort(surface_projection_ratio)[::-1][:top_n]
     
    return top_bands 

# Example usage



if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Find the top bands with the largest surface to bulk ratio.")
    parser.add_argument('directory', type=str,
                        help='Path to the directory containing the PROCAR file')
    args = parser.parse_args()
    parser = Parser(code='vasp', dir = args.directory)
    structure = Structure.from_file(f"{args.directory}/CONTCAR")
    ebs = parser.ebs 

    top_bands = analyze_surface_projections(ebs, structure)
    top_bands = [band + 1 for band in top_bands]

    print("Top bands with largest surface to bulk ratio (1 index):")
    print(top_bands)
