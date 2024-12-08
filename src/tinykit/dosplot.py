from pymatgen.io.vasp import Vasprun 
from pymatgen.core import Structure 
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.core import OrbitalType, Spin
import matplotlib.pyplot as plt

import numpy as np

def separate_structures(structure: Structure) -> Structure:
    """returns a structure with adsorbate removed"""

    slab = structure.copy().remove_species(["H", "C"])
    adsorbate = structure.copy().remove_species(["Pt", "Sn"])

    return slab, adsorbate

def get_surface_dos(completeDos: CompleteDos, structure: Structure, natoms: int = 8) -> list:
    """returns a list of surface sites"""
    
    #sort structure sites by z-coordinate
    surface_sites_sorted = sorted(structure.sites, key=lambda x: x.coords[2], reverse=True)[:natoms]
    spd_site_dos_list: list[dict] = [ completeDos.get_site_spd_dos(site) for site in surface_sites_sorted ]
    #add site dos together 
    total_site_dos = spd_site_dos_list[0]
    for spd_site_dos in spd_site_dos_list[1:]:
        for orbital in spd_site_dos.keys():
            total_site_dos[orbital] += spd_site_dos[orbital]

    return total_site_dos

def split_surface_and_bulk_dos(completeDos: CompleteDos, structure: Structure, natoms: int = 8) -> tuple:
    """returns a tuple of surface and bulk dos"""

    #sort structure sites by z-coordinate
    sites_sorted = sorted(structure.sites, key=lambda x: x.coords[2], reverse=True)
    surface_sites = sites_sorted[:natoms]
    bulk_sites = sites_sorted[natoms:]
    surface_spd_site_dos_list: list[dict] = [ completeDos.get_site_spd_dos(site) for site in surface_sites ]
    bulk_spd_site_dos_list: list[dict] = [ completeDos.get_site_spd_dos(site) for site in bulk_sites ]
    
    #add site dos together 
    bulk_pdos = bulk_spd_site_dos_list[0]
    surface_pdos = surface_spd_site_dos_list[0]

    for spd_site_dos in bulk_spd_site_dos_list[1:]:
        for orbital in spd_site_dos.keys():
            bulk_pdos[orbital] += spd_site_dos[orbital]

    for spd_site_dos in surface_spd_site_dos_list[1:]:
        for orbital in spd_site_dos.keys():
            surface_pdos[orbital] += spd_site_dos[orbital]

    return surface_pdos, bulk_pdos

#lets compare the dos of the surface atoms and the adsorbate 

vasprun_file = '/home/wladerer/research/profile/slabs_001/slab_0/pdos/vasprun.xml'
vasprun = Vasprun(vasprun_file, parse_potcar_file=False)
dos = vasprun.complete_dos
structure = vasprun.final_structure

surface_dos, bulk_dos = split_surface_and_bulk_dos(dos, structure)
energy = dos.energies - dos.efermi
#plot the surface dos, #energy on y-axis, dos on x-axis
plt.figure()

plt.plot(surface_dos[OrbitalType.d].densities[Spin.up], energy, label='Surface d')
plt.plot(bulk_dos[OrbitalType.d].densities[Spin.up], energy, label='Bulk d')

plt.legend()
plt.xlabel('Density of States')
plt.ylabel('Energy (eV)')
plt.title('Surface DOS')

plt.show()



#lets plot the first derivative of the surface and bulk d-orbital dos
plt.figure()

plt.scatter(surface_dos[OrbitalType.d].densities[Spin.up] - bulk_dos[OrbitalType.d].densities[Spin.up]/10, energy, label='d')

plt.legend()
plt.xlabel('Density of States')
plt.ylabel('Energy (eV)')
plt.title('Surface DOS')

plt.show()