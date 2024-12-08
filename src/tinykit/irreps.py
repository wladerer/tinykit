from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#get reciprocal lattice

structure = Structure.from_file('/home/wladerer/research/profile/slabs_001/slab_0/CONTCAR')
reciprocal_lattice = SpacegroupAnalyzer(structure).get_ir_reciprocal_mesh()
#get the orbit / wyckoof positions of the reciprocal lattice


