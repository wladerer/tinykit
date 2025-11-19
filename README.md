# TinyKit

A lightweight toolkit for computational materials science workflows, focused on surface adsorption studies and VASP calculations.

## Overview

TinyKit provides command-line tools for generating and analyzing surface structures, adsorption configurations, and related computational chemistry tasks using VASP (Vienna Ab initio Simulation Package).

## Installation

```bash
pip install -e .
```

## Tools

### `adsorb` - Surface Adsorption
Generate adsorbed structures on surfaces with support for both single and multiple adsorbates.

```bash
# Single adsorbate on a surface
adsorb POSCAR H2O --supercell 2 2 1 -d 1.8

# Multiple adsorbates with sampling
adsorb POSCAR Ag --multiple 2 --min-distance 2.0 --max-samples 50

# Specify adsorption sites
adsorb POSCAR OH --multiple 3 --sites ontop bridge
```

**Features:**
- Pre-defined molecules from JSON or single-atom adsorbates (e.g., Ag, Au, Pt)
- Multiple simultaneous adsorption with distance constraints
- Site type filtering (ontop, bridge, hollow)
- Random sampling for large configuration spaces
- Custom INCAR templates

### `slabgen` - Slab Generation
Generate surface slabs from bulk structures with automatic Miller index enumeration.

```bash
# Generate slabs up to (1,1,1) Miller indices
slabgen POSCAR --hkl 1 --thicknesses 12 15 --vacuums 15

# Allow asymmetric slabs
slabgen POSCAR --hkl 2 -a

# Skip Tasker analysis (use pymatgen only)
slabgen POSCAR --no-tasker
```

**Features:**
- Automatic Miller plane generation
- Multiple thickness/vacuum combinations
- Tasker analysis for polar surfaces (via surfaxe)
- Selective dynamics setup

### `deploy` - Batch VASP Input Generation
Convert structure trajectories into VASP calculation directories.

```bash
# From trajectory file
deploy structures.traj -i INCAR -k KPOINTS -o calculations/

# Freeze bottom layers
deploy structures.extxyz --freeze 10.0
```

**Supported formats:** VASP XDATCAR, ASE trajectory, extended XYZ, etc.

### `charge` - Charged Slab Calculations
Set up VASP calculations for charged surfaces with varying electron counts.

```bash
# Generate NELECT series
charge POSCAR --start 0.1 --stop 1.0 --step 0.1 --kpoints 5 5 1

# With dipole correction
charge POSCAR --dipole --start -0.5 --stop 0.5 --step 0.1
```

### `slabviz` - Structure Visualization
Render high-quality structure images using POV-Ray.

```bash
# Basic rendering
slabviz CONTCAR -o output.png

# Custom view and supercell
slabviz CONTCAR --rotation 90 0 45 --supercell 2 2 1

# Custom atom colors
slabviz CONTCAR -c colors.yaml
```

### `stmplot` - STM Image Simulation
Generate constant-current STM images from VASP charge density files.

```bash
# From PARCHG file
stmplot PARCHG -c 0.001 -o stm.png

# Tiled periodic image
stmplot PARCHG --current 0.001 --tiles 3
```

## Dependencies

- pymatgen
- ASE
- numpy
- matplotlib
- surfaxe
- POV-Ray (for visualization)

## Project Structure

```
src/tinykit/
├── adsorb.py       # Adsorption structure generation
├── slabgen.py      # Slab generation
├── deploy.py       # Batch VASP input creation
├── charge.py       # Charged surface calculations
├── slabviz.py      # Structure visualization
├── stmplot.py      # STM image simulation
└── molecules.json  # Pre-defined adsorbate molecules
```
