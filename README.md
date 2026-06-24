# TinyKit

A lightweight toolkit for computational materials science workflows, focused on surface adsorption studies and VASP calculations.

## Overview

TinyKit provides command-line tools for generating and analyzing surface structures, adsorption configurations, and related computational chemistry tasks using VASP (Vienna Ab initio Simulation Package).

## Installation

```bash
pip install -e .
```

This installs a single `tinykit` command with subcommands, plus an individual command for each tool.

```bash
tinykit --help              # list all subcommands
tinykit slabgen --help      # help for one tool
tinykit slabgen POSCAR --hkl 111   # equivalent to: slabgen POSCAR --hkl 111
```

Tab completion is provided by [argcomplete](https://github.com/kislyuk/argcomplete). To enable it for the unified command:

```bash
eval "$(register-python-argcomplete tinykit)"
```

## Shared options for input-generating tools

`adsorb`, `slabgen`, and `charge` share a common set of flags:

- `--preset NAME` selects a named INCAR preset (`adsorb`, `slab`, `charge`). Presets live in `src/tinykit/resources/incars.yaml`; edit values there.
- `--incar FILE` uses a custom INCAR file, overriding `--preset`.
- `--kpoints KX KY KZ` sets a gamma-centered k-point mesh.
- `--functional NAME` selects the POTCAR functional family (default `PBE`).
- `--no-overwrite` skips directories that already exist.

`deploy` reads its INCAR and KPOINTS from files instead of presets, and also accepts `--functional` and `--no-overwrite`.

## Tools

### `adsorb` - Surface Adsorption
Generate adsorbed structures on surfaces with support for both single and multiple adsorbates.

```bash
# Single adsorbate on a surface
adsorb POSCAR H2O --supercell 2 2 1 -d 1.8

# Multiple adsorbates with random sampling of the configuration space
adsorb POSCAR Ag --multiple 2 --min-distance 2.0 --max-samples 50 --seed 0

# Restrict to specific site types
adsorb POSCAR OH --multiple 3 --sites ontop bridge
```

**Features:**
- Pre-defined molecules from JSON or single-atom adsorbates (e.g., Ag, Au, Pt)
- Multiple simultaneous adsorption with site- and atom-distance constraints
- Site type filtering with `--sites` (ontop, bridge, hollow)
- Symmetry reduction of configurations, with optional random sampling (`--max-samples`, `--seed`)
- Parallel generation with `-j/--jobs`

### `slabgen` - Slab Generation
Generate surface slabs from bulk structures with automatic Miller index enumeration.

```bash
# Generate slabs for the (1,1,1) Miller index
slabgen POSCAR --hkl 111 --thicknesses 12 15 --vacuum 15

# Enumerate all slabs up to a maximum Miller index
slabgen POSCAR --max-hkl 2

# Freeze the bottom layers
slabgen POSCAR --hkl 111 --freeze-mode bottom --layers 2
```

**Features:**
- Specific Miller index (`--hkl`) or automatic enumeration (`--max-hkl`)
- Multiple thickness values, with vacuum in Angstroms
- Both symmetric and asymmetric terminations are generated and deduplicated
- Selective dynamics with `center`, `bottom`, or `top` freezing modes

### `deploy` - Batch VASP Input Generation
Convert structure trajectories into VASP calculation directories.

```bash
# From trajectory file
deploy structures.traj -i INCAR -k KPOINTS -o calculations/

# Freeze atoms below a z-coordinate
deploy structures.extxyz --freeze 10.0
```

**Supported formats:** VASP XDATCAR, ASE trajectory, extended XYZ, and other ASE-readable formats.

### `charge` - Charged Slab Calculations
Set up VASP calculations for charged surfaces with varying electron counts.

```bash
# Generate a NELECT series
charge POSCAR --start 0.1 --stop 1.0 --step 0.1 --kpoints 5 5 1

# With dipole correction (referenced at the center of mass)
charge POSCAR --dipole --start -0.5 --stop 0.5 --step 0.1
```

### `slabviz` - Structure Visualization
Render structure images, optionally with charge-density isosurfaces, using POV-Ray.

```bash
# Basic rendering
slabviz CONTCAR -o output.png

# Custom view, supercell, and resolution
slabviz CONTCAR --rotation 90 0 45 --supercell 2 2 1 --width 1600

# Per-element color and radius overrides
slabviz CONTCAR -c styles.yaml

# Dashed lines between atom pairs (e.g. an adsorbate and the surface)
slabviz CONTCAR --bond 46 40 --bond-color '#cc2222' --bond-radius 0.07
```

**Dashed bonds:** `--bond I J` draws a dashed line between zero-based atom indices `I` and `J` (repeatable; indices refer to the structure after `--supercell`). Tune with `--bond-color` (hex or comma-separated RGB), `--bond-radius`, `--dash-length`, and `--gap-length`.

### `bulkviz` - Bulk Structure Visualization
Render bulk structures with POV-Ray, sharing the same style and rendering options as `slabviz`.

```bash
bulkviz POSCAR -o bulk.png --rotation -52 -48 -30 --show-cell
```

**Rendering options (slabviz and bulkviz):** `--radius-scale` (ball-and-stick atom size relative to covalent radii; default `0.6`, use `1.0` for near space-filling), `--width`/`--height`, `--camera-dist`, `--orthographic`/`--perspective`, `--show-cell`, and `--keep-pov` (retain the intermediate `.pov`/`.ini` files for manual editing).

### Colors and style overrides

Default atom colors and radii come from the **VESTA** palette (covalent radii), generated from VESTA's `elements.ini` by `resources/generate_atom_templates.py`; elements VESTA omits fall back to ASE's jmol colors. `slabviz` and `bulkviz` take a YAML file via `-c/--colors` (alias `--styles`) that overrides per-element color and/or radius for a figure without touching the bundled templates:

```yaml
Fe: [255, 100, 0]            # color only (RGB 0-255)
O:  "#00ff00"                # color only (hex)
C:  {radius: 0.75}           # radius only, or {color: ..., radius: ...}
```

### `stmplot` - STM Image Simulation
Generate constant-current STM images from VASP charge density files. Plots in real Angstrom coordinates with correct periodic tiling of oblique (e.g. hexagonal) cells.

```bash
# From a PARCHG file
stmplot PARCHG -c 0.001 -o stm.png

# Tiled periodic image with a chosen colormap and contrast clip
stmplot PARCHG --current 0.001 --tiles 3 --cmap inferno --clip 2 98
```

### `surfind` - Surface State Analysis
Find surface-localized electronic states from PROCAR and OUTCAR.

```bash
surfind -s CONTCAR -p PROCAR -o OUTCAR --window 1.0 --layers 2
```

### `magviz` - Magnetic Moment Export
Extract magnetic moments from `vasprun.xml` and write them to a CIF.

```bash
magviz vasprun.xml -o magmoms.cif          # non-collinear (vector) moments
magviz vasprun.xml --collinear             # collinear (scalar) moments
```

## Dependencies

- numpy
- ASE
- pymatgen
- matplotlib
- scipy
- pyyaml
- argcomplete
- POV-Ray (external program, for visualization)

## Project Structure

```
src/tinykit/
├── cli.py          # Unified `tinykit` dispatcher + shared CLI helpers
├── adsorb.py       # Adsorption structure generation
├── slabgen.py      # Slab generation
├── deploy.py       # Batch VASP input creation
├── charge.py       # Charged surface calculations
├── slabviz.py      # Structure / isosurface visualization
├── bulkviz.py      # Bulk structure visualization
├── stmplot.py      # STM image simulation
├── surfind.py      # Surface state analysis
├── magviz.py       # Magnetic moment export
├── presets.py      # Named INCAR preset loading
├── vaspio.py       # Centralized VASP input assembly/writing
├── povray.py       # Shared POV-Ray rendering helpers
├── molecules.json  # Pre-defined adsorbate molecules
└── resources/
    ├── incars.yaml                  # Named INCAR presets
    ├── atom_templates.json          # Default atom colors and radii (VESTA palette)
    └── generate_atom_templates.py   # Regenerates the above from VESTA's elements.ini
```
