#!/usr/bin/env python3
"""Regenerate atom_templates.json from VESTA's elements.ini.

VESTA's palette is the de-facto standard in the materials community, so using
it keeps tinykit renders consistent with what people see in VESTA itself. Any
element VESTA does not cover (Z > 96) falls back to ASE's jmol colors and
covalent radii, so the table is complete for all elements ASE knows.

VESTA's ``elements.ini`` columns are::

    Z  Symbol  r_atomic  r_vdw  r_ionic  R  G  B        (R/G/B on the 0-1 scale)

Each output entry stores a hex ``color`` plus three physical radii. Renderers
use ``color`` and ``radius`` (the atomic/covalent radius); ``vdw_radius`` and
``ionic_radius`` are kept for future use (e.g. space-filling or ionic views).

Usage::

    python generate_atom_templates.py [path/to/elements.ini]

If no path is given, common VESTA install locations (including the Nix store)
are searched. Run from this directory; it overwrites atom_templates.json here.
"""
import glob
import json
import sys
from pathlib import Path

from ase.data import (atomic_numbers, chemical_symbols, covalent_radii,
                      vdw_radii)
from ase.data.colors import jmol_colors


def find_elements_ini() -> Path:
    candidates = []
    for pattern in (
        "/nix/store/*/lib/VESTA/elements.ini",
        "/usr/share/vesta/elements.ini",
        "/usr/local/share/VESTA/elements.ini",
        str(Path.home() / "Apps" / "**" / "elements.ini"),
    ):
        candidates += glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            "Could not locate VESTA elements.ini; pass its path explicitly.")
    return Path(sorted(candidates)[0])


def rgb_to_hex(r, g, b) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        *(max(0, min(255, round(c * 255))) for c in (r, g, b)))


def parse_vesta(path: Path) -> dict:
    """Return {symbol: {color, radius, vdw_radius, ionic_radius}} from VESTA."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        symbol = parts[1]
        if symbol == "D" or symbol not in atomic_numbers:  # skip deuterium alias
            continue
        try:
            r_atomic, r_vdw, r_ionic = (float(parts[2]), float(parts[3]), float(parts[4]))
            r, g, b = (float(parts[5]), float(parts[6]), float(parts[7]))
        except ValueError:
            continue
        out[symbol] = {
            "color": rgb_to_hex(r, g, b),
            "radius": round(r_atomic, 3),
            "vdw_radius": round(r_vdw, 3),
            "ionic_radius": round(r_ionic, 3),
        }
    return out


def main():
    ini = Path(sys.argv[1]) if len(sys.argv) > 1 else find_elements_ini()
    print(f"Reading VESTA palette from {ini}")
    vesta = parse_vesta(ini)

    templates = {}
    fallbacks = []
    for z in range(1, len(chemical_symbols)):
        sym = chemical_symbols[z]
        if sym in vesta:
            templates[sym] = vesta[sym]
        else:
            # ASE jmol colors + covalent radii for elements VESTA omits (Z > 96)
            r, g, b = jmol_colors[z] if z < len(jmol_colors) else (0.5, 0.5, 0.5)
            cov = covalent_radii[z] if z < len(covalent_radii) else 1.5
            vdw = vdw_radii[z] if z < len(vdw_radii) else float("nan")
            templates[sym] = {
                "color": rgb_to_hex(r, g, b),
                "radius": round(float(cov), 3),
                "vdw_radius": None if vdw != vdw else round(float(vdw), 3),  # NaN -> None
                "ionic_radius": None,
            }
            fallbacks.append(sym)

    out_path = Path(__file__).parent / "atom_templates.json"
    out_path.write_text(json.dumps(templates, indent=2) + "\n")
    print(f"Wrote {len(templates)} elements to {out_path}")
    if fallbacks:
        print(f"ASE jmol/covalent fallback for: {' '.join(fallbacks)}")


if __name__ == "__main__":
    main()
