#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import sys
from pymatgen.core import Structure
from pymatgen.io.vasp import Procar, Outcar

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_surface_character(structure_path, procar_path, outcar_path, 
                          layer_tolerance=1.0, target_k_idx=None, energy_window=1.0):
    
    # 1. Load Files
    try:
        struct = Structure.from_file(structure_path)
        procar = Procar(procar_path)
        outcar = Outcar(outcar_path)
    except Exception as e:
        logger.error(f"Failed to load VASP files: {e}")
        sys.exit(1)

    # 2. Safety check for Fermi Level
    e_fermi = outcar.efermi
    if e_fermi is None:
        logger.error(f"Fermi level not found in {outcar_path}. Is the calculation finished?")
        sys.exit(1)
    
    logger.info(f"Fermi Level: {e_fermi:.4f} eV")
    logger.info(f"Filtering states within ±{energy_window} eV of Fermi.")

    # 3. Geometry Logic
    a, b = struct.lattice.matrix[0], struct.lattice.matrix[1]
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)

    positions = struct.cart_coords
    projections = np.dot(positions, normal)

    unique_projections = np.unique(np.round(projections / layer_tolerance) * layer_tolerance)
    unique_projections.sort()

    # Define Surface Indices (Top 3 layers)
    surface_layers = unique_projections[-3:]
    surface_atom_indices = [
        i for i, val in enumerate(projections)
        if any(np.isclose(val, layer, atol=layer_tolerance) for layer in surface_layers)
    ]
    
    logger.debug(f"Identified {len(surface_atom_indices)} atoms in the top 3 surface layers.")

    results = []

    # 4. Projection Analysis
    for spin, data in procar.data.items():
        # data shape: (nkpoints, nbands, nions, norbitals)
        surface_intensity = np.sum(data[:, :, surface_atom_indices, :], axis=(2, 3))

        k_indices = [target_k_idx] if target_k_idx is not None else range(procar.nkpoints)

        for k_idx in k_indices:
            for b_idx in range(procar.nbands):
                energy_raw = procar.eigenvalues[spin][k_idx, b_idx]
                e_relative = energy_raw - e_fermi
                
                if abs(e_relative) <= energy_window:
                    results.append({
                        'spin': spin,
                        'kpoint_index': k_idx,
                        'band_index': b_idx,
                        'energy_raw': energy_raw,
                        'energy_fermi': e_relative,
                        'surface_char': surface_intensity[k_idx, b_idx]
                    })

    return sorted(results, key=lambda x: x['surface_char'], reverse=True)

def print_results(results, length=20):
    if not results:
        logger.warning("No states found within the specified energy window.")
        return

    # Using standard print for the table to keep it clean, or logger.info for consistency
    header = f"\n{'Spin':<10} | {'Band':<5} | {'K-idx':<6} | {'E-Ef (eV)':<10} | {'Surface Char'}"
    print(header)
    print("-" * 65)
    for state in results[:length]: 
        print(f"{str(state['spin']):<10} | {state['band_index']:<5} | "
              f"{state['kpoint_index']:<6} | {state['energy_fermi']:>10.4f} | "
              f"{state['surface_char']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze VASP outputs to find surface-localized states.")
    parser.add_argument("-s", "--structure", default="CONTCAR")
    parser.add_argument("-p", "--procar", default="PROCAR")
    parser.add_argument("-o", "--outcar", default="OUTCAR")
    parser.add_argument("-w", "--window", type=float, default=1.0)
    parser.add_argument("-t", "--tolerance", type=float, default=1.0)
    parser.add_argument("-k", "--kidx", type=int, default=None)
    parser.add_argument("-n", "--num", type=int, default=20)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logs")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    results = get_surface_character(
        args.structure, args.procar, args.outcar,
        args.tolerance, args.kidx, args.window
    )
    
    print_results(results, length=args.num)

if __name__ == "__main__":
    main()
