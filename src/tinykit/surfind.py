#!/usr/bin/env python3
import argparse
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Procar, Outcar

def get_surface_character(structure_path, procar_path, outcar_path, 
                          layer_tolerance=1.0, target_k_idx=None, energy_window=1.0):
    """
    Analyzes surface character with corrected indexing and energy filtering.
    """
    struct = Structure.from_file(structure_path)
    procar = Procar(procar_path)
    outcar = Outcar(outcar_path)
    e_fermi = outcar.efermi

    # Calculate surface normal and projections
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

    print(f"Fermi Level: {e_fermi:.4f} eV")
    print(f"Filtering for states within ±{energy_window} eV of Fermi.")

    results = []

    for spin, data in procar.data.items():
        # Calculate intensity for all bands/k-points at once for this spin
        # data shape: (nkpoints, nbands, nions, norbitals)
        surface_intensity = np.sum(data[:, :, surface_atom_indices, :], axis=(2, 3))

        k_indices = [target_k_idx] if target_k_idx is not None else range(procar.nkpoints)

        for k_idx in k_indices:
            for b_idx in range(procar.nbands):
                energy_raw = procar.eigenvalues[spin][k_idx, b_idx]
                e_relative = energy_raw - e_fermi
                
                # Energy Filter
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
    """Prints results in a clean fashion"""
    if not results:
        print("No states found within the specified energy window.")
        return

    print(f"\n{'Spin':<10} | {'Band':<5} | {'K-idx':<6} | {'E-Ef (eV)':<10} | {'Surface Char'}")
    print("-" * 65)
    for state in results[:length]: 
        print(f"{str(state['spin']):<10} | {state['band_index']:<5} | "
              f"{state['kpoint_index']:<6} | {state['energy_fermi']:>10.4f} | "
              f"{state['surface_char']:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze VASP outputs to find surface-localized states near the Fermi level."
    )

    # File paths
    parser.add_argument("-s", "--structure", default="CONTCAR", help="Path to structure file (default: CONTCAR)")
    parser.add_argument("-p", "--procar", default="PROCAR", help="Path to PROCAR file (default: PROCAR)")
    parser.add_argument("-o", "--outcar", default="OUTCAR", help="Path to OUTCAR file (default: OUTCAR)")

    # Analysis parameters
    parser.add_argument("-w", "--window", type=float, default=1.0, help="Energy window around Fermi level in eV (default: 1.0)")
    parser.add_argument("-t", "--tolerance", type=float, default=1.0, help="Layer grouping tolerance in Angstroms (default: 1.0)")
    parser.add_argument("-k", "--kidx", type=int, default=None, help="Specific K-point index to analyze (default: all)")
    parser.add_argument("-n", "--num", type=int, default=20, help="Number of results to display (default: 20)")

    args = parser.parse_args()

    # Execute main logic
    try:
        results = get_surface_character(
            structure_path=args.structure,
            procar_path=args.procar,
            outcar_path=args.outcar,
            layer_tolerance=args.tolerance,
            target_k_idx=args.kidx,
            energy_window=args.window
        )
        
        print_results(results, length=args.num)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
