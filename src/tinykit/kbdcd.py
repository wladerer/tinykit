#!/usr/bin/env python3
import argparse
import logging
import sys
from pymatgen.io.vasp import Wavecar, Poscar, Chgcar, Outcar

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_states_by_energy(wfn, e_fermi, window):
    """Finds k-point/band indices within E_fermi +/- window."""
    states = []
    logger.info(f"Searching for states within {window} eV of Fermi ({e_fermi:.4f} eV)...")
    
    for k_idx in range(wfn.nk):
        for b_idx in range(wfn.nb):
            # wfn.band_energy mapping: [kpoint_idx][band_idx] -> (energy, occupancy)
            energy = wfn.band_energy[k_idx][b_idx][0]
            if abs(energy - e_fermi) <= window:
                states.append((k_idx, b_idx))
    
    logger.debug(f"Found {len(states)} states in energy window.")
    return states

def main():
    parser = argparse.ArgumentParser(description="Generate a summed CHGCAR from specific WAVECAR states.")
    
    # Required Files
    parser.add_argument("-w", "--wavecar", default="WAVECAR", help="Path to WAVECAR")
    parser.add_argument("-p", "--poscar", default="CONTCAR", help="Path to POSCAR/CONTCAR")
    parser.add_argument("-o", "--outcar", default="OUTCAR", help="Path to OUTCAR")
    
    # Selection Criteria
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--window", type=float, help="Energy window (eV) around Fermi level to sum.")
    group.add_argument("--indices", type=str, help="Specific states as 'k1,b1;k2,b2' (0-based).")
    
    # Output/Misc settings
    parser.add_argument("--out", default="CHGCAR_SUM.vasp", help="Output filename")
    parser.add_argument("--scale", type=int, default=2, help="FFT grid scaling (default 2)")
    parser.add_argument("--spin", type=int, default=0, choices=[0, 1], help="Spin index (0 or 1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 1. Load basic VASP data
    logger.info(f"Loading input files: {args.wavecar}, {args.poscar}")
    try:
        wfn = Wavecar(args.wavecar)
        pos = Poscar.from_file(args.poscar)
    except Exception as e:
        logger.error(f"Failed to load VASP files: {e}")
        sys.exit(1)

    # 2. Identify states to process
    states_to_sum = []
    if args.window is not None:
        try:
            out = Outcar(args.outcar)
            states_to_sum = get_states_by_energy(wfn, out.efermi, args.window)
        except Exception as e:
            logger.error(f"Could not retrieve Fermi level from OUTCAR: {e}")
            sys.exit(1)
    else:
        try:
            for pair in args.indices.split(';'):
                k, b = map(int, pair.split(','))
                states_to_sum.append((k, b))
        except ValueError:
            logger.error("Invalid index format. Use 'k,b;k,b' (e.g., '0,120;0,121')")
            sys.exit(1)

    if not states_to_sum:
        logger.warning("No states matched your criteria. No output will be generated.")
        return

    logger.info(f"Commencing FFT sum for {len(states_to_sum)} states.")

    # 3. Sum the densities
    combined_data = None
    
    for i, (k, b) in enumerate(states_to_sum):
        logger.debug(f"Processing state {i+1}/{len(states_to_sum)}: K-point {k}, Band {b}")
        
        try:
            # Reconstructs partial charge density from plane-wave coefficients
            st_chgcar = wfn.get_parchg(pos, kpoint=k, band=b, spin=args.spin, scale=args.scale)
            
            if combined_data is None:
                combined_data = st_chgcar.data['total']
            else:
                combined_data += st_chgcar.data['total']
        except Exception as e:
            logger.error(f"Error processing K={k}, B={b}: {e}")
            continue

    # 4. Finalize and Write
    if combined_data is not None:
        logger.info(f"Writing summed charge density to {args.out}...")
        try:
            final_chgcar = Chgcar(pos.structure, data={'total': combined_data})
            final_chgcar.write_file(args.out)
            logger.info("Task completed successfully.")
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
    else:
        logger.error("Summation failed; no data to write.")

if __name__ == "__main__":
    main()
