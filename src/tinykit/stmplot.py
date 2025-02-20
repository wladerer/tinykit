import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Chgcar
import argparse

def load_charge_density(filename):
    """Load charge density from a PARCHG or CHGCAR file."""
    parchg = Chgcar.from_file(filename)
    charge_density = parchg.data["total"]
    lattice = parchg.structure.lattice
    return charge_density, lattice

def calculate_z_coordinates(lattice, grid_shape):
    """Calculate the z-coordinates for the charge density grid."""
    c_axis_length = lattice.c
    z_coords = (c_axis_length / grid_shape[2]) * np.arange(grid_shape[2])
    return z_coords

def simulate_constant_current_image(charge_density, z_coords, target_current):
    """Simulate a constant current STM image."""
    grid_shape = charge_density.shape
    dz = z_coords[1] - z_coords[0]
    constant_current_image = np.zeros((grid_shape[0], grid_shape[1]))

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            cumulative_density = 0
            for k, z in enumerate(z_coords):
                cumulative_density += charge_density[i, j, k] * dz
                if cumulative_density >= target_current:
                    constant_current_image[i, j] = z
                    break

    return constant_current_image

def normalize_image(image):
    """Normalize the image data."""
    return image - image.min()

def tile_image(image, n_tiles):
    """Tile an STM image into an NxN periodic supercell."""
    tiled_image = np.tile(image, (n_tiles, n_tiles))
    return tiled_image

def plot_stm_image(image, output_file=None, title="STM Image", cmap="viridis"):
    """Plot the STM image and save to file if specified."""
    plt.imshow(image, cmap=cmap, origin="lower")
    plt.colorbar(label="Tip Height (Å)")
    plt.title(title)
    plt.xlabel("x-axis (grid points)")
    plt.ylabel("y-axis (grid points)")
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()

def main(args=None):
    """Main function to process arguments and run the simulation."""
    parser = argparse.ArgumentParser(
        description="Simulate constant current STM images from VASP PARCHG/CHGCAR files."
    )
    parser.add_argument(
        "filename", type=str,
        help="Path to the PARCHG or CHGCAR file."
    )
    parser.add_argument(
        "-c","--current", type=float, default=0.001,
        help="Target tunneling current (arbitrary units)."
    )
    parser.add_argument(
        "--tiles", type=int, default=1,
        help="Number of tiles for periodic NxN STM image (default: 1)."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to save the STM image (default: display on screen)."
    )

    if args is None:
        args = parser.parse_args()

    charge_density, lattice = load_charge_density(args.filename)
    z_coords = calculate_z_coordinates(lattice, charge_density.shape)
    stm_image = simulate_constant_current_image(charge_density, z_coords, args.current)
    normalized_image = normalize_image(stm_image)

    if args.tiles > 1:
        tiled_image = tile_image(normalized_image, args.tiles)
        plot_stm_image(
            tiled_image,
            output_file=args.output,
            title=f"Tiled {args.tiles}x{args.tiles} STM Image"
        )
    else:
        plot_stm_image(
            normalized_image,
            output_file=args.output_file,
            title="Constant Current STM Image"
        )

if __name__ == "__main__":
    main()

