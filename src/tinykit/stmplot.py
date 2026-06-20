"""Simulate constant-current STM images from VASP PARCHG/CHGCAR files.

Plots in real Angstrom coordinates with correct periodic tiling of oblique
(e.g. hexagonal) cells, percentile-based contrast, and configurable colormap.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pymatgen.io.vasp.outputs import Chgcar


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
    """Simulate a constant current STM image (tip height map)."""
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
    """Shift the image so its minimum is zero."""
    return image - image.min()


def inplane_vectors(lattice):
    """Return the two in-plane (a, b) lattice vectors as 2D arrays."""
    a = np.array(lattice.matrix[0][:2])
    b = np.array(lattice.matrix[1][:2])
    return a, b


def tiled_meshgrid(a, b, shape, n_tiles):
    """Real-space X/Y meshgrid covering an n_tiles x n_tiles block of cells.

    Fractional indices are endpoint-exclusive (periodic), so replicating the
    grid and mapping through the real lattice vectors fills oblique cells with
    no whitespace and no overlap.
    """
    n0, n1 = shape
    f0 = np.arange(n0 * n_tiles) / n0
    f1 = np.arange(n1 * n_tiles) / n1
    F0, F1 = np.meshgrid(f0, f1, indexing="ij")
    X = F0 * a[0] + F1 * b[0]
    Y = F0 * a[1] + F1 * b[1]
    return X, Y


def contrast_limits(image, clip, vmin=None, vmax=None):
    """Resolve (vmin, vmax) from percentile clip, overridden by explicit values."""
    lo, hi = np.percentile(image, clip)
    if vmin is not None:
        lo = vmin
    if vmax is not None:
        hi = vmax
    return lo, hi


def plot_stm_image(X, Y, image, vmin, vmax, cmap, title, output_file=None, dpi=300):
    """Plot the STM image in real coordinates and save or display it."""
    fig, ax = plt.subplots(figsize=(6, 5))
    mesh = ax.pcolormesh(X, Y, image, shading="nearest", cmap=cmap,
                         vmin=vmin, vmax=vmax, rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax, label="Tip height (Å)")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_aspect("equal")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=dpi)
    else:
        plt.show()


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(
        description="Simulate constant current STM images from VASP PARCHG/CHGCAR files."
    )
    parser.add_argument("filename", type=str,
                        help="Path to the PARCHG or CHGCAR file.")
    parser.add_argument("-c", "--current", type=float, default=0.001,
                        help="Target tunneling current (arbitrary units).")
    parser.add_argument("--tiles", type=int, default=1,
                        help="Replicate the cell into an NxN periodic image (default: 1).")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to save the STM image (default: display on screen).")
    parser.add_argument("--cmap", default="inferno",
                        help="Matplotlib colormap (default: inferno).")
    parser.add_argument("--clip", type=float, nargs=2, default=[2, 98], metavar=("LO", "HI"),
                        help="Percentile range for contrast (default: 2 98).")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Explicit lower contrast limit (overrides --clip low).")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Explicit upper contrast limit (overrides --clip high).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution of the saved image (default: 300).")
    return parser


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = build_parser().parse_args(args)

    charge_density, lattice = load_charge_density(args.filename)
    z_coords = calculate_z_coordinates(lattice, charge_density.shape)
    image = normalize_image(
        simulate_constant_current_image(charge_density, z_coords, args.current)
    )

    a, b = inplane_vectors(lattice)
    X, Y = tiled_meshgrid(a, b, image.shape, args.tiles)
    tiled = np.tile(image, (args.tiles, args.tiles))

    # Contrast comes from the true (untiled) cell data.
    vmin, vmax = contrast_limits(image, args.clip, args.vmin, args.vmax)

    title = "Constant-Current STM Image"
    if args.tiles > 1:
        title = f"Tiled {args.tiles}x{args.tiles} STM Image"

    plot_stm_image(X, Y, tiled, vmin, vmax, args.cmap, title,
                   output_file=args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
