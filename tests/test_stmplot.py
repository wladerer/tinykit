"""STM helpers: contrast limits, oblique tiling, and the vectorized integral."""
import numpy as np

from tinykit.stmplot import (
    contrast_limits, inplane_vectors, simulate_constant_current_image, tiled_meshgrid,
)


def _reference_constant_current(charge_density, z_coords, target):
    """The original triple-loop implementation, kept here as the oracle."""
    nx, ny, nz = charge_density.shape
    dz = z_coords[1] - z_coords[0]
    image = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            acc = 0.0
            for k, z in enumerate(z_coords):
                acc += charge_density[i, j, k] * dz
                if acc >= target:
                    image[i, j] = z
                    break
    return image


def test_vectorized_matches_reference():
    rng = np.random.default_rng(0)
    charge = rng.random((4, 5, 12))
    z = np.linspace(0.0, 6.0, 12)
    target = 1.0
    fast = simulate_constant_current_image(charge, z, target)
    slow = _reference_constant_current(charge, z, target)
    assert np.allclose(fast, slow)


def test_never_reached_columns_are_zero():
    charge = np.zeros((2, 2, 5))  # integral never reaches the target
    z = np.linspace(0.0, 4.0, 5)
    out = simulate_constant_current_image(charge, z, target_current=10.0)
    assert np.all(out == 0.0)


def test_contrast_limits_percentile_and_override():
    img = np.arange(100, dtype=float)
    lo, hi = contrast_limits(img, clip=(0, 100))
    assert lo == 0 and hi == 99
    lo, hi = contrast_limits(img, clip=(2, 98), vmin=-5, vmax=5)
    assert lo == -5 and hi == 5


def test_tiled_meshgrid_shape_and_obliqueness():
    # hexagonal-ish in-plane vectors -> sheared (non-axis-aligned) grid
    a, b = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
    X, Y = tiled_meshgrid(a, b, shape=(2, 3), n_tiles=2)
    assert X.shape == (4, 6) and Y.shape == (4, 6)
    # b[0] != 0 means X varies along a row of constant a-index -> sheared (oblique)
    assert not np.allclose(X[0, :], X[0, 0])
