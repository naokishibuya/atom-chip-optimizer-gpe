"""The box-local position grid and Fourier wavevectors, in SI metres."""
import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import tree_util


class BoxGrid(NamedTuple):
    # fmt: off
    r_local_m    : jnp.ndarray                 # (nx, ny, nz, 3) box-local positions [m]
    k            : jnp.ndarray                 # (nx, ny, nz, 3) Fourier wavevectors [rad/m]
    dx_m         : tuple[float, float, float]  # grid spacing per axis [m]
    half_widths_m: tuple[float, float, float]  # box half-widths per axis [m]
    # fmt: on

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape (nx, ny, nz)."""
        return self.r_local_m.shape[:3]

    @property
    def dV_m3(self) -> float:
        """Cell volume [m³]."""
        return self.dx_m[0] * self.dx_m[1] * self.dx_m[2]

    @property
    def k_nyquist(self) -> tuple[float, float, float]:
        """Per-axis Nyquist wavenumber π/dx [rad/m]."""
        return tuple(math.pi / dx for dx in self.dx_m)


# Register BoxGrid as a PyTree node so it can be passed into JIT-compiled functions.
# Dynamic children are r_local and k; static metadata is (dx, half_widths).
tree_util.register_pytree_node(
    BoxGrid,
    lambda grid: ((grid.r_local_m, grid.k), (grid.dx_m, grid.half_widths_m)),
    lambda aux_data, children: BoxGrid(children[0], children[1], aux_data[0], aux_data[1]),
)


def build_box(
    half_widths_m: tuple[float, float, float],
    n_cells: tuple[int, int, int],
) -> BoxGrid:
    """Build the box-local grid (centered at 0) in SI metres.

    The grid is in metres; the lab-frame offset is the caller's job, applied per-block via box_center_mm.
    """
    axes_r, axes_k, dxs = [], [], []
    for hw, n in zip(half_widths_m, n_cells, strict=True):
        dx = 2.0 * hw / n
        dxs.append(dx)
        axes_r.append((jnp.arange(n) - n // 2) * dx)
        axes_k.append(2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx))

    # (nx, ny, nz) → (nx, ny, nz, 3): r_local[x, y, z] = [x-pos, y-pos, z-pos] in box-local coords
    # ij indexing: axis 0 runs along x, axis 1 runs along y, axis 2 runs along z
    Rx, Ry, Rz = jnp.meshgrid(axes_r[0], axes_r[1], axes_r[2], indexing="ij")
    r_local_m = jnp.stack([Rx, Ry, Rz], axis=-1).astype(jnp.float64)

    # momentum-space grid dual to the position grid (same shape, fftfreq ordering)
    # (nx, ny, nz) → (nx, ny, nz, 3): k[x, y, z] = [k_x, k_y, k_z]
    Kx, Ky, Kz = jnp.meshgrid(axes_k[0], axes_k[1], axes_k[2], indexing="ij")
    k = jnp.stack([Kx, Ky, Kz], axis=-1).astype(jnp.float64)

    return BoxGrid(r_local_m, k, tuple(dxs), tuple(float(hw) for hw in half_widths_m))


# Grid-sizing policy (tdgpe's call, not physics): how finely to resolve the healing length, and how
# much of the cloud's surface tail to keep inside the box. ξ/2 resolution held gentle T (ξ/0.7 blew
# up); the tail margin is set by a target leaked-mass tolerance, not a blanket factor. Public because
# `chip eval` exposes them as --cells-per-healing-length / --boundary-tol (these are the defaults).
CELLS_PER_HEALING_LENGTH = 2.0  # dx = ξ_min / this
BOUNDARY_TOL = 1e-6             # target |ψ|² fraction beyond the box; sets the ξ-scaled surface-tail margin


def box_shift_margin(r_mins_mm: np.ndarray) -> tuple[float, float, float]:
    """Maximum adjacent trap-minimum displacement per axis [m]."""
    r = np.asarray(r_mins_mm, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"r_mins_mm must have shape (n_nodes, 3), got {r.shape}")
    if r.shape[0] < 2:
        return (0.0, 0.0, 0.0)
    return tuple(np.abs(np.diff(r, axis=0)).max(axis=0) * 1e-3)


def tail_margin_m(xi_ref_m: float, boundary_tol: float) -> float:
    """Surface-tail margin from the leaked-mass tolerance [m]."""
    return (xi_ref_m / 2.0) * math.log(1.0 / boundary_tol)


def size_box(
    extent_m: tuple[float, float, float],
    slosh_m: tuple[float, float, float],
    xi_ref_m: float,
    xi_min_m: float,
    *,
    shift_margin_m: tuple[float, float, float],
    cells_per_xi: float = CELLS_PER_HEALING_LENGTH,
    boundary_tol: float = BOUNDARY_TOL,
) -> BoxGrid:
    """Size the grid from cloud extent, slosh, box-shift, and surface-tail margins."""
    tail = tail_margin_m(xi_ref_m, boundary_tol)
    box_half = tuple(e + s + x + tail
                     for e, s, x in zip(extent_m, slosh_m, shift_margin_m, strict=True))
    dx = xi_min_m / cells_per_xi
    n_cells = tuple(next_fft_friendly(math.ceil(2.0 * hw / dx)) for hw in box_half)
    return build_box(box_half, n_cells)


def is_fft_friendly(n: int) -> bool:
    """Whether n has only factors 2, 3, 5, and 7."""
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def next_fft_friendly(n: int) -> int:
    """Smallest 7-smooth integer >= n, used for FFT efficiency."""
    while not is_fft_friendly(n):
        n += 1
    return n
