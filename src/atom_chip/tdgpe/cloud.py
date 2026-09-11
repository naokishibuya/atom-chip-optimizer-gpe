"""Measure cloud position, size, state overlap, and spatial and spectral occupancy."""
import jax.numpy as jnp

from .box import BoxGrid


def fidelity(
    grid: BoxGrid,
    *,
    psi_a: jnp.ndarray,
    psi_b: jnp.ndarray,
) -> jnp.ndarray:
    """Squared overlap |⟨ψ_a|ψ_b⟩|² of two normalized wavefunctions on the same grid."""
    inner = jnp.sum(jnp.conj(psi_a) * psi_b) * grid.dV_m3
    return jnp.abs(inner) ** 2


def cloud_observables(
    grid: BoxGrid,
    *,
    psi: jnp.ndarray,  # wavefunction on the grid: (nx, ny, nz) complex values
    axes: jnp.ndarray,  # (3, 3) columns are unit directions for measuring RMS widths
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the COM, position covariance, and RMS widths along the supplied axes.

    Requires ∫|ψ|² dV = 1.

    Returns:

    - COM: (3,), in local box coordinates [m].
    - Position covariance: (3, 3) [m²].
    - RMS widths: (3,), along the supplied axes [m].
    """
    rho = jnp.abs(psi) ** 2
    # Average the position vector over the probability density.
    com = jnp.sum(rho[..., None] * grid.r_local_m, axis=(0, 1, 2)) * grid.dV_m3
    delta = grid.r_local_m - com
    # Covariance C_ij = ∫(r_i - COM_i)(r_j - COM_j) ρ dV.
    cov = jnp.einsum("xyzi,xyzj,xyz->ij", delta, delta, rho) * grid.dV_m3

    # Project the covariance onto each direction, then take the square root of its variance.
    sigma = jnp.sqrt(jnp.diag(axes.T @ cov @ axes))
    return com, cov, sigma


def boundary_mass(
    grid: BoxGrid,
    *,
    psi: jnp.ndarray,
    edge_frac: float) -> jnp.ndarray:
    """Probability near the box edges: |r_i| >= edge_frac * half_width_i.

    Requires ∫|ψ|² dV = 1.

    Returns (3,), one probability per grid axis, including both sides.

    - Uses box half-widths, not the largest sampled coordinates.
    - Regions overlap across axes; adding the values can count a cell more than once.
    """
    r = grid.r_local_m                                          # (nx, ny, nz, 3) [m]
    hw = jnp.asarray(grid.half_widths_m)                        # (3,) box half-widths [m]
    density = jnp.abs(psi) ** 2 * grid.dV_m3                    # probability per grid cell
    slabs = [jnp.sum(jnp.where(jnp.abs(r[..., i]) >= edge_frac * hw[i], density, 0.0))
             for i in range(3)]
    return jnp.stack(slabs)                                     # (3,)


def k_tail_mass(
    grid: BoxGrid,
    *,
    psi: jnp.ndarray,
    edge_frac: float) -> jnp.ndarray:
    """Spectral probability with |k_i| >= edge_frac * k_Nyquist_i, per grid axis.

    Returns (3,), one probability per grid axis.

    - Transport uses edge_frac=0.85, selecting the region nearest the cutoff.
    - Measures spectral occupancy, not an aliasing or discretization error bound.
    """
    spectral = jnp.abs(jnp.fft.fftn(psi)) ** 2
    spectral = spectral / jnp.sum(spectral)  # normalize to a probability over k
    k = grid.k                               # (nx, ny, nz, 3) [rad/m]
    k_nyq = grid.k_nyquist                   # (3,) π/dx per axis
    tails = [jnp.sum(jnp.where(jnp.abs(k[..., i]) >= edge_frac * k_nyq[i],
                               spectral, 0.0))
             for i in range(3)]
    return jnp.stack(tails)                                   # (3,)


def k_shell_masses(
    grid: BoxGrid,
    *,
    psi: jnp.ndarray,
    edges_frac: tuple[float, ...]) -> jnp.ndarray:
    """Spectral probability in separate high-wavevector bands, per grid axis.

    Returns (3, len(edges_frac)): one row per axis and one column per band.
    Supply edges_frac in increasing order. Each band includes its lower edge and
    excludes the next edge; the final band includes all remaining represented values.

    Transport uses three bands in |k_i| / k_Nyquist_i:

    - [0.5, 0.62996)
    - [0.62996, 0.79370)
    - [0.79370, 1], including Nyquist where represented.

    Summing these bands gives the probability with |k_i| >= 0.5 * k_Nyquist_i
    for each axis. The logged mshelf is the largest of the three axis sums.
    This measures spectral occupancy, not an error bound.
    """
    spectral = jnp.abs(jnp.fft.fftn(psi)) ** 2
    spectral = spectral / jnp.sum(spectral)  # normalize to a probability over k
    k = grid.k                               # (nx, ny, nz, 3) [rad/m]
    k_nyq = grid.k_nyquist                   # (3,) π/dx per axis
    # First count all probability above each edge; these cumulative regions overlap.
    tails = jnp.stack([
        jnp.stack([jnp.sum(jnp.where(jnp.abs(k[..., i]) >= f * k_nyq[i], spectral, 0.0))
                   for f in edges_frac])
        for i in range(3)])                                   # (3, S) cumulative tail masses
    # Subtract adjacent tails to obtain non-overlapping bands; retain the final tail.
    return tails - jnp.concatenate([tails[:, 1:], jnp.zeros((3, 1))], axis=1)  # (3, S) bands
