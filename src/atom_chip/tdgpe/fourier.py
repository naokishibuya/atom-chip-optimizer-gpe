"""Fourier-space operations on ψ (SI)."""
import jax
import jax.numpy as jnp

from .box import BoxGrid


def shift_box(
    grid: BoxGrid,
    *,
    psi: jnp.ndarray,        # (nx, ny, nz) wavefunction on the grid: complex values
    delta_r_m: jnp.ndarray,  # Δr = (Δx, Δy, Δz) [m]
) -> jnp.ndarray:
    """Shift the box by +Δr: ψ_new(r) = ψ_old(r + Δr) in box-local coords."""
    # k·Δr per grid cell: contract the vector component i, keep the grid (xyz)
    phase = jnp.exp(1j * jnp.einsum("xyzi,i->xyz", grid.k, delta_r_m))
    return jnp.fft.ifftn(phase * jnp.fft.fftn(psi))


# fmt: off
def evolve_psi(
    grid           : BoxGrid,
    *,
    psi            : jnp.ndarray,  # (nx, ny, nz) wavefunction on the grid: complex values
    potential_coeff: jnp.ndarray,  # (nx, ny, nz) potential rate U/ℏ [rad/s]
    dt_s           : float,        # time step [s]
    n_sub          : int,          # number of real-time substeps per step
    nonlinear_coeff: float,        # mean-field coefficient g/ℏ [m³/s] (0.0 for a linear run)
    kin_coeff      : float,        # kinetic coefficient ħ/2m [m²/s]
) -> jnp.ndarray:
# fmt: on
    """Advance ψ by `n_sub` real-time (unitary) Strang substeps under a frozen potential."""
    half_dt = dt_s / 2.0
    k2 = jnp.sum(grid.k ** 2, axis=-1)
    kin_half = jnp.exp(-1j * kin_coeff * k2 * half_dt)   # K½: kin_coeff·k² is the kinetic rate, half_dt the split step
    kin_full = kin_half * kin_half                 # K: the two boundary half-kicks merged into one interior kick

    def kinetic_half_step(psi: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.ifftn(jnp.fft.fftn(psi) * kin_half)

    def kinetic_step(psi: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.ifftn(jnp.fft.fftn(psi) * kin_full)

    def potential_step(psi: jnp.ndarray) -> jnp.ndarray:
        U_eff = potential_coeff + nonlinear_coeff * jnp.abs(psi) ** 2
        return psi * jnp.exp(-1j * U_eff * dt_s)

    def propagate(psi: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
        return kinetic_step(potential_step(psi)), None

    psi = kinetic_half_step(psi)                                  # leading K½
    psi, _ = jax.lax.scan(propagate, psi, None, length=n_sub - 1)  # interior (P · K)
    return kinetic_half_step(potential_step(psi))                # final P · K½


# fmt: off
def relax_psi(
    grid           : BoxGrid,
    *,
    potential_coeff: jnp.ndarray,  # (nx, ny, nz) potential rate U/ℏ [rad/s]
    dt_s           : float,        # imaginary time step [s]
    n_iters        : int,          # number of imaginary-time iterations to perform
    nonlinear_coeff: float,        # mean-field coefficient g/ℏ [m³/s] (0.0 for a linear run)
    kin_coeff      : float,        # kinetic coefficient ħ/2m [m²/s]
    sigma_init_m   : jnp.ndarray,  # (3,) Gaussian seed width per axis [m]
) -> jnp.ndarray:
# fmt: on
    """Imag-time relax to the (G)PE ground state from a Gaussian guess (seeded at `sigma_init_m`)."""
    psi = _normalize(_gaussian_state(grid.r_local_m, sigma_init_m), grid.dV_m3)

    half_dt = dt_s / 2.0
    k2 = jnp.sum(grid.k ** 2, axis=-1)
    kin_half = jnp.exp(-kin_coeff * k2 * half_dt).astype(jnp.complex128)  # K½ in imag time: a real decay, not a phase

    def kinetic_half_step(psi: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.ifftn(jnp.fft.fftn(psi) * kin_half)

    def potential_step(psi: jnp.ndarray) -> jnp.ndarray:
        U_eff = potential_coeff + nonlinear_coeff * jnp.abs(psi) ** 2
        return psi * jnp.exp(-U_eff * dt_s)

    # Renormalized every iteration, so no interior fusion: each relax is a plain K½·P·K½ then rescale.
    def relax(psi: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
        psi = kinetic_half_step(psi)
        psi = potential_step(psi)
        psi = kinetic_half_step(psi)
        return _normalize(psi, grid.dV_m3), None

    psi_T, _ = jax.lax.scan(relax, psi, None, length=n_iters)
    return psi_T


# fmt: off
def _gaussian_state(
    r_local_m  : jnp.ndarray,  # (nx, ny, nz, 3) box-local positions [m]
    sigma_xyz_m: jnp.ndarray,  # (3,) widths of the Gaussian along x, y, z [m]
) -> jnp.ndarray:
# fmt: on
    """Centered Gaussian ψ(r) = exp(-r²/(2σ²)), unnormalized."""
    sigma = jnp.asarray(sigma_xyz_m, dtype=jnp.float64)
    quad = jnp.sum((r_local_m / sigma) ** 2, axis=-1)
    return jnp.exp(-0.5 * quad).astype(jnp.complex128)


# fmt: off
def _normalize(
    psi: jnp.ndarray,  # (nx, ny, nz) wavefunction on the grid: complex values
    dV_m3: float,      # cell volume (m³)
) -> jnp.ndarray:
# fmt: on
    norm2 = jnp.sum(jnp.abs(psi) ** 2) * dV_m3
    return psi / jnp.sqrt(norm2)
