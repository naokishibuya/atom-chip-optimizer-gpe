from typing import NamedTuple

import jax.numpy as jnp

from ..field import BiasField, WireSegments
from . import constants
from .atom import Atom
from .hessian import Hessian, compute_hessian
from .search import trap_magnetic_fields, trap_potential_energies


class Frequency(NamedTuple):
    # fmt: off
    omega_hz : jnp.ndarray  # [Hz]
    omega_rad: jnp.ndarray  # [rad/s]
    # fmt: on


class TrapGeometry(NamedTuple):
    # fmt: off
    hessian: Hessian
    trap   : Frequency
    larmor : Frequency
    # fmt: on


class GaussianAnalysis(NamedTuple):
    """Non-interacting (Gaussian-regime) BEC analysis in a harmonic trap."""

    # fmt: off
    total_atoms           : int
    a_ho_m                : float
    omega_ho_rad          : float
    mu_0_J                : float
    a_ho_axes_m           : jnp.ndarray
    critical_temperature_K: float
    # fmt: on


class TFAnalysis(NamedTuple):
    # fmt: off
    condensed_atoms: int
    mu_J           : float
    radii_m        : jnp.ndarray
    # fmt: on


def larmor_frequency(atom: Atom, field_magnitude_G: jnp.ndarray) -> Frequency:
    omega_rad = atom.g_F * constants.mu_B * field_magnitude_G * 1e-4 / constants.hbar
    return Frequency(omega_rad / (2 * jnp.pi), omega_rad)


def trap_frequencies(eigenvalues: jnp.ndarray, mass_kg: float) -> Frequency:
    eigenvalues = eigenvalues * 1e6  # J/mm² -> J/m²
    omega_rad = jnp.sqrt(eigenvalues / mass_kg)
    omega_hz = omega_rad / (2 * jnp.pi)
    return Frequency(omega_hz, omega_rad)


def analyze_field(
    atom: Atom,
    wire_geometry: WireSegments,
    currents_A: jnp.ndarray,
    bias_field: BiasField,
    position_mm: jnp.ndarray,
) -> TrapGeometry:
    """Hessian + trap/Larmor frequencies of |B| at a position. Caller asserts it's the minimum."""
    def B_mag_at(p: jnp.ndarray) -> jnp.ndarray:
        B_mag, _ = trap_magnetic_fields(jnp.atleast_2d(p), wire_geometry, currents_A, bias_field)
        return B_mag[0]

    hessian = compute_hessian(B_mag_at, position_mm)
    eigenvalues = atom.magnetic_moment() * hessian.eigenvalues * 1e-4  # G/mm² → J/mm²
    trap = trap_frequencies(eigenvalues, atom.mass_kg)
    larmor = larmor_frequency(atom, B_mag_at(position_mm))
    return TrapGeometry(hessian=hessian, trap=trap, larmor=larmor)


def analyze_trap(
    atom: Atom,
    wire_geometry: WireSegments,
    currents_A: jnp.ndarray,
    bias_field: BiasField,
    position_mm: jnp.ndarray,
) -> TrapGeometry:
    """Hessian + trap/Larmor frequencies of U at a position. Caller asserts it's the minimum."""
    def U_at(p: jnp.ndarray) -> jnp.ndarray:
        U, _, _ = trap_potential_energies(jnp.atleast_2d(p), atom, wire_geometry, currents_A, bias_field)
        return U[0]

    _, B_mag, _ = trap_potential_energies(
        jnp.atleast_2d(position_mm), atom, wire_geometry, currents_A, bias_field
    )
    hessian = compute_hessian(U_at, position_mm)
    trap = trap_frequencies(hessian.eigenvalues, atom.mass_kg)
    larmor = larmor_frequency(atom, B_mag[0])
    return TrapGeometry(hessian=hessian, trap=trap, larmor=larmor)


def analyze_gaussian(
    trap_frequency: Frequency,
    atom: Atom,
    total_atoms: int,
) -> GaussianAnalysis:
    omega_rad = trap_frequency.omega_rad
    omega_ho_rad = jnp.prod(omega_rad) ** (1 / 3)
    a_ho_m = atom.harmonic_oscillator_length(omega_ho_rad)
    mu_0_J = 0.5 * constants.hbar * jnp.sum(omega_rad)
    a_ho_axes_m = atom.harmonic_oscillator_length(omega_rad)
    critical_temperature_K = 0.94 * constants.hbar / constants.k_B * omega_ho_rad * total_atoms ** (1 / 3)
    return GaussianAnalysis(
        total_atoms=total_atoms,
        a_ho_m=a_ho_m,
        omega_ho_rad=omega_ho_rad,
        mu_0_J=mu_0_J,
        a_ho_axes_m=a_ho_axes_m,
        critical_temperature_K=critical_temperature_K,
    )


def analyze_tf(
    trap_frequency: Frequency,
    atom: Atom,
    condensed_atoms: int,
) -> TFAnalysis:
    mu_J, radii_m = tf_quantities(trap_frequency.omega_rad, atom, condensed_atoms)
    return TFAnalysis(
        condensed_atoms=condensed_atoms,
        mu_J=mu_J,
        radii_m=radii_m,
    )


def tf_quantities(omega_rad: jnp.ndarray, atom: Atom, condensed_atoms: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Thomas-Fermi μ and R_TF from angular ω (rad/s). Broadcasts over leading axes:
    `(3,)` → scalars + `(3,)`; `(T, 3)` → `(T,)` + `(T, 3)`.
    """
    omega_ho_rad = jnp.prod(omega_rad, axis=-1) ** (1 / 3)
    a_ho_m = atom.harmonic_oscillator_length(omega_ho_rad)
    mu_J = 0.5 * constants.hbar * omega_ho_rad * (15 * atom.scattering_length_m * condensed_atoms / a_ho_m) ** (2 / 5)
    radii_m = jnp.sqrt(2 * mu_J[..., None] / atom.mass_kg) / omega_rad
    return mu_J, radii_m
