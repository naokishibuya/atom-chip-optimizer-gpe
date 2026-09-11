import jax.numpy as jnp
import optimistix as optx

from ..field import BiasField, WireSegments
from .atom import Atom
from .minimum import solve_bounded_bfgs


def trap_magnetic_fields(
    points_mm: jnp.ndarray,
    wire_geometry: WireSegments,
    currents_A: jnp.ndarray,
    bias_field: BiasField
) -> tuple[jnp.ndarray, jnp.ndarray]:
    B = bias_field.get_fields(points_mm) + wire_geometry.get_fields(points_mm, currents_A)
    B_mag = jnp.linalg.norm(B, axis=1)
    return B_mag, B


def find_field_minimum(
    wire_geometry: WireSegments,
    currents_A: jnp.ndarray,
    bias_field: BiasField,
    guess_mm: jnp.ndarray,
    half_bounds_mm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, optx.Solution]:
    """Returns (x, val, sol) for the |B| minimum. Caller owns the acceptance policy."""
    lower = guess_mm - half_bounds_mm
    upper = guess_mm + half_bounds_mm

    def B_mag_at(p: jnp.ndarray) -> jnp.ndarray:
        B_mag, _ = trap_magnetic_fields(jnp.atleast_2d(p), wire_geometry, currents_A, bias_field)
        return B_mag[0]

    return solve_bounded_bfgs(B_mag_at, guess_mm, lower, upper)


def trap_potential_energies(
    points_mm: jnp.ndarray,
    atom: Atom,
    wire_geometry: WireSegments,
    currents_A: jnp.ndarray,
    bias_field: BiasField,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B_mag, B = trap_magnetic_fields(points_mm, wire_geometry, currents_A, bias_field)
    z = points_mm[:, 2]
    return atom.potential_energy(B_mag, z), B_mag, B


def find_trap_minimum(
    atom: Atom,
    wire_geometry: WireSegments,
    currents_A: jnp.ndarray,
    bias_field: BiasField,
    guess_mm: jnp.ndarray,
    half_bounds_mm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, optx.Solution]:
    """Returns (x, val, sol) for the trap-potential minimum. Caller owns the acceptance policy."""
    lower = guess_mm - half_bounds_mm
    upper = guess_mm + half_bounds_mm

    def U_at(p: jnp.ndarray) -> jnp.ndarray:
        U, _, _ = trap_potential_energies(jnp.atleast_2d(p), atom, wire_geometry, currents_A, bias_field)
        return U[0]

    return solve_bounded_bfgs(U_at, guess_mm, lower, upper)
