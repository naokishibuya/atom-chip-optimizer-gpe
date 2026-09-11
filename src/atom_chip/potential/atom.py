from typing import NamedTuple

import jax.numpy as jnp
from jax import tree_util

from . import constants


class Atom(NamedTuple):
    name: str  # Species label (aux data — not traced by JIT)
    mass_kg: float  # Mass (kg)
    g_F: float  # Landé g-factor
    m_F: float  # Magnetic quantum number
    scattering_length_m: float  # s-wave scattering length (m)

    def magnetic_moment(self) -> float:
        return self.g_F * self.m_F * constants.mu_B

    def potential_energy(self, field_magnitude_G: jnp.ndarray, z_mm: float) -> jnp.ndarray:
        """
        Zeeman + gravitational potential energy. field magnitude in Gauss, z in mm, result in J.
        Broadcasts over field magnitude: scalar→scalar, (3,)→(3,), (T,3)→(T,3).
        """
        return self.magnetic_moment() * field_magnitude_G * 1e-4 - self.mass_kg * constants.g * z_mm * 1e-3

    def harmonic_oscillator_length(self, omega_rad: jnp.ndarray) -> jnp.ndarray:
        """
        HO length √(ℏ/(m·ω)).
        Broadcasts over omega (angular frequency) in rad: scalar→scalar, (3,)→(3,), (T,3)→(T,3).
        """
        return jnp.sqrt(constants.hbar / (self.mass_kg * omega_rad))


# `name` lives in PyTree aux data so JIT'd functions can take an Atom directly
# without tracing the string. Cost: aux data is part of the JIT cache key, so
# each distinct species name forces one extra JIT compile (one chip → zero cost;
# two species in the same process → 2 compiles instead of 1).
tree_util.register_pytree_node(
    Atom,
    lambda atom: ((atom.mass_kg, atom.g_F, atom.m_F, atom.scattering_length_m), atom.name),
    lambda name, fields: Atom(name, *fields),
)


rb87 = Atom(name="rb87", mass_kg=1.44316e-25, g_F=0.5, m_F=2, scattering_length_m=5.2e-9)
