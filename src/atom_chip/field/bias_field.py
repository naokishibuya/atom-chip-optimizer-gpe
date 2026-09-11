from typing import NamedTuple

import jax.numpy as jnp


class BiasField(NamedTuple):
    # fmt: off
    coil_factors_G_per_A: jnp.ndarray  # shape (3,)
    coil_currents_A     : jnp.ndarray  # shape (3,)
    stray_field_G       : jnp.ndarray  # shape (3,)
    # fmt: on

    def get_fields(self, points_mm: jnp.ndarray) -> jnp.ndarray:
        bias = self.coil_factors_G_per_A * self.coil_currents_A + self.stray_field_G
        return jnp.broadcast_to(bias, (points_mm.shape[0], 3))

    def to_dict(self) -> dict:
        # fmt: off
        return {
            "coil_factors_G_per_A": self.coil_factors_G_per_A.tolist(),
            "coil_currents_A"     : self.coil_currents_A.tolist(),
            "stray_field_G"       : self.stray_field_G.tolist(),
        }
        # fmt: on

    @classmethod
    def from_dict(cls, data: dict) -> "BiasField":
        # fmt: off
        return cls(
            coil_factors_G_per_A = jnp.array(data["coil_factors_G_per_A"], dtype=jnp.float64),
            coil_currents_A      = jnp.array(data["coil_currents_A"]     , dtype=jnp.float64),
            stray_field_G        = jnp.array(data["stray_field_G"]       , dtype=jnp.float64),
        )
        # fmt: on
