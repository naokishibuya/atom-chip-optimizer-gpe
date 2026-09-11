import jax

# The whole solver stack assumes float64; must run before any jnp array is created.
jax.config.update("jax_enable_x64", True)

from . import field, potential, schedule, surrogate, tdgpe, visualization

__all__ = ["field", "potential", "schedule", "surrogate", "tdgpe", "visualization"]
