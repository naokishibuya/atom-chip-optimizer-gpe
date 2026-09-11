from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Hessian(NamedTuple):
    eigenvalues : jnp.ndarray
    eigenvectors: jnp.ndarray


def compute_hessian(function: Callable[[jnp.ndarray], float], position_mm: jnp.ndarray) -> Hessian:
    H = jax.hessian(function)(position_mm)
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)
    return Hessian(eigenvalues=eigenvalues, eigenvectors=eigenvectors)
