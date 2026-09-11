import itertools

import jax
import jax.numpy as jnp

_CORNERS = tuple(itertools.product((-1, +1), repeat=3))


# Field evaluation holds the following intermediates:
# 1) 5 arrays of shape (M, N, 3):
#    - translated_points,
#    - rotated_points,
#    - B_rotated
#    - the back-rotated B,
#    - the final B_scaled.
#    (All of this is equivalent to 15 (M, N) matrices).
# 2) 6 arrays of shape (M, N):
#    - the extracted coordinates x, y, z
#    - the field components BX, BY, BZ
#
# Each (M, N) float64 matrix costs M (points) * N (segments) * 8 bytes.
#
# Not all of them live concurrently but conservatively, we have 21 explicit (M, N) arrays,
# plus a handful of JAX scratchpads during the math loops. This total is rounded up to
# the nearest power of 2 (2^5 = 32).
#
# So, _PEAK_FACTOR = 32 provides a safe memory ceiling of 32 * M * N * 8 bytes.
_PEAK_FACTOR = 32


# This limits batching to 50% of the total memory budget.
# It's a safe heuristic, not a strict mathematical guarantee. Reserving half the memory
# avoids the need for manual byte-tracking of the input arrays (starts_mm, midpoints, etc.)
# and JAX's unpredictable internal memory allocations.
#
# Note: this assumes the (M, N) intermediate matrices dominate the memory footprint, which
# holds true unless the number of wire segments (N) is exceptionally massive.
_BUDGET_FRACTION = 0.5


def _detect_field_mem_budget() -> int | None:
    """GPU memory available to JAX [bytes]; None on CPU or when it can't be determined."""
    device = jax.devices()[0]
    if device.platform == "cpu":
        return None
    try:
        stats = device.memory_stats()
    except (RuntimeError, NotImplementedError):
        return None
    if stats and stats.get("bytes_limit"):
        return int(stats["bytes_limit"])
    return None


_MEM_BUDGET_BYTES = _detect_field_mem_budget()


# fmt: off
def batched_biot_savart_rectangular(
    points_mm : jnp.ndarray,  # (M, 3) Evaluation points [mm]
    starts_mm : jnp.ndarray,  # (N, 3)
    ends_mm   : jnp.ndarray,  # (N, 3)
    widths_mm : jnp.ndarray,  # (N,)
    heights_mm: jnp.ndarray,  # (N,)
    currents_A: jnp.ndarray,  # (N,)
) -> jnp.ndarray:
# fmt: on
    """Memory-bounded `biot_savart_rectangular`: identical result, evaluated in point-batches."""
    n_points, n_segments = points_mm.shape[0], starts_mm.shape[0]

    if _MEM_BUDGET_BYTES is None:
        batch_size = n_points
    else:
        per_point_bytes = _PEAK_FACTOR * n_segments * 8
        batch_size = max(1, min(n_points, int(_BUDGET_FRACTION * _MEM_BUDGET_BYTES / per_point_bytes)))

    if batch_size >= n_points:
        return biot_savart_rectangular(points_mm, starts_mm, ends_mm, widths_mm, heights_mm, currents_A)

    n_batches = -(-n_points // batch_size)  # integer ceil
    pad = n_batches * batch_size - n_points
    padded = jnp.concatenate([points_mm, jnp.zeros((pad, 3), points_mm.dtype)]) if pad else points_mm
    fields = jax.lax.map(
        lambda p: biot_savart_rectangular(p, starts_mm, ends_mm, widths_mm, heights_mm, currents_A),
        padded.reshape(n_batches, batch_size, 3),
    )
    return fields.reshape(-1, 3)[:n_points]


# fmt: off
def biot_savart_rectangular(
    points_mm : jnp.ndarray,  # (M, 3) Evaluation points in 3D space [mm]
    starts_mm : jnp.ndarray,  # (N, 3) Start points of wire segments [mm]
    ends_mm   : jnp.ndarray,  # (N, 3) End points of wire segments [mm]
    widths_mm : jnp.ndarray,  # (N,  ) Widths of wire segments [mm]
    heights_mm: jnp.ndarray,  # (N,  ) Heights of wire segments [mm]
    currents_A: jnp.ndarray,  # (N,  ) Currents through wire segments [A]
) -> jnp.ndarray:
# fmt: on
    """Magnetic field from multiple rectangular conductor segments at multiple points.

    Returns (M, 3): the total field at each evaluation point [G].
    """
    # fmt: off
    points   = jnp.float64(points_mm)
    starts   = jnp.float64(starts_mm)
    ends     = jnp.float64(ends_mm)
    widths   = jnp.float64(widths_mm)
    heights  = jnp.float64(heights_mm)
    currents = jnp.float64(currents_A)
    # fmt: on

    # Compute vectors and lengths
    vectors = ends - starts  # (N, 3)
    lengths = jnp.linalg.norm(vectors, axis=1)  # (N,)

    # Translate points
    midpoints = (starts + ends) / 2  # (N, 3)
    translated_points = points[:, None, :] - midpoints[None, :, :]  # (M, N, 3)

    # Rotate points
    rotation_matrices = _rotation_matrix(vectors)  # (N, 3, 3)
    rotated_points = jnp.einsum("nij,mnj->mni", rotation_matrices, translated_points)  # (M, N, 3)

    # Compute field components
    x, y, z = rotated_points[:, :, 0], rotated_points[:, :, 1], rotated_points[:, :, 2]  # (M, N)
    L, W, H = lengths / 2, widths / 2, heights / 2  # (N,)

    BY = _compute_BY(x, y, z, L, W, H)  # (M, N)
    BZ = _compute_BZ(x, y, z, L, W, H)  # (M, N)
    BX = jnp.zeros_like(BY)  # (M, N): no field along the segment axis

    # Rotate field components back
    B_rotated = jnp.stack([BX, BY, BZ], axis=-1)  # (M, N, 3)
    B = jnp.einsum("nij,mnj->mni", jnp.transpose(rotation_matrices, (0, 2, 1)), B_rotated)  # (M, N, 3) back-rotated

    # Scale by current and conductor cross-section
    B_scaled = B * currents[None, :, None] / (widths * heights)[None, :, None]  # (M, N, 3) per cross-section

    # Sum contributions from all conductors
    return jnp.sum(B_scaled, axis=1)  # (M, 3)


def _rotation_matrix(vectors: jnp.ndarray) -> jnp.ndarray:
    """Per-segment rotation taking the segment direction onto the x-axis."""
    vectors = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)
    alpha = jnp.arctan2(vectors[:, 1], vectors[:, 0])
    beta = jnp.arctan2(vectors[:, 2], jnp.linalg.norm(vectors[:, :2], axis=1))

    cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
    cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
    zeros = jnp.zeros_like(alpha, dtype=jnp.float64)

    return jnp.array([
        [cos_a * cos_b, sin_a * cos_b, sin_b],
        [-sin_a, cos_a, zeros],
        [-cos_a * sin_b, -sin_a * sin_b, cos_b]
    ]).transpose(2, 0, 1)


def _compute_BY(
    x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray,
    L: jnp.ndarray, W: jnp.ndarray, H: jnp.ndarray,
) -> jnp.ndarray:
    result = 0.0
    for sx, sy, sz in _CORNERS:
        ax = x + sx * L
        ay = y + sy * W
        az = z + sz * H
        r = jnp.sqrt(ax**2 + ay**2 + az**2)
        result = result + sx * sy * sz * (
            ax * jnp.log(ay + r)
            + ay * jnp.log(ax + r)
            - az * jnp.arctan(ax * ay / (az * r))
        )
    return result


def _compute_BZ(
    x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray,
    L: jnp.ndarray, W: jnp.ndarray, H: jnp.ndarray,
) -> jnp.ndarray:
    result = 0.0
    for sx, sy, sz in _CORNERS:
        ax = x + sx * L
        ay = y + sy * W
        az = z + sz * H
        r = jnp.sqrt(ax**2 + ay**2 + az**2)
        result = result + sx * sy * sz * (
            -ax * jnp.log(az + r)
            - az * jnp.log(ax + r)
            + ay * jnp.arctan(ax * az / (ay * r))
        )
    return result
