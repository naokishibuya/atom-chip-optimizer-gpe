import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..field import WireSegments
from ..schedule import AtomChip, Minimum


def plot_layout_3d(
    atom_chip: AtomChip,
    minimum: Minimum,  # unused; kept for visualizer dispatch signature
    title: str = "Atom Chip Layout",
    size: tuple[int, int] = (7, 6),
    compute_zorder: bool = True,
    azim: float = 0.0,
    elev: float = 0.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    tick: float | list[float] | None = None,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    if fig is None:
        fig = plt.figure(figsize=size)
    else:
        fig.clear()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=compute_zorder)

    # z_order works only if compute_zorder is False
    for z_order, group in enumerate(reversed(atom_chip.wire_layout.values())):
        for lw in group:
            _plot_wire(ax, lw.segments, lw.material, lw.current_A, z_order)

    # set title, label, and limits
    ax.set_title(title + f" ({atom_chip.name})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # set limits if not provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    # Add dashed reference lines along axes
    ax.plot(ax.get_xlim(), [0, 0], [0, 0], color="gray", linestyle="--", linewidth=1, alpha=0.3)
    ax.plot([0, 0], ax.get_ylim(), [0, 0], color="gray", linestyle="--", linewidth=1, alpha=0.3)
    ax.plot([0, 0], [0, 0], ax.get_zlim(), color="gray", linestyle="--", linewidth=1, alpha=0.3)

    # Set ticks
    if tick is not None:
        if isinstance(tick, (int, float)):
            tick = [tick] * 3
        for axis, spacing in zip([ax.xaxis, ax.yaxis, ax.zaxis], tick, strict=True):
            axis.set_major_locator(plt.MultipleLocator(spacing))

    # set view angle and aspect ratio
    ax.set_box_aspect((np.ptp(ax.get_xlim3d()), np.ptp(ax.get_ylim3d()), np.ptp(ax.get_zlim3d())))
    ax.view_init(elev=elev, azim=azim)
    return fig


def _plot_wire(ax: plt.Axes, wire: WireSegments, material: str, current: float, z_order: int):
    color, alpha = _get_material_color(material, current, wire)
    for corners in _wire_vertices(wire):
        faces = [
            [corners[j] for j in [0, 1, 2, 3]],
            [corners[j] for j in [4, 5, 6, 7]],
            [corners[j] for j in [0, 1, 5, 4]],
            [corners[j] for j in [2, 3, 7, 6]],
            [corners[j] for j in [0, 3, 7, 4]],
            [corners[j] for j in [1, 2, 6, 5]],
        ]
        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=color,
                edgecolor="k",
                alpha=alpha,
                linewidth=0.1,
                axlim_clip=True,
                zorder=z_order,
            )
        )


def _wire_vertices(wire: WireSegments) -> jnp.ndarray:
    """Compute the 8 corners of each rectangular segment in a wire. Returns (N, 8, 3)."""
    starts, ends, widths, heights = wire.starts_mm, wire.ends_mm, wire.widths_mm, wire.heights_mm
    vectors = ends - starts
    lengths = jnp.linalg.norm(vectors, axis=1)

    # fmt: off
    offsets = jnp.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ])[jnp.newaxis, ...]  # (1, 8, 3)
    halves = jnp.array([lengths / 2, widths / 2, heights / 2]).T[:, jnp.newaxis, :]  # (N, 1, 3)
    # fmt: on

    vertices = offsets * halves  # (N, 8, 3)

    alpha = jnp.arctan2(vectors[:, 1], vectors[:, 0])
    beta = jnp.arcsin(vectors[:, 2] / lengths)
    cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
    cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
    zeros = jnp.zeros_like(alpha)
    ones = jnp.ones_like(alpha)

    rot_y = jnp.array([
        [ cos_b, zeros, sin_b],
        [ zeros, ones,  zeros],
        [-sin_b, zeros, cos_b],
    ])
    rot_z = jnp.array([
        [ cos_a,  sin_a, zeros],
        [-sin_a,  cos_a, zeros],
        [ zeros,  zeros, ones ],
    ])
    rot = rot_z.T @ rot_y.T  # (N, 3, 3)
    vertices = jnp.einsum("nij,nvj->nvi", rot, vertices)

    centers = (starts + ends) / 2
    return vertices + centers[:, jnp.newaxis, :]


def _get_material_color(material: str, current: float, wire: WireSegments):
    colors = {
        "copper": [
            [[198, 117, 26], 0.1],  # zero current
            [[255, 112, 66], 0.6],  # positive flow
            [[66, 112, 255], 0.6],  # negative flow
        ],
        "gold": [
            [[198, 117, 26], 0.1],
            [[255, 215, 0], 0.6],
            [[0, 215, 255], 0.6],
        ],
    }[material]
    index = 0
    if current != 0:
        # Flow direction is sign of the wire's displacement along its dominant axis
        vec = wire.ends_mm[0] - wire.starts_mm[0]
        flow_sign = float(jnp.sign(vec[int(jnp.argmax(jnp.abs(vec)))]))
        index = 1 if current * flow_sign > 0 else 2
    color, alpha = colors[index]
    return np.array(color) / 255, alpha
