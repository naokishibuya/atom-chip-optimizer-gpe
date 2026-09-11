import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..potential import constants
from ..schedule import AtomChip, Minimum


def plot_potential_3d(
    atom_chip: AtomChip,
    minimum: Minimum,
    size: tuple[int, int],
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
    z_range: tuple[float, float] | None= None,
    z: float | None = None,
    zlim: tuple[float, float] | None= None,
    fig: plt.Figure | None = None,
    elev: float = 30,
    azim: float = -60,
    zlim_cap: float | None = None
) -> plt.Figure:
    if fig is None:
        fig = plt.figure(figsize=size)
    else:
        fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    if not minimum.found:
        fig.text(0.5, 0.5, "Potential Minimum not found.", ha="center", va="center", fontsize=12)
        return fig

    x_vals = np.linspace(*x_range)
    y_vals = np.linspace(*y_range)
    X, Y = np.meshgrid(x_vals, y_vals)

    if z_range:
        z_vals = np.linspace(z_range[0], z_range[1], z_range[2] + 1)
        initial_z = z if z else minimum.position[2]
        initial_z = max(initial_z, z_vals[0])
        initial_z = min(initial_z, z_vals[-1])
    elif z is not None:
        initial_z = z
        z_vals = [z]
    else:
        initial_z = minimum.position[2]
        z_vals = [initial_z]

    # Initial plot
    surf = _plot_3d_trapping_potential(atom_chip, minimum, ax, X, Y, initial_z, zlim, zlim_cap)

    # Colorbar
    if not zlim_cap:
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=50, label="Energy [μK]", pad=0.1)

    # Slider
    if not zlim_cap and z_range:
        ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
        slider = Slider(
            ax_slider,
            "z [mm]",
            z_vals[0],
            z_vals[-1],
            valinit=initial_z,
            valstep=z_vals,
        )

        def update(val: float) -> None:
            ax.clear()
            _plot_3d_trapping_potential(atom_chip, minimum, ax, X, Y, val, zlim, zlim_cap)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        fig._slider = slider  # Prevent garbage collection

    fig.tight_layout()
    return fig


def _plot_3d_trapping_potential(
    atom_chip: AtomChip,
    minimum: Minimum,
    ax: plt.Axes,
    X: np.ndarray,  # Meshgrid x-coordinates
    Y: np.ndarray,  # Meshgrid y-coordinates
    z: float,  # z-coordinate of the minimum potential energy
    zlim: tuple[float, float],  # z-axis limits for the plot
    zlim_cap: float | None,
) -> Poly3DCollection:
    # Get the energy at a given z-coordinate or the minimum energy point
    z = z if z is not None else minimum.position[2]
    point = np.array([minimum.position[0], minimum.position[1], z])
    V_at_z = atom_chip.get_potentials(point)[0][0]
    points = np.array([[x, y, z] for x, y in zip(X.flatten(), Y.flatten(), strict=True)])
    E, _, _ = atom_chip.get_potentials(points)
    V = E.reshape(X.shape)

    T = constants.joule_to_microKelvin(V)
    T_at_z = constants.joule_to_microKelvin(V_at_z)

    surf = ax.plot_surface(X, Y, T, cmap="jet", edgecolor="none", vmin=zlim[0], vmax=zlim[1])
    levels = np.linspace(zlim[0], zlim[1], 20)
    ax.contour(X, Y, T, levels=levels, cmap="jet", offset=zlim[0])

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if not zlim_cap:
        ax.set_zlabel("Energy [μK]")
    ax.set_title(f"3D Trapping Potential @ z = {z:.4g} mm ({T_at_z:.1f} μK)", pad=0, y=0.99)
    if zlim_cap:
        ax.set_zlim((zlim[0], zlim_cap))
    else:
        ax.set_zlim(zlim)
    return surf
