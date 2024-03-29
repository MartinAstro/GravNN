import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.transformations import cart2sph
from GravNN.Trajectories.utils import (
    generate_near_hopper_trajectories,
    generate_near_orbit_trajectories,
)
from GravNN.Visualization.VisualizationBase import VisualizationBase


def plot_orbits_as_violins(trajectories, near_trajectories, color="black"):
    radial_dists = []
    orbit_start_times = []
    t0 = near_trajectories[0].times[0]
    for i in range(0, len(trajectories)):
        x = cart2sph(trajectories[i].positions)
        r = x[:, 0]
        t = near_trajectories[i].times
        radial_dists.append(r / 1000)
        orbit_start_times.append((t[0] - t0) / 86400)

    bodies = plt.violinplot(radial_dists, positions=orbit_start_times, widths=10)
    for pc in bodies["bodies"]:
        pc.set_color(color)
    bodies["cmaxes"].set_color(color)
    bodies["cmins"].set_color(color)
    bodies["cbars"].set_color(color)
    # bodies.set_color('black')


def plot_model_error(df, color, label, sampling_interval):
    df.sort_values(by=["samples"], inplace=True)

    samples = df["samples"].values
    outer_errors = df["outer_avg_error"].values
    inner_errors = df["inner_avg_error"].values
    surface_errors = df["surface_avg_error"].values
    times = samples * sampling_interval / 86400

    plt.semilogy(
        times,
        outer_errors,
        label=f"{label}",
        color=color,
        linestyle="-",
    )
    plt.semilogy(
        times,
        inner_errors,
        label=f"{label}",
        color=color,
        linestyle="--",
    )
    plt.semilogy(
        times,
        surface_errors,
        label=f"{label}",
        color=color,
        linestyle=":",
    )


def main(hoppers, noise):
    os.path.dirname(GravNN.__file__) + "/../"
    vis = VisualizationBase()
    vis.fig_size = vis.full_page_default
    vis.newFig()

    sampling_interval = 60 * 10
    noise_str = "0.1_1" if noise else "0.0_0"
    qual = f"{noise_str}_{hoppers}"
    data_dir = os.path.abspath(os.path.dirname(GravNN.__file__)) + "/../Data/Dataframes"
    nn_a_df_file = f"{data_dir}/eros_regression_{qual}_pinn_a_metrics.data"
    nn_al_df_file = f"{data_dir}/eros_regression_{qual}_pinn_al_metrics.data"
    sh_4_file = f"{data_dir}/eros_sh_regression_4_{qual}_metrics.data"
    sh_16_file = f"{data_dir}/eros_sh_regression_16_{qual}_metrics.data"

    ############
    # Networks #
    ############
    df = pd.read_pickle(nn_a_df_file)
    plot_model_error(df, "orange", "NN PINN A", sampling_interval)

    df = pd.read_pickle(nn_al_df_file)
    plot_model_error(df, "green", "NN PINN AL", sampling_interval)

    ######################
    # Spherical Harmonics#
    ######################
    df = pd.read_pickle(sh_4_file)
    plot_model_error(df, "blue", "SH 4", sampling_interval)

    df = pd.read_pickle(sh_16_file)
    plot_model_error(df, "red", "SH 16", sampling_interval)

    #################
    # Labels + Legend
    #################
    plt.xlabel("Days Since Insertion")
    plt.ylabel(r"Average Acceleration Error [\%]")
    plt.ylim(1e-1, 1e3)

    lines = plt.gca().get_lines()
    legend1 = plt.legend(handles=lines[0::3], loc="upper left")
    exterior_line = mlines.Line2D(
        [],
        [],
        color="black",
        marker="",
        markersize=15,
        linestyle="-",
        label="Exterior",
    )
    interior_line = mlines.Line2D(
        [],
        [],
        color="black",
        marker="",
        markersize=15,
        linestyle="--",
        label="Interior",
    )
    surface_line = mlines.Line2D(
        [],
        [],
        color="black",
        marker="",
        markersize=15,
        linestyle=":",
        label="Surface",
    )
    plt.legend(handles=[exterior_line, interior_line, surface_line], loc="upper right")
    plt.gca().add_artist(legend1)

    ######################
    # Trajectory Violins #
    ######################

    plt.twinx()
    near_trajectories = generate_near_orbit_trajectories(60 * 10)
    hopper_trajectories = generate_near_hopper_trajectories(60 * 10)
    plot_orbits_as_violins(near_trajectories, near_trajectories, color="black")

    # Add rectangle patch which shows the min and max radii of the asteroid
    poly_gm = Polyhedral(Eros(), Eros().obj_8k)
    min_radius = np.min(np.linalg.norm(poly_gm.mesh.vertices, axis=1))
    max_radius = np.max(np.linalg.norm(poly_gm.mesh.vertices, axis=1))
    rect = Rectangle(
        xy=(0, min_radius),
        height=max_radius - min_radius,
        width=350,
        alpha=0.3,
        color="skyblue",
    )
    plt.gca().add_patch(rect)

    if hoppers:
        plot_orbits_as_violins(hopper_trajectories, near_trajectories, color="magenta")
    else:
        plt.gca().annotate(
            "Permissible Altitudes Below Brillouin Radius",
            xy=(0.5, 0.5),
            xytext=(0.25, 0.25),
            xycoords=rect,
            textcoords=rect,
            color="dodgerblue",
            fontsize=6,
        )

    plt.ylabel("Radius (km)")

    directory = os.path.dirname(GravNN.__file__) + "/../Plots/"
    file_name = f"regression_error_near_shoemaker_hop_{hoppers}_noise_{noise}.pdf"
    vis.save(plt.gcf(), directory + file_name)

    # plt.show()


if __name__ == "__main__":
    main(hoppers=False, noise=False)
    main(hoppers=False, noise=True)
    main(hoppers=True, noise=False)
    main(hoppers=True, noise=True)
