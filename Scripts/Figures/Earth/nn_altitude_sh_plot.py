import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import GravNN
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Networks.Constraints import pinn_00
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph
from GravNN.Trajectories.ExponentialDist import ExponentialDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["blue", "green", "red", "blue", "green", "red"],
)


def generate_histogram(vis, trajectory, min_radius, name):
    main_fig, ax = vis.newFig()

    positions = cart2sph(trajectory.positions)
    r_mag = positions[:, 0]
    altitude = r_mag - min_radius
    altitude_km = altitude / 1000.0
    ax.hist(
        altitude_km,
        bins=100,
        alpha=0.3,
        label=name,
    )

    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, None])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax2 = ax.twinx()

    return ax2


def extract_sub_df(trajectory, df):
    sub_df = df[df["distribution"] == trajectory.__class__]
    sub_df = sub_df[
        (sub_df["num_units"] == 20)
        | (sub_df["num_units"] == 40)
        | (sub_df["num_units"] == 80)
    ].sort_values(by="num_units", ascending=False)
    return sub_df


def generate_altitude_curve(df, statistic):
    ids = df["id"].values
    fig_list = []
    labels = []
    linestyles = []
    GravNN_path = os.path.dirname(GravNN.__file__)
    vis = VisualizationBase()

    # Generate their altitude rse plot
    for model_id in ids:
        # Load model
        tf.keras.backend.clear_session()
        config, model = load_config_and_model(model_id, df)

        # Gather rse statistics for that model (must run analysis w/ altitude)
        directory = f"{GravNN_path}/Data/Networks/{model_id}/"
        df = pd.read_pickle(directory + "rse_alt.data")

        linestyle = "-"
        if config["PINN_constraint_fcn"][0] == pinn_00:
            linestyle = "--"

        fig, ax = vis.newFig()
        alt_km = df.index / 1000.0
        plt.plot(alt_km, df[statistic], linestyle=linestyle)
        plt.xlabel("Altitude [km]")
        plt.ylabel("Nearest SH Degree")
        ax.yaxis.set_ticks_position("left")

        fig_list.append(fig)
        labels.append(str(config["num_units"][0]))
        linestyles.append(linestyle)
    return fig_list, labels, linestyles


def generate_plot(df, trajectory, vis, name, statistic, legend=False, bounds=None):
    min_radius = df["radius_min"][0]

    # Generate trajectory histogram
    ax = generate_histogram(vis, trajectory, min_radius, name)

    # Generate the SH equivalent altitude curves individually
    fig_list, labels, linestyles = generate_altitude_curve(df, statistic)

    # Take the curves from each figure and put them all on the histogram plot
    handles = []
    colors = ["red", "red", "green", "green", "blue", "blue"]

    for j, fig in enumerate(fig_list):
        cur_fig = plt.figure(fig.number)
        cur_ax = cur_fig.get_axes()[0]
        data = cur_ax.get_lines()[0].get_xydata()
        alt_km = data[:, 0]
        sh_eq = data[:, 1]
        label = f"$N={labels[j]}$" if linestyles[j] == "-" else None
        (line,) = ax.plot(
            alt_km,
            sh_eq,
            label=label,
            linestyle=linestyles[j],
            c=colors[j],
        )
        handles.append(line)
        plt.close(cur_fig)

    ax.set_ylabel("Nearest SH Degree")
    if legend:
        ax.legend(handles=handles, loc="upper right")

    ax.yaxis.set_label_position("left")  # Works
    plt.tick_params(
        axis="y",  # changes apply to the x-axis
        which="minor",  # both major and minor ticks are affected
        left=True,  # ticks along the left edge are off
        right=False,  # ticks along the top edge are off
        labelleft=True,
        labelright=False,
    )  # labels along the left edge are off
    plt.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
    )
    ax.set_ylim(bottom=bounds[0], top=bounds[1])

    vis.save(plt.gcf(), f"Generalization/{name}nn_sh_altitude_equivalence.pdf")


def main():
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/")

    planet = Earth()
    points = 5000000
    radius_bounds = [planet.radius, planet.radius + 420000.0]

    trad_df = pd.read_pickle("Data/Dataframes/traditional_nn_df.data")
    pinn_df = pd.read_pickle("Data/Dataframes/pinn_df.data")
    df = pd.concat([trad_df, pinn_df])

    # * Random Brillouin
    bounds = [0, 270]
    traj = RandomDist(planet, radius_bounds, points)

    # Find all networks that trained on that trajectory
    sub_df = extract_sub_df(traj, df)

    generate_plot(
        sub_df,
        traj,
        vis,
        "Random",
        "param_rse_mean",
        legend=True,
        bounds=bounds,
    )
    generate_plot(
        sub_df,
        traj,
        vis,
        "Random_sigma",
        "param_sigma_2_mean",
        legend=True,
        bounds=bounds,
    )
    generate_plot(
        sub_df,
        traj,
        vis,
        "Random_comp",
        "param_sigma_2_c_mean",
        legend=True,
        bounds=bounds,
    )

    vis.fig_size = vis.half_page

    # * Exponential
    bounds = [0, 160]
    exp_df = pd.read_pickle("Data/Dataframes/exponential_invert_dist_all.data")
    traj = ExponentialDist(
        planet,
        radius_bounds,
        points=1200000,
        scale_parameter=[420000.0 / 3.0],
        invert=[True],
    )
    exp_df = extract_sub_df(traj, df)

    generate_plot(
        exp_df,
        traj,
        vis,
        "Exp",
        "param_rse_mean",
        legend=False,
        bounds=bounds,
    )
    generate_plot(
        exp_df,
        traj,
        vis,
        "Exp_sigma",
        "param_sigma_2_mean",
        legend=False,
        bounds=bounds,
    )
    generate_plot(
        exp_df,
        traj,
        vis,
        "Exp_comp",
        "param_sigma_2_c_mean",
        legend=False,
        bounds=bounds,
    )

    # * Exponential Narrower
    exp_df = pd.read_pickle("Data/Dataframes/exponential_invert_dist_v10.data")
    traj = ExponentialDist(
        planet,
        radius_bounds,
        points=1200000,
        scale_parameter=[420000.0 / 10.0],
        invert=[True],
    )
    exp_df = extract_sub_df(traj, df)

    generate_plot(
        exp_df,
        traj,
        vis,
        "Exp_narrow",
        "param_rse_mean",
        legend=False,
        bounds=bounds,
    )
    generate_plot(
        exp_df,
        traj,
        vis,
        "Exp_sigma_narrow",
        "param_sigma_2_mean",
        legend=False,
        bounds=bounds,
    )
    generate_plot(
        exp_df,
        traj,
        vis,
        "Exp_comp_narrow",
        "param_sigma_2_c_mean",
        legend=False,
        bounds=bounds,
    )
    plt.show()


if __name__ == "__main__":
    main()
