import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from GravNN.CelestialBodies.Planets import Earth
from GravNN.Networks.Constraints import pinn_00
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Plotting import Plotting
from GravNN.Support.transformations import cart2sph
from GravNN.Trajectories import *
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["blue", "green", "red", "blue", "green", "red"],
)


def generate_histogram(vis, trajectory, df, name):
    main_fig, ax = vis.newFig()

    positions = cart2sph(trajectory.positions)

    ax.hist(
        (positions[:, 0] - df["radius_min"][0]) / 1000.0,
        bins=100,
        alpha=0.3,
        label=name,
    )
    ax2 = ax.twinx()

    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, None])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    return ax2


def extract_sub_df(trajectory, df):
    sub_df = df[df["distribution"] == trajectory.__class__]
    sub_df = sub_df[
        (sub_df["num_units"] == 20)
        | (sub_df["num_units"] == 40)
        | (sub_df["num_units"] == 80)
    ].sort_values(by="num_units", ascending=False)
    return sub_df


def generate_altitude_curve(sub_df, df, statistic):
    ids = sub_df["id"].values
    fig_list = []
    labels = []
    linestyles = []
    # Generate their altitude rse plot
    for model_id in ids:
        tf.keras.backend.clear_session()

        config, model = load_config_and_model(model_id, df)
        if config["PINN_constraint_fcn"][0] == pinn_00:
            linestyle = "--"
        else:
            linestyle = "-"
        plotter = Plotting(model, config)
        fig = plotter.plot_alt_curve(
            statistic,
            ylabel="Nearest SH Degree",
            linestyle=linestyle,
            tick_position="left",
        )
        fig_list.append(fig)
        labels.append(str(sub_df[sub_df["id"] == model_id]["num_units"][0]))
        linestyles.append(linestyle)
    return fig_list, labels, linestyles


def generate_plot(df, trajectory, vis, name, statistic, legend=False, bounds=None):
    # Generate trajectory histogram
    ax = generate_histogram(vis, trajectory, df, name)

    # Find all networks that trained on that trajectory
    sub_df = extract_sub_df(trajectory, df)

    # Generate the altitude curves
    fig_list, labels, linestyles = generate_altitude_curve(sub_df, df, statistic)

    # Take the curves from each figure and put them all on the histogram plot
    handles = []
    colors = ["red", "red", "green", "green", "blue", "blue"]

    for j in range(0, len(fig_list)):
        cur_fig = plt.figure(fig_list[j].number)
        cur_ax = cur_fig.get_axes()[0]
        data = cur_ax.get_lines()[0].get_xydata()
        label = "$N=" + str(labels[j]) + "$" if linestyles[j] == "-" else None
        (line,) = ax.plot(
            data[:, 0],
            data[:, 1],
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

    vis.save(plt.gcf(), "Generalization/" + name + "nn_sh_altitude_equivalence.pdf")


def main():
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/")

    planet = Earth()
    points = 5000000
    radius_bounds = [planet.radius, planet.radius + 420000.0]

    df_file = "Data/Dataframes/sh_stats_earth_altitude_v2.data"
    pd.read_pickle(df_file)

    trad_df = pd.read_pickle("Data/Dataframes/traditional_nn_df.data")
    pinn_df = pd.read_pickle("Data/Dataframes/pinn_df.data")
    df = pd.concat([trad_df, pinn_df])
    # df = pinn_df

    # * Random Brillouin
    bounds = [0, 270]
    traj = RandomDist.RandomDist(planet, radius_bounds, points)
    generate_plot(df, traj, vis, "Random", "param_rse_mean", legend=True, bounds=bounds)
    generate_plot(
        df,
        traj,
        vis,
        "Random_sigma",
        "param_sigma_2_mean",
        legend=True,
        bounds=bounds,
    )
    generate_plot(
        df,
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
    traj = ExponentialDist.ExponentialDist(
        planet,
        radius_bounds,
        points=1200000,
        scale_parameter=[420000.0 / 3.0],
        invert=[True],
    )
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
    # generate_plot(
    #     exp_df,
    #     traj,
    #     vis,
    #     "Exp_comp",
    #     "param_sigma_2_c_mean",
    #     legend=False,
    #     bounds=bounds,
    # )

    # * Exponential
    # exp_df = pd.read_pickle("Data/Dataframes/exponential_invert_dist_v10.data")
    # traj = ExponentialDist.ExponentialDist(
    #     planet,
    #     radius_bounds,
    #     points=1200000,
    #     scale_parameter=[420000.0 / 10.0],
    #     invert=[True],
    # )
    # generate_plot(
    #     exp_df,
    #     traj,
    #     vis,
    #     "Exp_narrow",
    #     "param_rse_mean",
    #     legend=False,
    #     bounds=bounds,
    # )
    # generate_plot(
    #     exp_df,
    #     traj,
    #     vis,
    #     "Exp_sigma_narrow",
    #     "param_sigma_2_mean",
    #     legend=False,
    #     bounds=bounds,
    # )
    # generate_plot(
    #     exp_df,
    #     traj,
    #     vis,
    #     "Exp_comp_narrow",
    #     "param_sigma_2_c_mean",
    #     legend=False,
    #     bounds=bounds,
    # )
    plt.show()


if __name__ == "__main__":
    main()
