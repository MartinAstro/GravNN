import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Constraints import *
from GravNN.Networks.Data import DataSet
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Trajectories import RandomDist
from GravNN.Visualization.DataVisSuite import DataVisSuite


def make_fcn_name_latex_compatable(name):
    components = name.split("_")
    return components[0] + r"$_{" + components[1] + r"}$"


def get_title(config):
    fcn_name = make_fcn_name_latex_compatable(config["PINN_constraint_fcn"][0].__name__)
    title = (
        fcn_name
        + " "
        + str(config["N_train"][0])
        + " "
        + str(config["radius_max"][0] - config["planet"][0].radius)
    )
    return title


def minmax(values):
    print(np.min(values, axis=0))
    print(np.max(values, axis=0))


def get_spherical_data(x, a):
    x_sph = cart2sph(x)
    a_sph = project_acceleration(x_sph, np.array(a, dtype=float))
    return x_sph, a_sph


def overlay_hist(x_sph_train, twinx=True):
    if twinx:
        plt.gca().twinx()
    plt.hist(
        x_sph_train[:, 0],
        bins=30,
        alpha=0.7,
        range=[np.min(x_sph_train[:, 0]), np.max(x_sph_train[:, 0])],
        color="gray",
    )  # edgecolor='black', linewidth=0.5, fill=True)
    plt.ylabel("Frequency")


def plot_moving_average(
    x,
    y,
    y_pred,
    percent=True,
    alpha=0.5,
    label=None,
    marker=None,
    color="black",
):
    if not percent:
        diff = y - y_pred
    else:
        diff = (
            np.abs(np.linalg.norm(y - y_pred, axis=1) / np.linalg.norm(y, axis=1))
            * 100.0
        )
    df = pd.DataFrame(data=diff, index=x)
    df.sort_index(inplace=True)
    rolling_avg = df.rolling(500, 100).mean()
    plt.plot(df.index, rolling_avg, c=color)  # marker=marker, markersize=1, c=color)


def main():
    directory = os.path.abspath(".") + "/Plots/Asteroid/"
    os.makedirs(directory, exist_ok=True)

    data_vis = DataVisSuite()
    data_vis.fig_size = data_vis.tri_vert_page

    planet = Bennu()
    model_file = planet.stl_200k
    df, descriptor = (
        pd.read_pickle("Data/Dataframes/bennu_traditional_wo_annealing.data"),
        "[0,R*3]",
    )
    # df, descriptor = pd.read_pickle("Data/Dataframes/useless_070721_v2.data"), "[R,R*3]"
    # df, descriptor = pd.read_pickle("Data/Dataframes/useless_070621_v4.data"), "[R,R*3] + Some"
    # df, descriptor = pd.read_pickle("Data/Dataframes/useless_070721_v3.data"), "[0,10000]"# PLC, ALC, APLC

    df = df[df["N_train"] == 2500]
    df = df[df["PINN_constraint_fcn"] != pinn_00]

    test_trajectory = RandomDist(
        planet,
        [0, planet.radius * 3],
        20000,
        grav_file=[model_file],
    )
    test_poly_gm = Polyhedral(planet, model_file, trajectory=test_trajectory).load(
        override=False,
    )

    data_vis.newFig()
    data_vis.newFig()

    config, model = load_config_and_model(df, df["id"].values[0])
    data = DataSet(config)
    x_train = data.raw_data["x_train"]
    a_train = data.raw_data["a_train"]
    x_sph_train, a_sph_train = get_spherical_data(x_train, a_train)

    # plt.figure(1)
    # overlay_hist(x_sph_train,twinx=False)
    # plt.gca().twinx()

    plt.figure(2)
    overlay_hist(x_sph_train, twinx=False)
    plt.gca().twinx()

    plot_truth = True
    q = 0
    for model_id in df["id"].values[:]:
        config, model = load_config_and_model(df, model_id)

        if config["PINN_constraint_fcn"][0].__name__.lower() == "pinn_p":
            continue

        extra_samples = config.get("extra_N_train", [None])[0]
        directory = os.path.abspath(".") + "/Plots/Asteroid/%s/%s_%s_%s/" % (
            planet.__class__.__name__,
            str(np.round(config["radius_min"][0], 2)),
            str(np.round(config["radius_max"][0], 2)),
            str(extra_samples),
        )

        label = make_fcn_name_latex_compatable(
            config["PINN_constraint_fcn"][0].__name__,
        )
        os.makedirs(directory, exist_ok=True)

        x = test_poly_gm.positions
        a = test_poly_gm.accelerations

        a_pred = model.compute_acceleration(x)

        x_sph, a_sph = get_spherical_data(x, a)
        x_sph, a_sph_pred = get_spherical_data(x, a_pred)

        data = DataSet(config)
        x_train = data.raw_data["x_train"]
        a_train = data.raw_data["a_train"]
        x_sph_train, a_sph_train = get_spherical_data(x_train, a_train)

        # plt.figure(1)
        # data_vis.plot_residuals(x_sph[:,0], u, u_pred, alpha=0.5, label=label, plot_truth=plot_truth, ylabel='Potential')
        # plot_moving_average(x_sph[:,0], u, u_pred)#, marker=marker_list[q], color=color_list[q])

        plt.figure(2)
        data_vis.plot_residuals(
            x_sph[:, 0],
            a_sph,
            a_sph_pred,
            alpha=0.5,
            label=label,
            plot_truth=plot_truth,
            ylabel="Acceleration",
        )
        plot_moving_average(
            x_sph[:, 0],
            a_sph,
            a_sph_pred,
        )  # , marker=marker_list[q], color=color_list[q])

        plot_truth = False
        q += 1

    # plt.figure(1)
    # # overlay_hist(x_sph_train)
    # data_vis.plot_radii(x_sph[:,0], vlines=[planet.radius, config['radius_min'][0], config['radius_max'][0]], vline_labels=[r'$r_{Brill}$', r'$r_{min}$', r'$r_{max}$'])
    # data_vis.save(plt.gcf(), directory+"Potential_Error_Dist.png")

    plt.figure(2)
    # overlay_hist(x_sph_train)
    data_vis.plot_radii(
        x_sph[:, 0],
        vlines=[planet.radius, config["radius_min"][0], config["radius_max"][0]],
        vline_labels=[r"$r_{Brill}$", r"$r_{min}$", r"$r_{max}$"],
    )
    data_vis.save(plt.gcf(), directory + "Acceleration_Error_Dist.png")

    plt.show()


if __name__ == "__main__":
    main()
