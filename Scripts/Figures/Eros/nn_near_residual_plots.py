import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Trajectories import RandomAsteroidDist
from GravNN.Trajectories.utils import generate_near_orbit_trajectories
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
        bins=100,
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


def get_near_positions():
    model_file = Eros().obj_200k
    trajectories = generate_near_orbit_trajectories(60 * 10)
    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        r, a, u = get_poly_data(
            trajectory,
            model_file,
            remove_point_mass=[False],
            override=[False],
        )

        try:
            x = np.concatenate((x, r), axis=0)
        except:
            x = r

    r = cart2sph(np.array(x))
    return r


def main():
    directory = os.path.abspath(".") + "/Plots/Asteroid/"
    os.makedirs(directory, exist_ok=True)

    data_vis = DataVisSuite()
    data_vis.fig_size = data_vis.tri_vert_page

    planet = Eros()

    df, descriptor = (
        pd.read_pickle("Data/Dataframes/near_all_data.data"),
        "[5000,10000]",
    )

    model_file = planet.obj_200k

    test_trajectory = RandomAsteroidDist(
        planet,
        [0, planet.radius * 3],
        20000,
        grav_file=[model_file],
    )
    test_poly_gm = Polyhedral(planet, model_file, trajectory=test_trajectory).load()

    data_vis.newFig()
    x_sph_train = get_near_positions()

    plt.figure(1)
    overlay_hist(x_sph_train, twinx=False)
    plt.gca().twinx()

    plot_truth = True
    marker_list = ["v", ".", "s"]
    color_list = ["black", "gray", "light gray"]
    q = 0
    for model_id in df["id"].values[:]:
        config, model = load_config_and_model(df, model_id)

        if config["PINN_constraint_fcn"][0].__name__.lower() == "pinn_p":
            continue

        extra_samples = config.get("extra_N_train", [None])[0]
        directory = (
            os.path.abspath(".")
            + "/Plots/Asteroid/"
            + str(np.round(config["radius_min"][0], 2))
            + "_"
            + str(np.round(config["radius_max"][0], 2))
            + "_"
            + str(extra_samples)
            + "/"
        )
        label = make_fcn_name_latex_compatable(
            config["PINN_constraint_fcn"][0].__name__,
        )
        os.makedirs(directory, exist_ok=True)

        x = test_poly_gm.positions
        a = test_poly_gm.accelerations

        data_pred = {
            "u": model.compute_potential(x),
            "a": model.compute_acceleration(x),
        }
        a_pred = data_pred["a"]
        data_pred["u"]

        x_sph, a_sph = get_spherical_data(x, a)
        x_sph, a_sph_pred = get_spherical_data(x, a_pred)

        plt.figure(1)
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
            marker=marker_list[q],
            color=color_list[q],
        )

        plot_truth = False
        q += 1

    plt.figure(1)
    data_vis.plot_radii(
        x_sph[:, 0],
        vlines=[planet.radius, config["radius_min"][0], config["radius_max"][0]],
        vline_labels=[r"$r_{Brill}$", r"$r_{min}$", r"$r_{max}$"],
    )
    # data_vis.save(plt.gcf(), directory+"Acceleration_Error_Dist.png")

    plt.show()


if __name__ == "__main__":
    main()
