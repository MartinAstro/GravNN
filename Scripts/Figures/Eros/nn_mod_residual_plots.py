import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import RandomAsteroidDist
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Visualization.DataVisSuite import DataVisSuite
from GravNN.Networks.Data import get_raw_data


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
    x, y, y_pred, percent=True, alpha=0.5, label=None, marker=None, color="black"
):
    if not percent:
        diff = y - y_pred
    else:
        diff = np.abs((y - y_pred) / y) * 100.0
    df = pd.DataFrame(data=diff, index=x)
    df.sort_index(inplace=True)
    rolling_avg = df.rolling(500, 100).mean()
    plt.plot(df.index, rolling_avg, c=color)  # marker=marker, markersize=1, c=color)


def main():
    directory = os.path.abspath(".") + "/Plots/Asteroid/"
    os.makedirs(directory, exist_ok=True)

    data_vis = DataVisSuite(halt_formatting=False)
    data_vis.fig_size = data_vis.tri_vert_page

    planet = Eros()
    df = pd.read_pickle("Data/Dataframes/transformers_wo_constraints.data")

    test_trajectory = RandomAsteroidDist(
        planet, [0, planet.radius + 5000], 50000, grav_file=[planet.model_potatok]
    )
    test_poly_gm = Polyhedral(
        planet, planet.model_potatok, trajectory=test_trajectory
    ).load(override=False)


    config, model = load_config_and_model(df["id"].values[0], df)
    x_train, a_train, u_train, x_val, a_val, u_val = get_raw_data(config)
    x_sph_train, a_sph_train = get_spherical_data(x_train, a_train)

    data_vis.newFig()
    overlay_hist(x_sph_train, twinx=False)
    plt.gca().twinx()

    plot_truth = True
    marker_list = ["v", ".", "s"]
    color_list = ["black", "gray", "light gray"]
    q = 0
    for model_id in df["id"].values[:]:
        config, model = load_config_and_model(model_id, df)

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
            config["PINN_constraint_fcn"][0].__name__
        )
        os.makedirs(directory, exist_ok=True)

        x = test_poly_gm.positions
        a = test_poly_gm.accelerations
        u = test_poly_gm.potentials

        data_pred = model.generate_nn_data(x)
        a_pred = data_pred["a"]

        x_sph, a_sph = get_spherical_data(x, a)
        x_sph, a_sph_pred = get_spherical_data(x, a_pred)

        x_train, a_train, u_train, x_val, a_val, u_val = get_raw_data(config)
        x_sph_train, a_sph_train = get_spherical_data(x_train, a_train)

        data_vis.plot_residuals(
            x_sph[:, 0],
            a_sph[:, 0],
            a_sph_pred[:, 0],
            alpha=0.5,
            label=label,
            plot_truth=plot_truth,
            ylabel="Acceleration",
        )

        plot_moving_average(
            x_sph[:, 0],
            a_sph[:, 0],
            a_sph_pred[:, 0],
            marker=marker_list[q],
            color=color_list[q],
        )

        plot_truth = False
        q += 1

    data_vis.plot_radii(
        x_sph[:, 0],
        vlines=[planet.radius, config["radius_min"][0], config["radius_max"][0]],
        vline_labels=[r"$r_{Brill}$", r"$r_{min}$", r"$r_{max}$"],
    )

    data_vis.save(plt.gcf(), directory + "Mod_Acceleration_Error_Dist.png")

    plt.show()


if __name__ == "__main__":
    main()
