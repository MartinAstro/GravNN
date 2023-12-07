import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Configs.Earth_Configs import get_default_earth_config
from GravNN.Networks.Configs.Eros_Configs import get_default_eros_config
from GravNN.Networks.Data import DataSet
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Visualization.VisualizationBase import VisualizationBase

vis = VisualizationBase()
plt.rc("font", size=7)
vis.fig_size = (vis.w_half, vis.w_half)

plt.rc("text", usetex=True)


def get_earth_data_config(max_radius_scale):
    config = get_default_earth_config()
    max_radius = config["planet"][0].radius * max_radius_scale
    config.update(
        {
            "radius_max": [max_radius],
            "N_dist": [10000],
            "N_train": [9500],
            "N_val": [500],
            "dummy_transformer": [DummyScaler()],
        },
    )
    return config


def get_data_config(max_radius_scale):
    config = get_default_eros_config()
    max_radius = config["planet"][0].radius * max_radius_scale
    config.update(
        {
            "radius_max": [max_radius],
            "N_dist": [10000],
            "N_train": [9500],
            "N_val": [500],
            "dummy_transformer": [DummyScaler()],
        },
    )
    return config


def analytic_potential_fcn(mu, r, R_max, R_min):
    def u_exterior_fcn(mu, r):
        return -mu / r

    def u_trans_fcn(mu, r, R_max):
        return mu * (r / R_max) ** 3 / r + u_exterior_fcn(mu, R_max) * 2

    def u_interior_fcn(mu, r, R_max):
        return mu * (r / R_max) ** 3 / r + u_exterior_fcn(mu, R_max) * 2

    u_exterior = u_exterior_fcn(mu, r)
    u_transition = u_trans_fcn(mu, r, R_max)
    u_interior = u_interior_fcn(mu, r, R_max)

    u_ideal = np.where(r < R_max, u_transition, u_exterior)
    u_ideal = np.where(r < R_min, u_interior, u_ideal)
    # u_ideal = np.clip(u_exterior, 0.0, 1.0)
    return u_ideal


def plot(x, y, label, new_fig=True):
    if new_fig:
        vis.newFig()
    plt.scatter(x, y, alpha=0.3, s=2, label=label)
    plt.xlabel("Planet Radii from Surface [-]")
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()
    plt.gca().set_xscale("log")


def main():
    planet = Eros()
    MAX_RADIUS = 10
    # config = get_earth_data_config(MAX_RADIUS)
    config = get_data_config(MAX_RADIUS)
    planet = config["planet"][0]
    mu = planet.mu
    R = planet.radius
    R_min = planet.radius_min

    data = DataSet(data_config=config)

    # True Potential
    r_train = np.linalg.norm(data.raw_data["x_train"], axis=1)
    u_train = data.raw_data["u_train"].squeeze()
    u_analytic = analytic_potential_fcn(mu, r_train, R, R_min)

    # Analytic Potential (Uniform Sphere)
    r_line = np.linspace(0, MAX_RADIUS * planet.radius, 100)
    u_ideal_line = analytic_potential_fcn(mu, r_line, R, R_min)
    u_ideal_scatter = analytic_potential_fcn(mu, r_train, R, R_min)
    u_ideal_pm_scatter = -mu / r_train

    u_brill = mu / R
    dU = u_train - u_analytic
    r_plot = r_train / R

    # True vs Analytic potential
    plot(r_plot, u_train, "True Potential")
    plt.plot(
        r_line / R,
        u_ideal_line,
        label="Analytic Potential",
        color="black",
        linewidth=2,
    )

    percent_error = (u_train - u_ideal_scatter) / u_train * 100
    percent_error_pm = (u_train - u_ideal_pm_scatter) / u_train * 100
    plot(r_plot, percent_error, "Percent Error")
    plot(r_plot, percent_error_pm, "Percent Error PM")

    # Simple U * r scaling =~ mu(r)
    U_plot = u_train * r_plot
    plot(r_plot, U_plot, "U * r")

    # dU
    U_plot = dU
    plot(r_plot, U_plot, "dU")

    U_plot = dU / u_brill
    plot(r_plot, U_plot, "dU Non-Dim")
    plt.scatter(r_plot, -1 / (r_plot) ** 2, alpha=0.5, s=2, color="gray")
    plt.scatter(r_plot, 1 / (r_plot) ** 2, alpha=0.5, s=2, color="gray")

    U_plot = dU * r_plot
    plot(r_plot, U_plot, "dU * r")


def main_layer():
    planet = Eros()
    MAX_RADIUS = 1000
    # config = get_earth_data_config(MAX_RADIUS)
    config = get_data_config(MAX_RADIUS)
    planet = config["planet"][0]

    data = DataSet(data_config=config)

    # True Potential
    r_train = np.linalg.norm(data.raw_data["x_train"], axis=1)
    u_train = data.raw_data["u_train"].squeeze()

    # Analytic Potential (Uniform Sphere)
    np.linspace(0, MAX_RADIUS * planet.radius, 100)

    import pandas as pd

    from GravNN.Networks.Layers import ScaleNNPotential
    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle("Data/Dataframes/pinn_III_mods_scaling.data")
    config, model = load_config_and_model(df, idx=-1)

    layer = ScaleNNPotential(3.0, **config)

    r_train.reshape((-1, 1)) / planet.radius
    scale_input = np.ones_like(u_train).reshape((-1, 1))
    s = layer(r_train.reshape((-1, 1)) / planet.radius, scale_input).numpy().squeeze()
    u_nn = u_train * s
    plot(r_train / planet.radius, u_train, "True Potential", new_fig=False)
    plot(r_train / planet.radius, u_nn, "NN Potential", new_fig=False)


if __name__ == "__main__":
    main()
    main_layer()
    plt.show()
