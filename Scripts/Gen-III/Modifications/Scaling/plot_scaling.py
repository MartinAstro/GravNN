import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Configs.Eros_Configs import get_default_eros_config
from GravNN.Networks.Data import DataSet
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Visualization.VisualizationBase import VisualizationBase

vis = VisualizationBase()
plt.rc("font", size=7)
vis.fig_size = (vis.w_half, vis.w_half)

plt.rc("text", usetex=True)


def get_data_config():
    config = get_default_eros_config()
    config.update(
        {
            "N_dist": [8192],
            "radius_max": [Eros().radius * 50],
            "N_train": [5000],
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
    plt.scatter(x, y, alpha=0.3, s=1, label=label)
    plt.xlabel("Distance from COM [Radii]")
    plt.tight_layout()
    # plt.gca().set_xscale("log")


def main():
    planet = Eros()
    MAX_RADIUS = 10
    # config = get_earth_data_config(MAX_RADIUS)
    config = get_data_config()
    planet = config["planet"][0]
    mu = planet.mu
    R = planet.radius
    R_min = planet.radius_min

    data = DataSet(data_config=config)

    # True Potential
    r_train = np.linalg.norm(data.raw_data["x_train"], axis=1)
    u_train = data.raw_data["u_train"].squeeze()

    # Analytic Potential (Uniform Sphere)
    r_line = np.linspace(0, MAX_RADIUS * planet.radius, 100)
    u_analytic = analytic_potential_fcn(mu, r_line, R, R_min)

    u_brill = mu / R
    r_plot = r_train / R

    # True vs Analytic potential
    plot(r_plot, u_train, "$U$")
    plt.plot(
        r_line / R,
        u_analytic,
        label=r"$U_{\text{PM}}$",
        color="black",
        linewidth=0.5,
    )
    plt.xlim([0.5, 10])
    plt.ylabel("Potential")
    plt.legend()
    vis.save(plt.gcf(), "Original_Potential")

    u_train_ND = u_train / u_brill
    u_train_ND_scaled = u_train_ND * r_plot
    plot(r_plot, u_train_ND_scaled, "$U * n(r)$")
    plt.ylabel(r"Proxy Potential, $\hat{U}_{\text{NN}}$")
    plt.xlim([0.5, 10])
    vis.save(plt.gcf(), "Scaled_Potential")

    # u_analytic_ND = u_analytic / u_brill
    # u_analytic_ND_scaled = u_analytic_ND * r_line / R
    # plot(r_line / R, u_analytic_ND_scaled, "$U * n(r)$")


if __name__ == "__main__":
    main()
    plt.show()
