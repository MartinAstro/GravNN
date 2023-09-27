import os

import matplotlib.pyplot as plt
import pandas as pd
import StatOD
from StatOD.utils import *

from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.Networks.Configs import *
from GravNN.Networks.Model import load_config_and_model

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def plot_analytic_model(gravity_model):
    config = gravity_model.config
    planet = config["planet"][0]
    x_coordinates = np.linspace(1, 2 * config["planet"][0].radius, 200)
    positions = np.zeros((x_coordinates.shape[0], 3))
    positions[:, 1] = x_coordinates

    analytic_potential = gravity_model.compute_potential(positions).numpy().squeeze()
    u_point_mass = -planet.mu / x_coordinates
    u_point_mass_mod = (-planet.mu / x_coordinates) * (1 / x_coordinates) ** 0.75

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x_coordinates, analytic_potential, label="Analytic Model")
    plt.plot(x_coordinates, u_point_mass, label="1/r")
    plt.plot(x_coordinates, u_point_mass_mod, label="1/r-mod")
    plt.ylim([-60, 0])

    poly_gm = generate_heterogeneous_model(planet, planet.obj_8k)
    poly_potential = poly_gm.compute_potential(positions)
    plt.subplot(3, 1, 2)
    plt.plot(x_coordinates, poly_potential, label="Poly Model")
    plt.plot(x_coordinates, u_point_mass, label="1/r")
    plt.plot(x_coordinates, u_point_mass_mod, label="1/r-mod")

    # plt.plot(x_coordinates, -(planet.radius/x_coordinates)**0.75, label="1/r-0.75")

    plt.ylim([-60, 0])

    plt.subplot(3, 1, 3)
    plt.plot(x_coordinates, poly_potential - analytic_potential, label="Poly Model")
    plt.ylim([-60, 0])


def main():
    statOD_dir = os.path.dirname(StatOD.__file__)
    pinn_name = "eros_pm_061023"
    # pinn_name = "eros_poly_061023"
    # pinn_name = "eros_pm_053123"
    pinn_file = f"{statOD_dir}/../Data/Dataframes/{pinn_name}.data"

    df = pd.read_pickle(pinn_file)
    config, gravity_model = load_config_and_model(
        df.id.values[-1],
        df,
        custom_dtype="float32",
        only_weights=True,
    )
    gravity_model.config = config
    plot_analytic_model(gravity_model)
    plt.show()


if __name__ == "__main__":
    main()
