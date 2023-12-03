import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.PointMass import PointMass
from GravNN.Networks.Model import load_config_and_model


def plot_analytic_model(fcn, min_R, max_R, radius, label, new_fig=True):
    r = np.linspace(min_R, max_R, 1000)
    x_array = np.zeros((r.shape[0], 3))
    y_array = np.zeros((r.shape[0], 3))
    z_array = np.zeros((r.shape[0], 3))

    x_array[:, 0] = r
    y_array[:, 1] = r
    z_array[:, 2] = r

    U_x = fcn(x_array)
    U_y = fcn(y_array)
    U_z = fcn(z_array)

    plt.figure(1)
    plt.plot(r / radius, U_x, label=f"{label} x")
    plt.xlabel("x")

    plt.figure(2)
    plt.plot(r / radius, U_y, label=f"{label} y")
    plt.xlabel("y")

    plt.figure(3)
    plt.plot(r / radius, U_z, label=f"{label} z")
    plt.xlabel("z")

    plt.legend()

    return U_x, U_y, U_z, r


def main():
    planet = Eros()
    poly_model = generate_heterogeneous_model(planet, planet.obj_8k)
    point_mass_model = PointMass(planet)

    R_max = planet.radius * 5
    R_min = planet.radius_min
    plot_analytic_model(
        poly_model.compute_potential,
        R_min,
        R_max,
        planet.radius,
        "Poly",
    )
    plot_analytic_model(
        point_mass_model.compute_potential,
        R_min,
        R_max,
        planet.radius,
        "PM",
        new_fig=False,
    )

    # plot_analytic_model(poly_model.compute_acceleration, R_min, R_max, planet.radius, "Poly" )
    # plot_analytic_model(point_mass_model.compute_acceleration, R_min, R_max, planet.radius, "PM", new_fig=False)

    plt.show()


def main_NN():
    df_file = "Data/Dataframes/eros_cost_fcn_pinn_III_fuse.data"
    config, model = load_config_and_model(df_file)

    def analytic_model(x):
        x_input = model.x_preprocessor(x)
        u_pred = model.analytic_model(x_input, training=False)
        return u_pred

    def network_model(x):
        x_input = model.x_preprocessor(x)
        u_pred = model.network(x_input, training=False)
        return u_pred

    planet = Eros()
    R_max = planet.radius * 5
    R_min = planet.radius_min
    Ux, Uy, Uz, r = plot_analytic_model(
        analytic_model,
        R_min,
        R_max,
        planet.radius,
        "Analytic",
    )
    Ux_NN, Uy_NN, Uz_NN, r = plot_analytic_model(
        network_model,
        R_min,
        R_max,
        planet.radius,
        "Network",
        new_fig=False,
    )

    Ux_err = (Ux - Ux_NN) / Ux
    Uy_err = (Uy - Uy_NN) / Uy
    Uz_err = (Uz - Uz_NN) / Uz

    plt.figure()
    plt.plot(r / planet.radius, Ux_err, label="Ux")
    plt.plot(r / planet.radius, Uy_err, label="Uy")
    plt.plot(r / planet.radius, Uz_err, label="Uz")
    plt.plot(
        r / planet.radius,
        np.mean([Ux_err, Uy_err, Uz_err], axis=0),
        label="Mean",
        linestyle="--",
    )

    plt.show()


if __name__ == "__main__":
    # main()
    main_NN()
