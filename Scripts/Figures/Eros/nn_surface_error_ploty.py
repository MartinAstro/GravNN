import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Trajectories import SurfaceDist
from GravNN.Visualization.PolyVisualization import PolyVisualization


def format():
    plt.gca().set_xticklabels("")
    plt.gca().set_yticklabels("")
    plt.gca().set_zticklabels("")
    plt.gca().view_init(elev=35, azim=180 + 45, roll=0)


def main():
    planet = Eros()
    trajectory = SurfaceDist(planet, planet.obj_200k)

    x_hetero_poly, a_hetero_poly, u_hetero_poly = get_hetero_poly_data(
        trajectory,
        planet.obj_200k,
        remove_point_mass=[False],
    )

    x_homo_poly, a_homo_poly, u_homo_poly = get_poly_data(
        trajectory,
        planet.obj_200k,
        remove_point_mass=[False],
    )

    #######################################
    # Surface Acceleration of homogeneous
    #######################################
    vis = PolyVisualization()
    vis.fig_size = (vis.w_half, vis.w_half)
    vis.plot_polyhedron(
        planet.obj_200k,
        a_homo_poly,
        label="Acceleration [m/$s^2$]",
        log=False,
        cmap="bwr",
        cmap_reverse=False,
        percent=False,
        alpha=1,
    )
    format()
    vis.save(plt.gcf(), "eros_homogenous_field.pdf")

    #######################################
    # Surface Acceleration of heterogenous
    #######################################
    vis.plot_polyhedron(
        planet.obj_200k,
        a_hetero_poly,
        label="Acceleration [m/$s^2$]",
        log=False,
        cmap="bwr",
        cmap_reverse=False,
        percent=False,
        alpha=1,
    )
    format()
    vis.save(plt.gcf(), "eros_heterogenous_field.pdf")

    #######################################
    # Error of homogenous assumption
    #######################################
    da_norm = np.linalg.norm(a_homo_poly - a_hetero_poly, axis=1)
    a_norm = np.linalg.norm(a_hetero_poly, axis=1)

    a_error = da_norm / a_norm * 100
    a_error = a_error.reshape((-1, 1))
    vis.plot_polyhedron(
        planet.obj_200k,
        a_error,
        label="Acceleration Errors",
        log=False,
        percent=True,
        max_percent=0.3,
        alpha=1,
    )
    format()
    vis.save(plt.gcf(), "eros_homo_surface_error.pdf")

    #######################################
    # Error of PINN model
    #######################################
    df = pd.read_pickle("Data/Dataframes/heterogenous_eros_041823.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(df, model_id)

    a_pinn = model.compute_acceleration(x_hetero_poly)
    da_norm = np.linalg.norm(a_pinn - a_hetero_poly, axis=1)
    a_norm = np.linalg.norm(a_hetero_poly, axis=1)

    a_error = da_norm / a_norm * 100
    a_error = a_error.reshape((-1, 1))
    vis.plot_polyhedron(
        planet.obj_200k,
        a_error,
        label="Acceleration Errors",
        log=False,
        percent=True,
        max_percent=0.3,
        alpha=1,
    )
    format()
    vis.save(plt.gcf(), "eros_hetero_pinn_surface_error.pdf")

    # plt.show()


if __name__ == "__main__":
    main()
