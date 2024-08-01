import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    generate_heterogeneous_model,
)
from GravNN.Visualization.PolyVisualization import PolyVisualization


def symmetric_eros():
    planet = Eros()
    gravity_model = generate_heterogeneous_model(planet, planet.obj_8k, symmetric=True)

    vis = PolyVisualization()
    vis.fig_size = (vis.w_half, vis.w_half)
    vis.plot_polyhedron(planet.obj_8k, None, cmap="Greys", cbar=False, alpha=0.1)

    offset_0 = gravity_model.offset_list[0]
    offset_1 = gravity_model.offset_list[1]
    offset_2 = gravity_model.offset_list[2]

    plt.gca().scatter(offset_0[0], offset_0[1], offset_1[2], s=200, color="blue")
    plt.gca().scatter(offset_1[0], offset_1[1], offset_1[2], s=100, color="red")
    plt.gca().scatter(offset_2[0], offset_2[1], offset_1[2], s=100, color="red")

    plt.gca().set_xticklabels("")
    plt.gca().set_yticklabels("")
    plt.gca().set_zticklabels("")
    plt.gca().view_init(elev=35, azim=180 + 45, roll=0)

    vis.save(plt.gcf(), "eros_hetero_density_symmetric")


def asymmetric_eros():
    planet = Eros()
    gravity_model = generate_heterogeneous_model(planet, planet.obj_8k, symmetric=False)

    vis = PolyVisualization()
    vis.fig_size = (vis.h_quad, vis.h_quad)
    vis.plot_polyhedron(planet.obj_8k, None, cmap="Greys", cbar=False, alpha=0.1)

    offset_0 = gravity_model.offset_list[0]
    offset_1 = gravity_model.offset_list[1]

    plt.gca().scatter(offset_0[0], offset_0[1], offset_1[2], s=70, color="blue")
    plt.gca().scatter(offset_1[0], offset_1[1], offset_1[2], s=70, color="red")

    plt.gca().set_ylabel("Y [R]")
    plt.gca().set_xlabel("X [R]")
    plt.gca().set_zlabel("Z [R]")

    # set x, y, z lim as radius
    plt.gca().set_xlim(-planet.radius, planet.radius)
    plt.gca().set_ylim(-planet.radius, planet.radius)
    plt.gca().set_zlim(-planet.radius, planet.radius)

    tick_locations = np.linspace(-planet.radius, planet.radius, 5)
    tick_labels = [str(x / planet.radius) for x in tick_locations]
    plt.gca().set_xticks(tick_locations, tick_labels)
    plt.gca().set_yticks(tick_locations, tick_labels)
    plt.gca().set_zticks(tick_locations, tick_labels)

    # decrease tick padding
    plt.gca().tick_params(axis="x", pad=0)
    plt.gca().tick_params(axis="y", pad=0)
    plt.gca().tick_params(axis="z", pad=0)

    # decrease label padding
    plt.gca().xaxis.labelpad = -2
    plt.gca().yaxis.labelpad = -2
    plt.gca().zaxis.labelpad = -2

    # plt.gca().view_init(elev=35, azim=180 + 45, roll=0)
    # plt.gca().view_init(elev=53, azim=-114, roll=0)
    # plt.gca().view_init(elev=13, azim=-103, roll=0)
    plt.gca().view_init(elev=45, azim=-130, roll=0)

    # bbox_tight causes the saved figure to be cropped, so crop within latex instead.
    vis.save(plt.gcf(), "eros_hetero_density_asymmetric", bbox_inches=None)


if __name__ == "__main__":
    symmetric_eros()
    asymmetric_eros()

    plt.show()
