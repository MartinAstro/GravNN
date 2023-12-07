import matplotlib.pyplot as plt

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    generate_heterogeneous_model,
)
from GravNN.Visualization.PolyVisualization import PolyVisualization

if __name__ == "__main__":
    planet = Eros()
    gravity_model = generate_heterogeneous_model(planet, planet.obj_8k)

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

    vis.save(plt.gcf(), "eros_hetero_density")

    plt.show()
