import matplotlib.pyplot as plt

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import HeterogeneousPoly
from GravNN.GravityModels.PointMass import PointMass
from GravNN.Trajectories.SurfaceDist import SurfaceDist

if __name__ == "__main__":
    planet = Eros()

    traj = SurfaceDist(planet, planet.obj_8k)
    gravity_model = HeterogeneousPoly(planet, planet.obj_8k, traj)

    mass_1 = Eros()
    mass_1.mu = planet.mu / 10
    r_offset_1 = [planet.radius / 3, 0, 0]

    mass_2 = Eros()
    mass_2.mu = -planet.mu / 10
    r_offset_2 = [-planet.radius / 3, 0, 0]

    point_mass_1 = PointMass(mass_1, traj)
    point_mass_2 = PointMass(mass_2, traj)

    gravity_model.add_point_mass(point_mass_1, r_offset_1)
    gravity_model.add_point_mass(point_mass_2, r_offset_2)

    gravity_model.load()

    from GravNN.Visualization.PolyVisualization import PolyVisualization

    vis = PolyVisualization()
    vis.plot_polyhedron(planet.obj_8k, None, cmap="Greys", cbar=False, alpha=0.1)
    plt.gca().scatter(r_offset_1[0], r_offset_1[1], r_offset_1[2], s=200, color="red")
    plt.gca().scatter(r_offset_2[0], r_offset_2[1], r_offset_2[2], s=200, color="blue")

    plt.gca().set_xticklabels("")
    plt.gca().set_yticklabels("")
    plt.gca().set_zticklabels("")
    plt.gca().view_init(elev=35, azim=180 + 45, roll=0)

    vis.save(plt.gcf(), "eros_hetero_density.pdf")

    plt.show()
