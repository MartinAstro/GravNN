import copy

import numpy as np

from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.PathTransformations import make_windows_path_posix


def get_hetero_poly_data(trajectory, obj_file, **kwargs):
    override = bool(kwargs.get("override", [False])[0])
    remove_point_mass = bool(kwargs.get("remove_point_mass", [False])[0])

    obj_file = make_windows_path_posix(obj_file)

    poly_r0_gm = HeterogeneousPoly(
        trajectory.celestial_body,
        obj_file,
        trajectory=trajectory,
    )

    # Force the following mass inhomogeneity
    mass_1 = copy.deepcopy(trajectory.celestial_body)
    mass_1.mu = mass_1.mu / 10
    r_offset_1 = [mass_1.radius / 3, 0, 0]

    mass_2 = copy.deepcopy(trajectory.celestial_body)
    mass_2.mu = -mass_2.mu / 10
    r_offset_2 = [-mass_2.radius / 3, 0, 0]

    point_mass_1 = PointMass(mass_1, trajectory)
    point_mass_2 = PointMass(mass_2, trajectory)

    poly_r0_gm.add_point_mass(point_mass_1, r_offset_1)
    poly_r0_gm.add_point_mass(point_mass_2, r_offset_2)

    poly_r0_gm.load(override=override)

    x = poly_r0_gm.positions  # position (N x 3)
    a = poly_r0_gm.accelerations
    u = np.array([poly_r0_gm.potentials]).transpose()  # potential (N x 1)

    # TODO: Determine if this is valuable -- how do dynamics and representation change
    # inside brillouin sphere
    if remove_point_mass:
        point_mass_r0_gm = PointMass(trajectory.celestial_body, trajectory=trajectory)
        point_mass_r0_gm.load(override=override)
        a_pm = point_mass_r0_gm.accelerations
        u_pm = np.array([point_mass_r0_gm.potentials]).transpose()

        a = a - a_pm
        u = u - u_pm

    return x, a, u


class HeterogeneousPoly(Polyhedral):
    def __init__(self, celestial_body, obj_file, trajectory=None):
        super().__init__(celestial_body, obj_file, trajectory)

        self.point_mass_list = []
        self.offset_list = []

    def add_point_mass(self, point_mass, r_offset):
        self.point_mass_list.append(point_mass)
        self.offset_list.append(r_offset)

    def load(self, override=False):
        # get homoegenous poly accelerations
        super().load(override)
        self.a_poly = copy.deepcopy(self.accelerations)
        self.u_poly = copy.deepcopy(self.potentials)

        x = self.positions
        a_poly = self.accelerations
        u_poly = self.potentials

        for i in range(len(self.point_mass_list)):
            r_offset = self.offset_list[i]
            x_pm = x - r_offset

            a_pm = self.point_mass_list[i].compute_acceleration(x_pm)
            u_pm = self.point_mass_list[i].compute_potential(x_pm)

            a_poly += a_pm
            u_poly += u_pm

        self.accelerations = a_poly
        self.potentials = u_poly

    def compute_acceleration(self, x=None):
        a_poly = super().compute_acceleration(x)

        for i in range(len(self.point_mass_list)):
            r_offset = np.array(self.offset_list[i]).reshape((-1, 3))
            x_pm = x - r_offset
            a_pm = self.point_mass_list[i].compute_acceleration(x_pm)
            a_poly += a_pm
        return a_poly

    def compute_potentials(self, x=None):
        u_poly = super().compute_potential(x)

        for i in range(len(self.point_mass_list)):
            r_offset = self.offset_list[i]
            x_pm = x - r_offset
            u_pm = self.point_mass_list[i].compute_potential(x_pm)
            u_poly += u_pm
        return u_poly


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.Trajectories.SurfaceDist import SurfaceDist

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
    vis.plot_polyhedron(planet.obj_8k, gravity_model.accelerations, cmap="bwr")
    plt.gca().scatter(r_offset_1[0], r_offset_1[1], r_offset_1[2], s=400)
    plt.gca().scatter(r_offset_2[0], r_offset_2[1], r_offset_2[2], s=400)
    vis.plot_polyhedron(planet.obj_8k, gravity_model.a_poly, cmap="bwr")

    plt.show()
