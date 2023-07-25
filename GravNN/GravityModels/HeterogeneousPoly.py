import copy
import os

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


def get_hetero_poly_symmetric_data(trajectory, obj_file, **kwargs):
    override = bool(kwargs.get("override", [False])[0])
    remove_point_mass = bool(kwargs.get("remove_point_mass", [False])[0])

    obj_file = make_windows_path_posix(obj_file)

    poly_r0_gm = generate_heterogeneous_sym_model(
        trajectory.celestial_body,
        obj_file,
        trajectory=trajectory,
    )

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

    def generate_full_file_directory(self):
        self.file_directory += (
            os.path.splitext(os.path.basename(__file__))[0]
            + "_"
            + os.path.basename(self.obj_file).split(".")[0]
            + "/"
        )
        pass

    def load(self, override=False):
        # get homoegenous poly accelerations
        super().load(override)
        self.a_poly = copy.deepcopy(self.accelerations)
        self.u_poly = copy.deepcopy(self.potentials)

        # x = self.positions
        # a_poly = self.accelerations
        # u_poly = self.potentials

        # for i in range(len(self.point_mass_list)):
        #     r_offset = self.offset_list[i]
        #     x_pm = x - r_offset

        #     a_pm = self.point_mass_list[i].compute_acceleration(x_pm)
        #     u_pm = self.point_mass_list[i].compute_potential(x_pm)

        #     a_poly += a_pm
        #     u_poly += u_pm

        # self.accelerations = a_poly
        # self.potentials = u_poly

    def compute_acceleration(self, positions=None):
        if positions is None:
            positions = self.trajectory.positions

        a_poly = super().compute_acceleration(positions.reshape((-1, 3)))

        for i in range(len(self.point_mass_list)):
            r_offset = np.array(self.offset_list[i]).reshape((-1, 3))
            x_pm = positions - r_offset
            a_pm = self.point_mass_list[i].compute_acceleration(x_pm.reshape((-1, 3)))
            a_poly += a_pm
        return a_poly

    def compute_potential(self, positions=None):
        if positions is None:
            positions = self.trajectory.positions

        u_poly = super().compute_potential(positions)

        for i in range(len(self.point_mass_list)):
            r_offset = self.offset_list[i]
            x_pm = positions - r_offset
            u_pm = self.point_mass_list[i].compute_potential(x_pm)
            u_poly += u_pm
        return u_poly


def generate_heterogeneous_sym_model(planet, shape_model, trajectory=None):
    poly_r0_gm = HeterogeneousPoly(planet, shape_model, trajectory=trajectory)

    # Force the following mass inhomogeneity
    mass_0 = copy.deepcopy(planet)
    mass_0.mu = -2 * mass_0.mu / 20
    r_offset_0 = [0, 0, 0]

    mass_1 = copy.deepcopy(planet)
    mass_1.mu = mass_1.mu / 20
    r_offset_1 = [mass_1.radius / 2, 0, 0]

    mass_2 = copy.deepcopy(planet)
    mass_2.mu = mass_2.mu / 20
    r_offset_2 = [-mass_2.radius / 2, 0, 0]

    point_mass_0 = PointMass(mass_0)
    point_mass_1 = PointMass(mass_1)
    point_mass_2 = PointMass(mass_2)

    poly_r0_gm.add_point_mass(point_mass_0, r_offset_0)
    poly_r0_gm.add_point_mass(point_mass_1, r_offset_1)
    poly_r0_gm.add_point_mass(point_mass_2, r_offset_2)

    return poly_r0_gm


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.Trajectories.SurfaceDist import SurfaceDist

    planet = Eros()

    traj = SurfaceDist(planet, planet.obj_8k)
    gravity_model = HeterogeneousPoly(planet, planet.obj_8k, traj)

    mass_1 = Eros()
    mass_1.mu = mass_1.mu / 10
    r_offset_1 = [mass_1.radius / 3, 0, 0]

    mass_2 = Eros()
    mass_2.mu = -mass_2.mu / 10
    r_offset_2 = [-mass_2.radius / 3, 0, 0]

    point_mass_1 = PointMass(mass_1)
    point_mass_2 = PointMass(mass_2)

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
