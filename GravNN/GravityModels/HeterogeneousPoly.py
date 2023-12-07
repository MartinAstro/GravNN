import copy
import dataclasses
import os

import numpy as np

from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.PathTransformations import make_windows_path_posix


def get_hetero_poly_data(trajectory, obj_shape_file, **kwargs):
    override = bool(kwargs.get("override", [False])[0])
    remove_point_mass = bool(kwargs.get("remove_point_mass", [False])[0])

    obj_shape_file = make_windows_path_posix(obj_shape_file)

    poly_r0_gm = generate_heterogeneous_model(
        trajectory.celestial_body,
        obj_shape_file,
        trajectory=trajectory,
    )

    poly_r0_gm.load(override=override)

    x = poly_r0_gm.positions  # position (N x 3)
    a = poly_r0_gm.accelerations
    u = poly_r0_gm.potentials  # potential (N,)

    # TODO: Determine if this is valuable -- how do dynamics and representation change
    # inside brillouin sphere
    if remove_point_mass:
        point_mass_r0_gm = PointMass(trajectory.celestial_body, trajectory=trajectory)
        point_mass_r0_gm.load(override=override)
        a_pm = point_mass_r0_gm.accelerations
        u_pm = point_mass_r0_gm.potentials

        a = a - a_pm
        u = u - u_pm

    return x, a, u


def generate_heterogeneous_model(planet, obj_file, trajectory=None, symmetric=False):
    if symmetric:
        return generate_symmetric_heterogeneous_model(planet, obj_file, trajectory)
    else:
        return generate_asymmetric_heterogeneous_model(planet, obj_file, trajectory)


def generate_asymmetric_heterogeneous_model(planet, obj_file, trajectory=None):
    # Force the following mass inhomogeneity

    mass_1 = copy.deepcopy(planet)
    mass_1.mu = mass_1.mu / 10
    r_offset_1 = [mass_1.radius / 2, 0, 0]

    mass_2 = copy.deepcopy(planet)
    mass_2.mu = -mass_2.mu / 10
    r_offset_2 = [-mass_2.radius / 2, 0, 0]

    point_mass_1 = PointMass(mass_1)
    point_mass_2 = PointMass(mass_2)

    mascon_1 = Heterogeneity(point_mass_1, r_offset_1)
    mascon_2 = Heterogeneity(point_mass_2, r_offset_2)

    heterogeneities = [mascon_1, mascon_2]
    poly_r0_gm = HeterogeneousPoly(
        planet,
        obj_file,
        heterogeneities,
        trajectory=trajectory,
    )

    return poly_r0_gm


def generate_symmetric_heterogeneous_model(planet, obj_file, trajectory=None):
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

    mascon_0 = Heterogeneity(point_mass_0, r_offset_0)
    mascon_1 = Heterogeneity(point_mass_1, r_offset_1)
    mascon_2 = Heterogeneity(point_mass_2, r_offset_2)

    heterogeneities = [mascon_0, mascon_1, mascon_2]
    poly_r0_gm = HeterogeneousPoly(
        planet,
        obj_file,
        heterogeneities,
        trajectory=trajectory,
    )

    return poly_r0_gm


@dataclasses.dataclass
class Heterogeneity:
    model: PointMass
    r_offset: np.ndarray


class HeterogeneousPoly(GravityModelBase):
    def __init__(self, celestial_body, obj_file, heterogeneities, trajectory=None):
        self.homogeneous_poly = Polyhedral(celestial_body, obj_file, trajectory)
        self.planet = celestial_body
        self.obj_file = obj_file
        self.point_mass_list = []
        self.offset_list = []

        for heterogeneity in heterogeneities:
            self.point_mass_list.append(heterogeneity.model)
            self.offset_list.append(heterogeneity.r_offset)

        homo_id = self.homogeneous_poly.id

        # specify the lists as args to make unique hash.
        super().__init__(
            celestial_body,
            obj_file,
            homo_id,
            self.offset_list,
            self.point_mass_list,
            trajectory,
        )
        self.configure(trajectory)

    def add_point_mass(self, point_mass, r_offset):
        self.point_mass_list.append(point_mass)
        self.offset_list.append(r_offset)

    def generate_full_file_directory(self):
        # unique identifier takes the mu + offset of each point mass
        unique_str = ""
        class_name = self.__class__.__name__
        obj_file = os.path.basename(self.obj_file).split(".")[0]
        for i in range(len(self.point_mass_list)):
            mu_str = f"{self.point_mass_list[i].celestial_body.mu}"
            offset_str = f"{self.offset_list[i]}"
            unique_str += f"{mu_str}_{offset_str}_"
        self.file_directory += f"{class_name}_{obj_file}_{unique_str}/"

    def load(self, override=False):
        # If heterogeneous model exists, load it
        data_exists = os.path.exists(self.file_directory + "acceleration.data")
        if data_exists:
            # you need to load the homogeneous solution
            self.homogeneous_poly.trajectory = self.trajectory
            self.homogeneous_poly.configure(self.trajectory)
            self.homogeneous_poly.load()

            # now load the heterogeneous solution
            super().load(override)

        # If not, generate / load the homogeneous data first,
        # then add point mass contributions
        elif not data_exists:
            data_dir = os.path.relpath(self.file_directory)
            print(f"Generating accelerations + potentials at {data_dir}")

            self.homogeneous_poly.load()
            accelerations = self.homogeneous_poly.accelerations
            potentials = self.homogeneous_poly.potentials

            # add the point mass contributions to the homogeneous solution
            positions = self.positions
            for i in range(len(self.point_mass_list)):
                r_offset = np.array(self.offset_list[i]).reshape((-1, 3))
                x_pm = positions - r_offset
                x_pm_2D = x_pm.reshape((-1, 3))
                u_pm = self.point_mass_list[i].compute_potential(x_pm_2D)
                a_pm = self.point_mass_list[i].compute_acceleration(x_pm_2D)
                accelerations += a_pm
                potentials += u_pm

            self.accelerations = accelerations
            self.potentials = potentials
            self.save()

    def compute_acceleration(self, positions=None):
        if positions is None:
            positions = self.trajectory.positions

        a_poly = self.homogeneous_poly.compute_acceleration(positions.reshape((-1, 3)))

        for i in range(len(self.point_mass_list)):
            r_offset = np.array(self.offset_list[i]).reshape((-1, 3))
            x_pm = positions - r_offset
            a_pm = self.point_mass_list[i].compute_acceleration(x_pm.reshape((-1, 3)))
            a_poly += a_pm
        return a_poly

    def compute_potential(self, positions=None):
        if positions is None:
            positions = self.trajectory.positions

        u_poly = self.homogeneous_poly.compute_potential(positions)

        for i in range(len(self.point_mass_list)):
            r_offset = self.offset_list[i]
            x_pm = positions - r_offset
            u_pm = self.point_mass_list[i].compute_potential(x_pm)
            u_poly += u_pm
        return u_poly


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.Trajectories.SurfaceDist import SurfaceDist

    planet = Eros()

    traj = SurfaceDist(planet, planet.obj_8k)

    mass_1 = Eros()
    mass_1.mu = mass_1.mu / 10
    r_offset_1 = [mass_1.radius / 3, 0, 0]

    mass_2 = Eros()
    mass_2.mu = -mass_2.mu / 10
    r_offset_2 = [-mass_2.radius / 3, 0, 0]

    point_mass_1 = PointMass(mass_1)
    point_mass_2 = PointMass(mass_2)

    mascon_1 = Heterogeneity(point_mass_1, r_offset_1)
    mascon_2 = Heterogeneity(point_mass_2, r_offset_2)
    heterogeneities = [mascon_1, mascon_2]

    gravity_model = HeterogeneousPoly(planet, planet.obj_8k, heterogeneities, traj)

    gravity_model.load()

    from GravNN.Visualization.PolyVisualization import PolyVisualization

    vis = PolyVisualization()
    vis.plot_polyhedron(planet.obj_8k, gravity_model.accelerations, cmap="bwr")
    plt.gca().scatter(r_offset_1[0], r_offset_1[1], r_offset_1[2], s=400)
    plt.gca().scatter(r_offset_2[0], r_offset_2[1], r_offset_2[2], s=400)
    vis.plot_polyhedron(
        planet.obj_8k,
        gravity_model.homogeneous_poly.accelerations,
        cmap="bwr",
    )

    plt.show()
