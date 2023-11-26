import os

import numpy as np

import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.Support.transformations import cart2sph


def get_mascons_data(trajectory, gravity_file, **kwargs):
    # Handle cases where the keyword wasn't properly wrapped as a list []
    override = bool(kwargs.get("override", [False])[0])

    point_mass_r0_gm = Mascons(kwargs["planet"][0], trajectory=trajectory)
    accelerations = point_mass_r0_gm.load(override=override).accelerations
    potentials = point_mass_r0_gm.potentials

    x = point_mass_r0_gm.positions  # position (N x 3)
    a = accelerations
    u = potentials  # (N,)

    return x, a, u


class Mascons(GravityModelBase):
    def __init__(self, celestial_body, mass_csv, trajectory=None):
        """Gravity model that only produces accelerations and potentials
        as if there were only a point mass.

        Args:
            celestial_body (CelestialBody): body used to generate gravity measurements
            trajectory (TrajectoryBase, optional): trajectory for which gravity
            measurements must be produced. Defaults to None.
        """
        super().__init__(celestial_body, mass_csv, trajectory=trajectory)
        self.celestial_body = celestial_body
        self.mu = celestial_body.mu
        self.mass_csv = mass_csv
        self.read_mass_csv()
        self.configure(trajectory)

    def read_mass_csv(self):
        gravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__))
        # if the filename is absolute, read it from there
        if os.path.isabs(self.mass_csv):
            open_file = self.mass_csv
        else:
            open_file = (
                f"{gravNN_dir}/Files/GravityModels/Regressed/Mascons/" + self.mass_csv
            )
        with open(open_file, "r") as f:
            lines = f.readlines()
            lines = lines[1:]  # Skip header
            lines = [line.strip().split(",") for line in lines]
            lines = np.array(lines).astype(np.float64)
            self.masses_mu = lines[:, 0:1]
            self.masses_position = lines[:, 1:]

    def generate_full_file_directory(self):
        model_name = os.path.splitext(os.path.basename(__file__))[0]
        masses_file = os.path.splitext(os.path.basename(self.mass_csv))[0]
        self.file_directory += f"{model_name}_{masses_file}/"
        pass

    def compute_acceleration(self, positions=None):
        """Compute the acceleration for an existing trajectory or provided
        set of positions"""
        if positions is None:
            positions = self.trajectory.positions

        self.accelerations = np.zeros(positions.shape)
        for i in range(len(self.accelerations)):
            self.accelerations[i] = self.compute_acceleration_value(positions[i])

        return self.accelerations

    def compute_potential(self, positions=None):
        "Compute the potential for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions

        positions = cart2sph(positions)
        self.potentials = np.zeros(len(positions))
        for i in range(len(self.potentials)):
            self.potentials[i] = self.compute_potential_value(positions[i])

        return self.potentials

    def compute_acceleration_value(self, position):
        # remember that a = -dU/dx
        # U = -mu/r
        # dU/dx = mu/r^2
        # a = -dU/dx = -mu/r^2
        dr = position - self.masses_position
        dr_mag = np.linalg.norm(dr, axis=1, keepdims=True)
        a_mass_i = -self.masses_mu * dr / dr_mag**3
        a_mass = np.sum(a_mass_i, axis=0)
        return a_mass

    def compute_potential_value(self, position):
        dr = position - self.masses_position
        dr_mag = np.linalg.norm(dr, axis=1, keepdims=True)
        u_mass_i = -self.masses_mu / dr_mag
        u_mass = np.sum(u_mass_i)
        return u_mass


def main():
    import time

    start = time.time()
    planet = Eros()
    GravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__))
    mass_csv = f"{GravNN_dir}/../Data/Comparison/MASCONS_55_500_0.0.csv"
    mascons = Mascons(planet, mass_csv=mass_csv)
    print(time.time() - start)

    position = (
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]) * planet.radius
    )  # Must be in meters

    print(position)
    start = time.time()
    mascons.compute_acceleration(position)
    mascons.compute_potential(position)


if __name__ == "__main__":
    main()
