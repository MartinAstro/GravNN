import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np


class RandomDist(TrajectoryBase):
    def __init__(self, celestial_body, radius_bounds, points, **kwargs):
        """A sample distribution that randomly selects from uniform distributions in latitude, longitude, and radius.

        Args:
            celestial_body (Celestial Body): Planet about which samples should be taken
            radius_bounds (list): range of radii from which the sample can be drawn
            points (int): number of samples
        """
        if points % np.sqrt(points) != 0:
            print("The total number of points is not a perfect square")
            N = int(np.sqrt(points / 2))
            points = 2 * N ** 2
            print("The total number of points changed to " + str(points))
        self.radius_bounds = radius_bounds

        self.points = points
        self.celestial_body = celestial_body

        super().__init__(**kwargs)

        pass

    def generate_full_file_directory(self):
        self.trajectory_name = (
            os.path.splitext(os.path.basename(__file__))[0]
            + "/"
            + self.celestial_body.body_name
            + "N_"
            + str(self.points)
            + "_RadBounds"
            + str(self.radius_bounds)
        )
        self.file_directory += self.trajectory_name + "/"
        pass

    def generate(self):
        """Randomly sample from uniform latitude, longitude, and radial distributions

        Returns:
            np.array: cartesian positions of the samples
        """
        X = []
        Y = []
        Z = []
        idx = 0
        X.extend(np.zeros((self.points,)).tolist())
        Y.extend(np.zeros((self.points,)).tolist())
        Z.extend(np.zeros((self.points,)).tolist())

        for i in range(self.points):
            phi = np.random.uniform(0, np.pi)
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(self.radius_bounds[0], self.radius_bounds[1])
            X[idx] = r * np.sin(phi) * np.cos(theta)
            Y[idx] = r * np.sin(phi) * np.sin(theta)
            Z[idx] = r * np.cos(phi)
            idx += 1
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
