import os

import numpy as np

from GravNN.Trajectories.TrajectoryBase import TrajectoryBase


class GaussianDist(TrajectoryBase):
    def __init__(self, celestial_body, radius_bounds, points, **kwargs):
        """Distribution drawn from a gaussian density profile.

        Args:
            celestial_body (CelestialBody): planet about which samples are taken
            radius_bounds (list): lower and upper limits for the distribution [low, high]
            points (int): number of samples to be drawn
            mu (float): center of the distribution
            sigma (float): 1-sigma value for the gaussian distribution
        """
        self.radius_bounds = radius_bounds
        self.points = points
        self.celestial_body = celestial_body
        self.mu = kwargs["mu"][0]
        self.sigma = kwargs["sigma"][0]
        super().__init__()
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
            + "_mu"
            + str(self.mu)
            + "_sigma"
            + str(self.sigma)
        )
        self.file_directory += self.trajectory_name + "/"
        pass

    def generate(self):
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
            r = np.random.normal(self.mu, self.sigma)
            while r > self.radius_bounds[1] or r < self.radius_bounds[0]:
                r = np.random.normal(self.mu, self.sigma)

            X[idx] = r * np.sin(phi) * np.cos(theta)
            Y[idx] = r * np.sin(phi) * np.sin(theta)
            Z[idx] = r * np.cos(phi)
            idx += 1
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
