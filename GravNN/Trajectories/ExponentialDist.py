import os

import numpy as np

from GravNN.Trajectories.TrajectoryBase import TrajectoryBase


class ExponentialDist(TrajectoryBase):
    def __init__(self, celestial_body, radiusBounds, points, **kwargs):
        """Distribution with samples drawn from an exponential distribution.

        Useful for simulating samples drawn primarily from orbit, but with occasional samples closer to the surface.

        Args:
            celestial_body (CelestialBody): Planet about which these distributions will be drawn
            radiusBounds (list): limits of the exponential distribution
            points (int): number of samples to be drawn
            scale_parameter (float): b in 1/b*exp(-x/b) such that small scale parameter leads to narrower distributions
            invert (bool): invert the distribution such that samples decay in frequency from the higher to the lower altitudes
        """
        # scale_parameter = beta -- e^(x/beta)
        self.radiusBounds = radiusBounds
        self.points = points
        self.celestial_body = celestial_body
        self.scale_parameter = kwargs["scale_parameter"][
            0
        ]  # TODO: Make this a required parameter
        self.invert = kwargs["invert"][
            0
        ]  # if true, higher probabilities occur at higher altitude TODO: Make this a required param

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
            + str(self.radiusBounds)
            + "_scale"
            + str(self.scale_parameter)
            + "_invert"
            + str(self.invert)
        )
        self.file_directory += self.trajectory_name + "/"
        pass

    def generate(self):
        """Draw theta and phi from a uniform distribution, but then
        pull radius samples from an exponential distribution defined by
        the scale parameter.
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

            # If the curve is inverted, make sure the position isn't inside of the body.
            # If it is, resample until its within bounds.
            if self.invert:
                alt = -1.0 * np.random.exponential(self.scale_parameter)
                r = self.radiusBounds[1] + alt
                while r < self.radiusBounds[0]:
                    alt = -1.0 * np.random.exponential(self.scale_parameter)
                    r = self.radiusBounds[1] + alt
            else:
                alt = np.random.exponential(self.scale_parameter)
                r = self.radiusBounds[0] + alt

            X[idx] = r * np.sin(phi) * np.cos(theta)
            Y[idx] = r * np.sin(phi) * np.sin(theta)
            Z[idx] = r * np.cos(phi)
            idx += 1
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
