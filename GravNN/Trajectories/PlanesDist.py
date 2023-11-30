import os

import numpy as np

from GravNN.Trajectories.TrajectoryBase import TrajectoryBase


class PlanesDist(TrajectoryBase):
    def __init__(self, celestial_body, bounds, samples_1d, **kwargs):
        self.celestial_body = celestial_body
        self.bounds = bounds
        self.samples_1d = samples_1d

        super().__init__(**kwargs)
        pass

    def generate_full_file_directory(self):
        # Often saved with .0 in the bounds on linux, that prevents proper loading
        # This is a hack to remove that
        bounds_str = str(self.bounds)
        if ".0]" in bounds_str:
            bounds_str = bounds_str.replace(".0", ".")
            bounds_str = bounds_str.replace(",", " ")
        self.trajectory_name = (
            os.path.splitext(os.path.basename(__file__))[0]
            + f"/{self.celestial_body.body_name}_Bounds_{bounds_str}_{self.samples_1d}"
        )
        self.file_directory += self.trajectory_name + "/"
        pass

    def generate(self):
        """Sample the grid at uniform intervals defined by the maximum
        degree to be observed

        Returns:
            np.array: cartesian positions of samples_1d
        """
        X, Y, Z = [], [], []
        X = np.linspace(self.bounds[0], self.bounds[1], self.samples_1d)
        Y = np.linspace(self.bounds[0], self.bounds[1], self.samples_1d)
        Z = np.linspace(self.bounds[0], self.bounds[1], self.samples_1d)
        xy_x, xy_y = np.meshgrid(X, Y)
        xz_x, xz_z = np.meshgrid(X, Z)
        yz_y, yz_z = np.meshgrid(Y, Z)

        xy_x = xy_x.reshape((-1, 1))
        xy_y = xy_y.reshape((-1, 1))
        xz_x = xz_x.reshape((-1, 1))
        xz_z = xz_z.reshape((-1, 1))
        yz_y = yz_y.reshape((-1, 1))
        yz_z = yz_z.reshape((-1, 1))

        zeros = np.zeros_like(xy_x)

        planes = np.block(
            [
                [xy_x, xy_y, zeros],
                [xz_x, zeros, xz_z],
                [zeros, yz_y, yz_z],
            ],
        )
        positions = planes
        self.positions = positions
        return positions
