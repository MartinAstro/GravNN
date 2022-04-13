import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np


class PlanesDist(TrajectoryBase):
    def __init__(self, celestial_body, bounds, samples_1d, **kwargs):
        """Driscoll and Heeley grid (Uniformally spaced at constant radius).
        Used to ensure that there is a properly dense distribution to observe a
        particular degree harmonic on a 2d image.

        This distribution is generally used to produce 2D gravity maps / grids
        using the MapVisualization module.

        Args:
            celestial_body (CelestialBody): planet about which the grid will be placed
            radius (float): radius at which the grid should be placed (typically Brillouin)
            degree (int): The maximum degree that should be observed
        """
        self.celestial_body = celestial_body
        self.bounds = bounds
        self.samples_1d = samples_1d

        super().__init__(**kwargs)
        pass

    def generate_full_file_directory(self):
        self.trajectory_name = (
            os.path.splitext(os.path.basename(__file__))[0]
            + f"/{self.celestial_body.body_name}_Bounds_{self.bounds}_{self.samples_1d}"
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
        xy_x, xy_y = np.meshgrid(X,Y)
        xz_x, xz_z = np.meshgrid(X,Z)
        yz_y, yz_z = np.meshgrid(Y,Z)

        xy_x = xy_x.reshape((-1,1))
        xy_y = xy_y.reshape((-1,1))
        xz_x = xz_x.reshape((-1,1))
        xz_z = xz_z.reshape((-1,1))
        yz_y = yz_y.reshape((-1,1))
        yz_z = yz_z.reshape((-1,1))

        zeros = np.zeros_like(xy_x)

        planes = np.block([
            [xy_x, xy_y, zeros],
            [xz_x, zeros, xz_z],
            [zeros, yz_y, yz_z],

        ])
        positions = planes
        self.positions = positions
        return positions
