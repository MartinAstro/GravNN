import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np
import trimesh
from numba import njit, prange
from GravNN.Support.ProgressBar import ProgressBar

class RandomAsteroidDist(TrajectoryBase):
    def __init__(
        self, celestial_body, radius_bounds, points, model_file=None, **kwargs
    ):
        """A sample distribution which can sample from altitudes all the way down to the surface of the body.

        This is unlike the RandomDist class which samples randomly between the radius bounds without accounting
        for if the point exists within the body or not. As such this is generally most useful when generating
        distributions around irregularly shaped asteroids / bodies.

        Args:
            celestial_body (Celestial Body): planet about which samples are collected
            radius_bounds (list): upper and lower altitude bounds
            points (int): total number of samples
            model_file (str, optional): The path to the shape model. Defaults to None.
        """
        self.radius_bounds = radius_bounds
        self.model_file = kwargs.get("grav_file", [model_file])[0]
        filename, file_extension = os.path.splitext(self.model_file)

        self.shape_model = trimesh.load_mesh(self.model_file, file_type=file_extension[1:])
        self.points = points
        self.celestial_body = celestial_body

        # Reduce the search space to minimum altitude if it exists on the planet
        if self.radius_bounds[0] == 0.0:
            try: 
                self.radius_bounds[0] = celestial_body.min_radius
            except:
                pass

        super().__init__()

        pass

    def generate_full_file_directory(self):
        """Define the output directory based on number of points sampled,
        the radius/altitude limits, and the shape model used
        """
        try:
            model_name = os.path.basename(self.model_file).split('.')[0]
        except:
            model_name = str(self.model_file.split("Blender_")[1]).split(".")[0]
        self.trajectory_name = (
            os.path.splitext(os.path.basename(__file__))[0]
            + "/"
            + self.celestial_body.body_name
            + "N_"
            + str(self.points)
            + "_RadBounds"
            + str(self.radius_bounds)
            + "_shape_model"
            + model_name
        )
        self.file_directory += self.trajectory_name + "/"
        pass

    def generate(self):
        """Generate samples from uniform lat, lon, and radial distributions, but also check
        that those samples exist above the surface of the shape model. If not, resample the radial component

        Returns:
            np.array: cartesian positions of samples
        """
        X = []
        Y = []
        Z = []
        idx = 0
        X.extend(np.zeros((self.points,)).tolist())
        Y.extend(np.zeros((self.points,)).tolist())
        Z.extend(np.zeros((self.points,)).tolist())
        pbar = ProgressBar(self.points, enable=True)
        while idx < self.points:
            phi = np.random.uniform(0, np.pi)
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(self.radius_bounds[0], self.radius_bounds[1])
            X_inst = r * np.sin(phi) * np.cos(theta)
            Y_inst = r * np.sin(phi) * np.sin(theta)
            Z_inst = r * np.cos(phi)

            distance = self.shape_model.nearest.signed_distance(
                np.array([[X_inst, Y_inst, Z_inst]]) / 1e3
            )
            # ensure that the point is outside of the body
            i = 0
            while distance > 0:
                # Note that this loop my get stuck if the radius bounds do not extend beyond the body
                # (i.e. the RA and Dec are fixed so if the upper bound does not extend beyond the shape
                # this criteria is never satisfied)
                r = np.random.uniform(self.radius_bounds[0], self.radius_bounds[1])
                X_inst = r * np.sin(phi) * np.cos(theta)
                Y_inst = r * np.sin(phi) * np.sin(theta)
                Z_inst = r * np.cos(phi)
                distance = self.shape_model.nearest.signed_distance(
                    np.array([[X_inst, Y_inst, Z_inst]]) / 1e3
                )
                i += 1
                # Try to keep value at same RA and Dec, but if it can't find any then just select a new position entirely. 
                if i > 10:
                    phi = np.random.uniform(0, np.pi)
                    theta = np.random.uniform(0, 2 * np.pi)
                    i = 0
            X[idx] = X_inst
            Y[idx] = Y_inst
            Z[idx] = Z_inst
            idx += 1
            pbar.update(idx)
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
