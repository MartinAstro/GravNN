from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Trajectories import SurfaceDist, RandomAsteroidDist

import numpy as np


def get_spherical_data(x, a):
    x_sph = cart2sph(x)
    a_sph = project_acceleration(x_sph, np.array(a, dtype=float))
    return x_sph, a_sph


class AsteroidAnalyzer:
    def __init__(self, model, config, interior_bound, exterior_bound):
        """Class responsible for analyzing the performance of a PINN gravity model
        1) at the surface, 2) within the Brillouin sphere, and 3) outside the brillouin sphere

        Args:
            model (CustomModel): PINN gravity model to be evaluated
            config (dict): hyperparameter and configuration settings for model
            interior_bound (float): interior radius limit (typically the brillouin radius)
            exterior_bound (float): exterior radius limit (ex. LEO altitude for Earth)
        """
        self.config = config
        self.model = model
        self.planet = config["planet"][0]
        self.model_file = self.config["grav_file"][0]
        self.interior_bound = interior_bound
        self.exterior_bound = exterior_bound

    def compute_stats(self, trajectory, prefix):
        x, a, u = get_poly_data(trajectory, self.model_file, **self.config)

        # data_pred = self.model.generate_nn_data(x)
        a_pred = self.model.generate_acceleration(x.astype(np.float32))

        diff = a - a_pred
        diff_percent = np.abs((a - a_pred) / a) * 100.0

        rse = np.linalg.norm(diff, axis=1) ** 2
        percent = np.linalg.norm(diff_percent, axis=1)

        stats = {
            prefix + "_rse_mean": np.mean(rse),
            prefix + "_rse_std": np.std(rse),
            prefix + "_rse_max": np.max(rse),
            prefix + "_percent_mean": np.mean(percent),  # error
            prefix + "_percent_std": np.std(percent),  # error
            prefix + "_percent_max": np.max(percent),  # error
        }
        return stats

    def compute_surface_stats(self):
        trajectory = SurfaceDist(self.planet, self.model_file)
        stats = self.compute_stats(trajectory, "surface")
        return stats

    def compute_interior_stats(self):
        trajectory = RandomAsteroidDist(
            self.planet, [0, self.interior_bound], 50000, self.model_file
        )
        stats = self.compute_stats(trajectory, "interior")
        return stats

    def compute_exterior_stats(self):
        trajectory = RandomAsteroidDist(
            self.planet,
            [self.interior_bound, self.exterior_bound],
            50000,
            self.model_file,
        )
        stats = self.compute_stats(trajectory, "exterior")
        return stats
