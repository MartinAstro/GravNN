import numpy as np

from GravNN.Networks.Losses import *
from GravNN.Support.PathTransformations import make_windows_path_posix
from GravNN.Trajectories.RandomDist import RandomDist


class DataInterface:
    def __init__(
        self,
        model,
        config,
        points,
        radius_bounds=None,
        random_seed=1234,
        remove_J2=False,
    ):
        self.config = config.copy()
        self.model = model
        self.points = points
        self.radius_bounds = radius_bounds
        self.remove_J2 = remove_J2
        self.distribution = config["distribution"][0]

        if self.radius_bounds is None:
            min_radius = self.config["radius_min"][0]
            max_radius = self.config["radius_max"][0]

            # If training data was augmented data take the largest radius
            augment_data = self.config.get("augment_data_config", [{}])[0]
            extra_max_radius = augment_data.get("radius_max", [0])[0]
            max_radius = np.max([max_radius, extra_max_radius])

            self.radius_bounds = [min_radius, max_radius]

        # attributes to be populated in run()
        self.positions = None
        self.accelerations = None
        self.potentials = None

        self.a_pred = None
        self.u_pred = None

        np.random.seed(random_seed)

    def get_data(self):
        planet = self.config["planet"][0]
        obj_file = self.config["obj_file"][0]

        obj_file = make_windows_path_posix(obj_file)

        trajectory = self.distribution(
            planet,
            self.radius_bounds,
            self.points,
            **self.config,
        )
        get_analytic_data_fcn = self.config["gravity_data_fcn"][0]

        x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(
            trajectory,
            obj_file,
            **self.config,
        )

        self.positions = x_unscaled
        self.accelerations = a_unscaled
        self.potentials = u_unscaled

    def get_PINN_data(self):
        positions = self.positions
        self.a_pred = self.model.compute_acceleration(positions).numpy().astype(float)
        self.u_pred = self.model.compute_potential(positions).numpy().astype(float)

    def get_J2_data(self):
        planet = self.config["planet"][0]
        obj_file = self.config["obj_file"][0]

        trajectory = RandomDist(planet, self.radius_bounds, self.points, **self.config)
        get_analytic_data_fcn = self.config["gravity_data_fcn"][0]
        config_mod = self.config.copy()
        config_mod["max_deg"] = [2]

        x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(
            trajectory,
            obj_file,
            **config_mod,
        )

        if self.remove_J2:
            # Low Fidelity
            self.LF_accelerations = a_unscaled
            self.LF_potentials = u_unscaled
        else:
            self.LF_accelerations = np.zeros_like(x_unscaled)
            self.LF_potentials = np.zeros_like(x_unscaled[:, 0:1])

    def gather_data(self):
        self.get_data()
        self.get_PINN_data()
        self.get_J2_data()
