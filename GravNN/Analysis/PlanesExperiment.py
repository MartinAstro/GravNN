import os

import numpy as np
import tensorflow as tf
import trimesh

from GravNN.Analysis.ExperimentBase import ExperimentBase
from GravNN.Networks.Data import DataSet
from GravNN.Networks.Losses import get_loss_fcn
from GravNN.Support.PathTransformations import make_windows_path_posix
from GravNN.Support.ProgressBar import ProgressBar
from GravNN.Trajectories.PlanesDist import PlanesDist


class PlanesExperiment(ExperimentBase):
    def __init__(self, model, config, bounds, samples_1d, **kwargs):
        super().__init__(model, config, bounds, samples_1d, **kwargs)
        self.config = config
        self.model = model
        self.bounds = np.array(bounds)
        self.samples_1d = samples_1d
        self.model_data_loaded = False

        self.brillouin_radius = config["planet"][0].radius
        original_max_radius = self.config["radius_max"][0]
        extra_max_radius = self.config.get("extra_radius_max", [0])[0]
        max_radius = np.max([original_max_radius, extra_max_radius])
        self.training_bounds = np.array([-max_radius, max_radius])

        if kwargs.get("remove_error", True):
            self.config["acc_noise"] = [0.0]

        self.loss_fcn_list = ["rms", "percent", "angle", "magnitude"]

    def get_train_data(self):
        data = DataSet(self.config)
        self.x_train = data.raw_data["x_train"]
        self.a_train = data.raw_data["a_train"]

    def get_test_data(self):
        if not hasattr(self, "x_test"):
            planet = self.config["planet"][0]
            obj_file = self.config.get("obj_file", [None])[0]
            gravity_data_fcn = self.config["gravity_data_fcn"][0]
            interpolation_dist = PlanesDist(
                planet,
                bounds=self.bounds,
                samples_1d=self.samples_1d,
                **self.config,
            )

            full_dist = interpolation_dist

            x, a, u = gravity_data_fcn(full_dist, obj_file, **self.config)

            self.x_test = x
            self.a_test = a
            self.u_test = u

    def get_model_data(self):
        if not hasattr(self, "a_pred"):
            try:
                dtype = self.model.network.compute_dtype
            except Exception:
                dtype = float
            positions = self.x_test.astype(dtype)
            self.a_pred = tf.cast(
                self.model.compute_acceleration(positions),
                dtype=dtype,
            )
            self.u_pred = tf.cast(self.model.compute_potential(positions), dtype)

            self.a_pred = self.a_pred.numpy().astype(float)
            self.u_pred = self.u_pred.numpy().astype(float)

            # Traditional Network Doesn't have U
            u_pred_2D = self.u_pred.ndim == 2
            if u_pred_2D:
                u_pred_TNN = np.shape(self.u_pred)[1] == 3
                if u_pred_TNN:
                    self.u_pred = self.u_pred[:, 0] * np.nan

    def compute_percent_error(self):
        def percent_error(x_hat, x_true):
            diff_mag = np.linalg.norm(x_true - x_hat, axis=1)
            true_mag = np.linalg.norm(x_true, axis=1)
            percent_error = diff_mag / true_mag * 100
            return percent_error

        self.percent_error_acc = percent_error(self.a_pred, self.a_test)
        self.percent_error_pot = percent_error(
            self.u_pred.reshape((-1, 1)),
            self.u_test.reshape((-1, 1)),
        )

        # nan out interior
        self.percent_error_acc[self.interior_mask] = np.nan
        self.percent_error_pot[self.interior_mask] = np.nan

    def compute_RMS(self):
        def RMS(x_hat, x_true):
            return np.sqrt(np.sum(np.square(x_true - x_hat), axis=1))

        self.RMS_acc = RMS(self.a_pred, self.a_test)
        self.RMS_pot = RMS(
            self.u_pred.reshape((-1, 1)),
            self.u_test.reshape((-1, 1)),
        )

        self.RMS_acc[self.interior_mask] = np.nan
        self.RMS_pot[self.interior_mask] = np.nan

    def compute_losses(self, loss_fcn_list):
        losses = {}
        for loss_key in loss_fcn_list:
            loss_fcn = get_loss_fcn(loss_key)
            key = f"{loss_fcn.__name__}"
            values = loss_fcn(self.a_pred, self.a_test).numpy()
            values[self.interior_mask] = np.nan
            # Compute loss on acceleration and potential
            losses.update({key: values})
        self.losses = losses

    def get_planet_mask(self):
        # Don't recompute this
        if not hasattr(self, "interior_mask"):
            # asteroids obj_file is the shape model
            obj_file = self.config.get("obj_file", [None])[0]
            # planets have shape model (sphere currently)
            self.obj_file = self.config.get("obj_file", [obj_file])[0]
            self.obj_file = make_windows_path_posix(self.obj_file)
            filename, file_extension = os.path.splitext(self.obj_file)
            self.obj_mesh = trimesh.load_mesh(
                self.obj_file,
                file_type=file_extension[1:],
            )

            N = len(self.x_test)
            step = 100
            mask = np.full((N,), False)
            pbar = ProgressBar(N, True)
            rayObject = trimesh.ray.ray_triangle.RayMeshIntersector(self.obj_mesh)
            for i in range(0, N, step):
                end_idx = (i // step + 1) * step
                position_subset = self.x_test[i:end_idx] / 1e3
                mask[i:end_idx] = rayObject.contains_points(position_subset)
                pbar.update(i)
            pbar.close()
            self.interior_mask = mask
        return self.interior_mask

    def generate_data(self):
        self.get_test_data()
        self.get_model_data()
        self.get_planet_mask()
        self.compute_percent_error()
        self.compute_RMS()
        self.compute_losses(self.loss_fcn_list)
        data = {
            "interior_mask": self.interior_mask,
            "a_pred": self.a_pred,
            "u_pred": self.u_pred,
        }
        return data

    def load_model_data(self, model):
        planet = self.config["planet"][0]

        interpolation_dist = PlanesDist(
            planet,
            bounds=self.bounds,
            samples_1d=self.samples_1d,
            **self.config,
        )
        model.trajectory = interpolation_dist
        model.configure(interpolation_dist)
        model.load()
        self.x_pred = model.trajectory.positions
        self.u_pred = model.potentials
        self.a_pred = model.accelerations

        self.model_data_loaded = True


def main():
    import matplotlib.pyplot as plt
    import pandas as pd

    from GravNN.Networks.Model import load_config_and_model
    from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

    # df = pd.read_pickle("Data/Dataframes/eros_poly_071123.data")
    df = pd.read_pickle("Data/Dataframes/heterogeneous_asymmetric_080523.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(df, model_id)

    planet = config["planet"][0]
    points = 100
    radius_bounds = [-2 * planet.radius, 2 * planet.radius]
    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        points,
        remove_error=True,
    )
    planes_exp.run()

    vis = PlanesVisualizer(planes_exp)
    vis.plot(z_max=10, annotate_stats=True)

    plt.show()


def simple():
    import matplotlib.pyplot as plt
    import pandas as pd

    from GravNN.Networks.Model import load_config_and_model
    from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

    df = pd.read_pickle("Data/Dataframes/eros_poly_071123.data")
    # df = pd.read_pickle("Data/Dataframes/heterogeneous_asymmetric_080523.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(df, model_id)

    planet = config["planet"][0]
    points = 10
    radius_bounds = [-2 * planet.radius, 2 * planet.radius]
    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        points,
    )
    planes_exp.run()

    vis = PlanesVisualizer(planes_exp)
    vis.plot(z_max=10, annotate_stats=True)

    plt.show()


if __name__ == "__main__":
    # main()
    simple()
