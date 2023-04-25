import os

import numpy as np
import trimesh

from GravNN.Networks.Data import DataSet
from GravNN.Networks.Losses import *
from GravNN.Support.transformations import cart2sph
from GravNN.Trajectories import RandomDist


class ExtrapolationExperiment:
    def __init__(self, model, config, points, random_seed=1234):
        self.config = config
        self.model = model
        self.points = points
        self.loss_fcn_list = ["rms", "percent"]

        self.brillouin_radius = config["planet"][0].radius
        original_max_radius = self.config["radius_max"][0]
        extra_max_radius = np.nan_to_num(self.config.get("extra_radius_max", [0])[0], 0)
        max_radius = np.max([original_max_radius, extra_max_radius])
        self.training_bounds = [config["radius_min"][0], max_radius]

        # attributes to be populated in run()
        self.positions = None
        self.test_accelerations = None
        self.test_potentials = None

        self.predicted_accelerations = None
        self.predicted_potentials = None

        self.percent_error_acc = None
        self.percent_error_pot = None

        np.random.seed(random_seed)

    def get_train_data(self):
        data = DataSet(self.config)
        x_train = data.raw_data["x_train"]
        train_r_COM = cart2sph(x_train)[:, 0]

        # sort
        self.train_dist_2_COM_idx = np.argsort(train_r_COM)
        self.train_r_COM = train_r_COM[self.train_dist_2_COM_idx]

        # asteroids grav_file is the shape model
        grav_file = self.config.get("grav_file", [None])[0]

        # planets have shape model (sphere currently)
        obj_file = self.config.get("shape_model", [grav_file])[0]

        # Compute distance to surface
        filename, file_extension = os.path.splitext(obj_file)
        mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
        x_train_km = x_train / 1000
        # _, train_r, _ = trimesh.proximity.closest_point(mesh, x_train_km)

        # compute the intersections of the mesh and the field point
        ray_origins = np.zeros_like(x_train_km)
        intersections, ray_idx, _ = mesh.ray.intersects_location(
            ray_origins,
            x_train_km,
        )
        # the intersections aren't in order of the input arrays
        # sort based on ray order
        intersections = intersections[np.argsort(ray_idx)] * 1000

        # compute the distance to the intersection point
        train_r = np.linalg.norm(intersections - x_train_km, axis=1)

        # Sort
        self.train_dist_2_surf_idx = np.argsort(train_r)
        self.train_r_surf = train_r[self.train_dist_2_surf_idx] * 1000

    def get_test_data(self):
        planet = self.config["planet"][0]
        original_max_radius = self.config["radius_max"][0]
        augment_data = self.config.get("augment_data_config", [{}])[0]
        extra_max_radius = augment_data.get("radius_max", [0])[0]
        max_radius = np.max([original_max_radius, extra_max_radius])

        self.config["radius_min"][0]
        obj_file = self.config.get("grav_file", [None])[0]

        gravity_data_fcn = self.config["gravity_data_fcn"][0]

        interpolation_dist = RandomDist(
            planet,
            radius_bounds=self.training_bounds,
            points=self.points,
            **self.config,
        )
        extrapolation_dist = RandomDist(
            planet,
            radius_bounds=[max_radius, max_radius * 10],
            points=self.points,
            **self.config,
        )

        x, a, u = gravity_data_fcn(interpolation_dist, obj_file, **self.config)
        x_extra, a_extra, u_extra = gravity_data_fcn(
            extrapolation_dist,
            obj_file,
            **self.config,
        )

        x = np.append(x, x_extra, axis=0)
        a = np.append(a, a_extra, axis=0)
        u = np.append(u, u_extra, axis=0)

        full_dist = interpolation_dist
        full_dist.positions = np.append(
            full_dist.positions,
            extrapolation_dist.positions,
            axis=0,
        )

        # Compute distance to COM
        x_sph = cart2sph(x)
        self.test_dist_2_COM_idx = np.argsort(x_sph[:, 0])
        self.test_r_COM = x_sph[self.test_dist_2_COM_idx, 0]

        mesh = interpolation_dist.shape_model
        _, test_r, _ = trimesh.proximity.closest_point(mesh, x / 1000)

        # Sort
        self.test_dist_2_surf_idx = np.argsort(test_r)
        self.test_r_surf = test_r[self.test_dist_2_surf_idx] * 1000

        self.positions = x
        self.test_accelerations = a
        self.test_potentials = u

    def get_PINN_data(self):
        positions = self.positions
        self.predicted_accelerations = (
            self.model.compute_acceleration(
                positions,
            )
            .numpy()
            .astype(float)
        )
        self.predicted_potentials = (
            self.model.compute_potential(positions).numpy().astype(float)
        )

    def compute_losses(self, loss_fcn_list):
        losses = {}
        for loss_key in loss_fcn_list:
            loss_fcn = get_loss_fcn(loss_key)

            # Compute loss on acceleration and potential
            losses.update(
                {
                    f"{loss_fcn.__name__}": loss_fcn(
                        self.predicted_accelerations,
                        self.test_accelerations,
                    ).numpy(),
                },
            )
        self.losses = losses

    def compute_loss(self):
        loss_fcns = self.config.get("loss_fcns", [["rms", "percent"]])[0]
        loss_list = [get_loss_fcn(loss_key) for loss_key in loss_fcns]
        y_hat = {"accelerations": self.predicted_accelerations}
        y = {"accelerations": self.test_accelerations}
        losses = MetaLoss(y, y_hat, loss_list)
        self.loss_acc = tf.reduce_sum(
            [tf.reduce_mean(loss) for loss in losses.values()],
        )

    def run(self):
        self.get_train_data()
        self.get_test_data()
        self.get_PINN_data()
        self.compute_losses(self.loss_fcn_list)
        self.compute_loss()


def main():
    import matplotlib.pyplot as plt
    import pandas as pd

    from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
    from GravNN.Networks.Model import load_config_and_model
    from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer

    df = pd.read_pickle("Data/Dataframes/optimize_20.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(model_id, df)
    extrapolation_exp = ExtrapolationExperiment(model, config, 500)
    extrapolation_exp.run()
    vis = ExtrapolationVisualizer(extrapolation_exp)
    vis.plot_interpolation_percent_error()
    vis.plot_extrapolation_percent_error()
    # vis.plot_interpolation_loss()
    plt.show()


if __name__ == "__main__":
    main()
