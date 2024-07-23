import numpy as np
import trimesh

from GravNN.Analysis.ExperimentBase import ExperimentBase
from GravNN.Networks.Losses import *
from GravNN.Support.batches import batch_function
from GravNN.Support.transformations import cart2sph
from GravNN.Trajectories.RandomDist import RandomDist


class ExtrapolationExperiment(ExperimentBase):
    def __init__(
        self,
        model,
        config,
        points,
        extrapolation_bound=10,
        random_seed=1234,
        **kwargs,
    ):
        super().__init__(
            model,
            config,
            points,
            extrapolation_bound,
            random_seed=1234,
            **kwargs,
        )
        self.config = config
        self.model = model
        self.points = points
        self.loss_fcn_list = ["mse", "rms", "percent"]

        self.brillouin_radius = config["planet"][0].radius
        original_max_radius = self.config["radius_max"][0]
        extra_max_radius = np.nan_to_num(self.config.get("extra_radius_max", [0])[0], 0)
        max_radius = np.max([original_max_radius, extra_max_radius])
        self.training_bounds = [config["radius_min"][0], max_radius]
        self.extrapolation_bound = extrapolation_bound

        np.random.seed(random_seed)

    def get_test_data(self):
        planet = self.config["planet"][0]
        self.config["radius_min"][0]
        obj_file = self.config.get("obj_file", [None])[0]

        gravity_data_fcn = self.config["gravity_data_fcn"][0]

        interpolation_dist = RandomDist(
            planet,
            radius_bounds=self.training_bounds,
            points=self.points,
            **self.config,
        )

        point_density = self.points / np.diff(self.training_bounds)[0]
        extrap_R = self.extrapolation_bound * planet.radius
        training_R = self.training_bounds[1]
        dr = extrap_R - training_R
        extrapolation_points = int(point_density * dr)
        extrapolation_dist = RandomDist(
            planet,
            radius_bounds=[training_R, extrap_R],
            points=extrapolation_points,
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

        self.positions = x
        self.a_test = a
        self.u_test = u

        # Compute distance to COM
        x_sph = cart2sph(x)
        self.test_dist_2_COM_idx = np.argsort(x_sph[:, 0])
        self.test_r_COM = x_sph[self.test_dist_2_COM_idx, 0]

        if not hasattr(self, "test_dist_2_surf_idx"):
            mesh = interpolation_dist.obj_mesh

            def closest_point_fcn(x):
                return trimesh.proximity.closest_point(mesh, x)[1]

            test_r = batch_function(closest_point_fcn, (len(x),), x / 1000, 100)

            # Sort
            self.test_dist_2_surf_idx = np.argsort(test_r)
            self.test_r_surf = test_r[self.test_dist_2_surf_idx] * 1000

    def get_model_data(self):
        if not hasattr(self, "a_pred"):
            positions = self.positions
            pred_acc = self.model.compute_acceleration(positions)
            pred_pot = self.model.compute_potential(positions)
            try:
                pred_acc = pred_acc.numpy().astype(float)
                pred_pot = pred_pot.numpy().astype(float)
            except Exception:
                pass

            self.u_pred = pred_pot
            self.a_pred = pred_acc

    def compute_losses(self, loss_fcn_list):
        losses = {}
        for loss_key in loss_fcn_list:
            loss_fcn = get_loss_fcn(loss_key)

            # Compute loss on acceleration and potential
            losses.update(
                {
                    f"{loss_fcn.__name__}": loss_fcn(
                        self.a_pred,
                        self.a_test,
                    ).numpy(),
                },
            )
        self.losses = losses

    def compute_loss(self):
        loss_fcns = self.config.get("loss_fcns", [["rms", "percent"]])[0]
        loss_list = [get_loss_fcn(loss_key) for loss_key in loss_fcns]
        y_hat = {"accelerations": self.a_pred}
        y = {"accelerations": self.a_test}
        losses = MetaLoss(y, y_hat, loss_list)
        self.loss_acc = tf.reduce_sum(
            [tf.reduce_mean(loss) for loss in losses.values()],
        )

    def generate_data(self):
        self.get_test_data()
        self.get_model_data()
        self.compute_losses(self.loss_fcn_list)
        self.compute_loss()

        data = {
            "test_dist_2_surf_idx": self.test_dist_2_surf_idx,
            "test_r_surf": self.test_r_surf,
            "test_dist_2_COM_idx": self.test_dist_2_COM_idx,
            "test_r_COM": self.test_r_COM,
            "a_pred": self.a_pred,
            "u_pred": self.u_pred,
        }
        return data


def main():
    import matplotlib.pyplot as plt
    import pandas as pd

    from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
    from GravNN.Networks.Model import load_config_and_model
    from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer

    df = pd.read_pickle("Data/Dataframes/eros_poly_071123.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(df, model_id)
    extrapolation_exp = ExtrapolationExperiment(model, config, 100)
    extrapolation_exp.run()
    vis = ExtrapolationVisualizer(extrapolation_exp)
    vis.plot_interpolation_percent_error()
    vis.plot_extrapolation_percent_error()
    # vis.plot_interpolation_loss()
    plt.show()


if __name__ == "__main__":
    main()
