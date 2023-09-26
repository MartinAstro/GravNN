import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import (
    cart2sph,
    invert_projection,
    project_acceleration,
)
from GravNN.Visualization.MapBase import MapBase
from GravNN.Visualization.VisualizationBase import VisualizationBase


# TODO : Consider using subplot2grid to make more compact figures
# TODO: Make data plotting routines, and model plotting routines
class Plotting:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.x_transformer = config["x_transformer"][0]
        self.a_transformer = config["a_transformer"][0]
        self.directory = (
            os.path.abspath(".") + "/Data/Networks/" + str(config["id"][0]) + "/"
        )
        self.vis = VisualizationBase()

    def plot_maps(self, map_trajectories):
        for name, map_traj in map_trajectories.items():
            sh_file = map_traj.celestial_body.sh_file
            x, a, u = get_sh_data(map_traj, sh_file, **self.config)

            if self.config["basis"][0] == "spherical":
                x = cart2sph(x)
                a = project_acceleration(x, a)
                x[:, 1:3] = np.deg2rad(x[:, 1:3])

            x = self.x_transformer.transform(x)
            a = self.a_transformer.transform(a)

            U_pred, acc_pred, laplace, curl = self.model.predict(x.astype("float32"))

            x = self.x_transformer.inverse_transform(x)
            a = self.a_transformer.inverse_transform(a)
            acc_pred = self.a_transformer.inverse_transform(acc_pred)

            if self.config["basis"][0] == "spherical":
                x[:, 1:3] = np.rad2deg(x[:, 1:3])
                a = invert_projection(x, a)
                invert_projection(
                    x,
                    acc_pred.astype(float),
                )  # numba requires that the types are the same

            grid_true = Grid(trajectory=map_traj, accelerations=a)
            grid_pred = Grid(trajectory=map_traj, accelerations=acc_pred)
            diff = grid_pred - grid_true

            mapUnit = "mGal"
            map_vis = MapBase(mapUnit)
            plt.rc("text", usetex=False)

            fig_true, ax = map_vis.plot_grid(grid_true.total, "True Grid [mGal]")
            fig_pred, ax = map_vis.plot_grid(grid_pred.total, "NN Grid [mGal]")
            fig_pert, ax = map_vis.plot_grid(
                diff.total,
                "Acceleration Difference [mGal]",
            )

            fig, ax = map_vis.newFig(fig_size=(5 * 4, 3.5 * 4))
            vlim = [0, np.max(grid_true.total) * 10000.0]
            plt.subplot(311)
            im = map_vis.new_map(grid_true.total, vlim=vlim, log_scale=False)
            map_vis.add_colorbar(im, "[mGal]", vlim)

            plt.subplot(312)
            im = map_vis.new_map(grid_pred.total, vlim=vlim, log_scale=False)
            map_vis.add_colorbar(im, "[mGal]", vlim)

            plt.subplot(313)
            im = map_vis.new_map(diff.total, vlim=vlim, log_scale=False)
            map_vis.add_colorbar(im, "[mGal]", vlim)

            os.makedirs(self.directory + name, exist_ok=True)
            map_vis.save(fig_true, self.directory + name + "/" + "true.pdf")
            map_vis.save(fig_pred, self.directory + name + "/" + "pred.pdf")
            map_vis.save(fig_pert, self.directory + name + "/" + "diff.pdf")
            map_vis.save(fig, self.directory + name + "/" + "all.pdf")

    def plot_loss(self, log=False):
        fig, ax = self.vis.newFig()
        history = self.config["history"][0]
        epochs = np.arange(0, len(history["loss"]), 1)
        start = 100
        if log is False:
            plt.plot(epochs[start:], history["loss"][start:])
            plt.plot(epochs[start:], history["val_loss"][start:])
        else:
            plt.semilogy(epochs[start:], history["loss"][start:])
            plt.semilogy(epochs[start:], history["val_loss"][start:])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        self.vis.save(fig, self.directory + "loss.pdf")
        return fig

    def plot_alt_curve(self, stat, ylabel=None, linestyle="-", tick_position=None):
        df = pd.read_pickle(self.directory + "rse_alt.data")
        fig, ax = self.vis.newFig()
        plt.plot(df.index / 1000.0, df[stat], linestyle=linestyle)
        plt.xlabel("Altitude [km]")
        if ylabel is None:
            plt.ylabel(stat)
        else:
            plt.ylabel(ylabel)
        if tick_position is not None:
            ax.yaxis.set_ticks_position(tick_position)
        self.vis.save(fig, self.directory + "altitude.pdf")
        return fig

    def plot_data_alt_curve(self, stat):
        fig = self.plot_alt_curve(stat)
        distribution = self.config["distribution"][0]
        trajectory = distribution(
            self.config["planet"][0],
            [self.config["radius_min"][0], self.config["radius_max"][0]],
            self.config["N_dist"][0],
            **self.config,
        )  # points=1000000)
        positions = cart2sph(trajectory.positions) - self.config["planet"][0].radius
        ax1 = plt.gca()
        ax1.tick_params("y", colors="r")
        ax1.set_ylabel("RSE", color="r")
        ax1.get_lines()[0].set_color("r")

        ax2 = ax1.twinx()
        plt.hist(positions[:, 0], bins=100, alpha=0.5)
        ax2.tick_params("y", colors="b")
        ax2.set_ylabel("Frequency", color="b")
        self.vis.save(fig, self.directory + "data_distribution.pdf")
        return fig

    def plot_model_graph(self):
        dot_img_file = self.directory + "model_graph.pdf"
        tf.keras.utils.plot_model(
            self.model.network,
            to_file=dot_img_file,
            show_shapes=True,
        )
