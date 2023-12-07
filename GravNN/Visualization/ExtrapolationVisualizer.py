import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sigfig

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.VisualizationBase import VisualizationBase


class ExtrapolationVisualizer(VisualizationBase):
    def __init__(self, experiment, **kwargs):
        super().__init__(**kwargs)
        plt.rc("font", size=7.0)
        self.experiment = experiment
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.radius = self.experiment.config["planet"][0].radius
        self.training_bounds = np.array(self.experiment.training_bounds)
        self.max_idx = np.where(self.experiment.test_r_COM > self.training_bounds[1])[
            0
        ][0]
        self.set_x_axis(kwargs.get("x_axis", "dist_2_surf"))
        self.plot_fcn = kwargs.get("plot_fcn", plt.plot)

    def set_x_axis(self, x_type):
        if x_type == "dist_2_COM":
            self.x_test = self.experiment.test_r_COM / self.radius
            self.idx_test = self.experiment.test_dist_2_COM_idx
            self.x_label = "Distance to COM [R]"
        elif x_type == "dist_2_surf":
            self.x_test = self.experiment.test_r_surf / self.radius
            self.idx_test = self.experiment.test_dist_2_surf_idx
            self.x_label = "Distance to Surface [R]"

        else:
            raise ValueError()

    def annotate_metrics(self, x, values, xy=(0.3, 0.95), critical_radius=1.0):
        interior_mask = x < critical_radius

        interior_values = values[interior_mask]
        exterior_values = values[~interior_mask]

        def compute_stats_str(vals):
            avg = sigfig.round(np.mean(vals), sigfigs=2)
            std = sigfig.round(np.std(vals), sigfigs=2)
            max = sigfig.round(np.max(vals), sigfigs=2)
            if avg > 1e3:
                stat_str = "%.1E ± %.1E (%.1E)" % (avg, std, max)
            else:
                stat_str = f"{avg}±{std} ({max})"
            return stat_str

        interior_stats = "Interior: " + compute_stats_str(interior_values)
        exterior_stats = "Exterior: " + compute_stats_str(exterior_values)

        plt.gca().annotate(
            interior_stats,
            xy=(0.5, 0.9),
            ha="center",
            va="center",
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", fc="w"),
        )
        plt.gca().annotate(
            exterior_stats,
            xy=(0.5, 0.8),
            ha="center",
            va="center",
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", fc="w"),
        )

    def plot_histogram(self, x):
        ax = plt.twinx()
        plt.hist(x, 50, alpha=0.2)
        plt.ylabel("Frequency")
        ax.grid(False)
        ax.set_zorder(1)

    def plot(self, x, value, **kwargs):
        # compute trend lines
        def get_rolling_lines(data):
            df = pd.DataFrame(data=data, index=None)
            avg_window = kwargs.get("avg_window", 50)
            std_window = kwargs.get("std_window", 50)
            max_window = kwargs.get("max_window", 10)
            avg = df.rolling(avg_window, 25).mean()
            std = df.rolling(std_window, 25).std()
            max = df.rolling(max_window, 10).max()
            return avg, std, max

        # sort entries
        avg_line, std_line, max_line = get_rolling_lines(value)
        label = kwargs.get("label", None)
        linewidth = kwargs.get("linewidth", 0.5)
        color = kwargs.get("color", None)

        if kwargs.get("new_fig", True):
            self.newFig()
        plt.scatter(x, value, c=color, alpha=0.1, s=1)
        line_color = kwargs.get("line_color", color)
        self.plot_fcn(x, avg_line, label=label, linewidth=linewidth, color=line_color)

        if kwargs.get("plot_std", True):
            y_std_upper = np.squeeze(avg_line + 1 * std_line)
            y_std_lower = np.squeeze(avg_line - 1 * std_line)
            plt.fill_between(x, y_std_lower, y_std_upper, color="C0", alpha=0.5)

        if kwargs.get("plot_max", True):
            self.plot_fcn(x, max_line, color="red")

        training_bounds = self.training_bounds / self.radius
        plt.vlines(training_bounds[0], ymin=0, ymax=np.max(value), color="green")
        plt.vlines(training_bounds[1], ymin=0, ymax=np.max(value), color="green")
        plt.vlines(1, ymin=0, ymax=np.max(value), color="grey")
        if kwargs.get("annotate", True):
            self.annotate_metrics(
                x,
                value,
                critical_radius=kwargs.get("critical_radius", 1.0),
            )
        plt.tight_layout()

    def plot_interpolation_loss(self, **kwargs):
        self.plot(
            self.x_test[: self.max_idx],
            self.experiment.loss_acc[self.idx_test][: self.max_idx],
            **kwargs,
        )
        plt.gca().set_yscale("log")
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("Loss")
        plt.xlabel(self.x_label)
        plt.ylim([0, None])
        plt.xlim([0, 2])
        # self.plot_histogram(self.x_train)

    def plot_extrapolation_loss(self, **kwargs):
        self.plot(self.x_test, self.experiment.loss_acc[self.idx_test], **kwargs)
        plt.gca().set_yscale("log")
        plt.ylabel("Loss")
        plt.xlabel(self.x_label)

    def plot_interpolation_rms(self, **kwargs):
        self.plot(
            self.x_test[: self.max_idx],
            self.experiment.losses["rms"][self.idx_test][: self.max_idx],
            **kwargs,
        )
        plt.gca().set_yscale("log")
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("RMS [$m/s^2$]")
        plt.xlabel(self.x_label)
        plt.ylim([0, None])
        # self.plot_histogram(self.x_train)

    def plot_extrapolation_rms(self, **kwargs):
        self.plot(self.x_test, self.experiment.losses["rms"][self.idx_test], **kwargs)
        plt.gca().set_yscale("log")
        plt.ylabel("RMS [$m/s^2$]")
        plt.xlabel(self.x_label)

    def plot_interpolation_percent_error(self, **kwargs):
        interp_error = self.experiment.losses["percent"][self.idx_test][: self.max_idx]
        self.plot(
            self.x_test[: self.max_idx],
            interp_error * 100,
            **kwargs,
        )
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("Percent Error")
        plt.xlabel(self.x_label)
        plt.ylim([0, None])
        # self.plot_histogram(self.x_train)

    def plot_extrapolation_percent_error(self, **kwargs):
        R_max = self.experiment.config["radius_max"][0] / self.radius
        extrapol_error = self.experiment.losses["percent"][self.idx_test]
        self.plot(
            self.x_test,
            extrapol_error * 100,
            critical_radius=R_max,
            **kwargs,
        )
        plt.ylabel("Percent Error")
        plt.xlabel(self.x_label)

    def plot_extrapolation_acc(self, **kwargs):
        R_max = self.experiment.config["radius_max"][0] / self.radius
        a = self.experiment.a_pred[self.idx_test]
        a_mag = np.linalg.norm(a, axis=1)
        self.plot(
            self.x_test,
            a_mag,
            critical_radius=R_max,
            **kwargs,
        )
        plt.ylabel("Acceleration Magnitude")
        plt.xlabel(self.x_label)


def main():
    # df = pd.read_pickle("Data/Dataframes/LR_Anneal_With_Noise_032223.data")
    df = pd.read_pickle("Data/Dataframes/LR_Anneal_No_Noise_032223.data")

    for i in range(1, 2):
        model_id = df["id"].values[-i]  # with scaling
        config, model = load_config_and_model(df, model_id)

        # evaluate the error at "training" altitudes and beyond
        extrapolation_exp = ExtrapolationExperiment(model, config, 1000)
        extrapolation_exp.run()

        # visualize error @ training altitude and beyond
        vis = ExtrapolationVisualizer(
            extrapolation_exp,
            x_axis="dist_2_COM",
            plot_fcn=plt.semilogy,
        )
        vis.plot_interpolation_percent_error()
        plt.gca().set_ylim([1e-3, 1e2])
        vis.plot_extrapolation_percent_error()
    # vis.plot_interpolation_mse()
    # vis.plot_extrapolation_mse()

    plt.show()


if __name__ == "__main__":
    main()
