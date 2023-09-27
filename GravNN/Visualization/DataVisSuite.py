import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.Visualization.DataVisualization import DataVisualization


class DataVisSuite(DataVisualization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"

    def plot_potential_dist(self, x, u, u_pred, title=None):
        self.newFig()
        plt.subplot(2, 1, 1)
        plt.scatter(x[:, 0], u_pred[:, 0], s=1, c="b", label="Pred")
        plt.scatter(x[:, 0], u, s=1, c="r", label="True")
        plt.ylabel("potential")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.scatter(x[:, 0], u_pred[:, 0] - u, s=1)
        plt.ylabel("residual")
        plt.suptitle(title)

    def plot_acceleration_dist(self, x, a, a_pred, title=None, vlines=None):
        self.newFig()
        plt.subplot(3, 2, 1)
        plt.scatter(x[:, 0], a_pred[:, 0], s=1, c="b", label="Pred")
        plt.scatter(x[:, 0], a[:, 0], s=1, c="r", label="True")

        if (
            not np.any(x[:, 0] < 0.0) and vlines is not None
        ):  # Confirm that this is spherical coordinates
            for vline in vlines:
                plt.axvline(vline)

        plt.ylabel("a 1")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.scatter(x[:, 0], a[:, 0] - a_pred[:, 0], s=1)
        if (
            not np.any(x[:, 0] < 0.0) and vlines is not None
        ):  # Confirm that this is spherical coordinates
            for vline in vlines:
                plt.axvline(vline)

        plt.gca().yaxis.set_ticks_position("right")
        plt.gca().yaxis.set_label_position("right")
        plt.ylabel("residual")

        plt.subplot(3, 2, 3)
        plt.scatter(x[:, 1], a_pred[:, 1], s=1, c="b")
        plt.scatter(x[:, 1], a[:, 1], s=1, c="r")
        plt.ylabel("a 2")

        plt.subplot(3, 2, 4)
        plt.scatter(x[:, 1], a[:, 1] - a_pred[:, 1], s=1)
        plt.gca().yaxis.set_ticks_position("right")
        plt.gca().yaxis.set_label_position("right")
        plt.ylabel("residual")

        plt.subplot(3, 2, 5)
        plt.scatter(x[:, 2], a_pred[:, 2], s=1, c="b")
        plt.scatter(x[:, 2], a[:, 2], s=1, c="r")
        plt.ylabel("a 3")

        plt.subplot(3, 2, 6)
        plt.scatter(x[:, 2], a[:, 2] - a_pred[:, 2], s=1)
        plt.ylabel("residual")
        plt.gca().yaxis.set_ticks_position("right")
        plt.gca().yaxis.set_label_position("right")
        plt.suptitle(title)

    def plot_acceleration_residuals(self, x, a, a_pred, title, percent=False):
        if percent:
            y = np.abs((a - a_pred) / a)
        else:
            y = a - a_pred

        def compute_rolling_avg_std(idx, window, no_of_std):
            data = {"x": x[:, idx], "y": y[:, idx]}
            df = pd.DataFrame.from_dict(data)
            df.sort_values("x", inplace=True, axis=0)
            df["y_mean"] = df["y"].rolling(window).mean()
            df["y_std"] = df["y"].rolling(window).std()
            df["y_std_max"] = df["y_mean"] + no_of_std * df["y_std"]
            df["y_std_min"] = df["y_mean"] - no_of_std * df["y_std"]
            plt.plot(df["x"].values, df["y_mean"].values)
            plt.gca().fill_between(
                df["x"],
                df["y_std_min"],
                df["y_std_max"],
                color="#888888",
                alpha=0.4,
            )

        self.newFig()
        plt.subplot(3, 1, 1)
        compute_rolling_avg_std(0, 500, 3)
        plt.subplot(3, 1, 2)
        compute_rolling_avg_std(1, 500, 3)
        plt.subplot(3, 1, 3)
        compute_rolling_avg_std(2, 500, 3)
        plt.suptitle(title)

    def plot_acceleration_box_and_whisker(self, x, a, a_pred, percent=True):
        # https://plotly.com/python/box-plots/ -- Rainbow Box Plots section

        import numpy as np
        import plotly.graph_objects as go

        if percent:
            y = np.abs((a - a_pred) / a)
        else:
            y = a - a_pred

        idx = 0
        data = {"x": x[:, idx], "y": y[:, idx]}
        df = pd.DataFrame.from_dict(data)
        df.sort_values("x", inplace=True, axis=0)
        out, bins = pd.cut(df["x"], 100, retbins=True)
        boxes = []
        for i in range(1, len(bins)):
            samples = df.loc[(df["x"] >= bins[i - 1]) & (df["x"] < bins[i])]
            boxes.append(go.Box(y=samples["y"]))
        fig = go.Figure(data=boxes)
        fig.show()
