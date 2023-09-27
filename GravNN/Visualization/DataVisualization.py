import os

import matplotlib.pyplot as plt
import numpy as np

from GravNN.Visualization.VisualizationBase import VisualizationBase


class DataVisualization(VisualizationBase):
    def __init__(self, **kwargs):
        """Visualization for trajectory distributions"""
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"

    def plot_values_and_residuals(
        self,
        x,
        y,
        y_pred,
        label="Pred",
        ylabel=None,
        title=None,
        vlines=None,
        vline_labels=None,
        percent=False,
        alpha=1.0,
        plot_truth=True,
    ):
        plt.subplot(2, 1, 1)
        plt.scatter(x, y_pred, s=1, label=label, alpha=alpha)
        if plot_truth:
            plt.scatter(x, y, s=1, label="True", alpha=alpha, c="c")

        self.plot_radii(x, vlines, vline_labels)
        # for line, name in zip(ax.lines, [r'$r_B$', r'$r_{\text{min}}$', r'$r_{\text{max}}$']):
        #     y = line.get_xdata()[-1]
        #     ax.annotate(name, xy=(x,1), xytext=(0,6), color=line.get_color(),
        #     xycoords = ax.get_xaxis_transform(), textcoords="offset points",
        #     size=8, va="center")

        if ylabel is not None:
            plt.ylabel(ylabel)

        if percent:
            diff = y - y_pred
            ylabel2 = "Residual"
        else:
            diff = np.abs((y - y_pred) / y) * 100.0
            ylabel2 = "Percent Error"

        plt.subplot(2, 1, 2)
        plt.scatter(x, diff, s=1, alpha=alpha, label=label)
        plt.legend(loc="upper left")

        if (
            not np.any(x < 0.0) and vlines is not None
        ):  # Confirm that this is spherical coordinates
            plt.gca()
            i = 0
            colors = ["r", "b", "g"]
            for vline in vlines:
                plt.axvline(vline, c=colors[i])
                i += 1
            # for line, name in zip(ax.lines, [r'$r_B$', r'$r_{\text{min}}$', r'$r_{\text{max}}$']):
            #     y = line.get_xdata()[-1]
            #     ax.annotate(name, xy=(x,1), xytext=(0,6), color=line.get_color(),
            #     xycoords = ax.get_yaxis_transform(), textcoords="offset points",
            #     size=14, va="center")

        # plt.gca().yaxis.set_ticks_position('right')
        # plt.gca().yaxis.set_label_position('right')
        plt.ylabel(ylabel2)

    def plot_residuals(
        self,
        x,
        y,
        y_pred,
        label="Pred",
        ylabel=None,
        title=None,
        vlines=None,
        vline_labels=None,
        percent=False,
        alpha=1.0,
    ):
        if percent:
            diff = np.linalg.norm(y - y_pred, axis=1)
            # diff = y - y_pred
            ylabel2 = ylabel + "\n" + "Residual"
        else:
            diff = (
                np.linalg.norm(y - y_pred, axis=1) / np.linalg.norm(y, axis=1) * 100.0
            )
            # diff = np.abs((y - y_pred)/y)*100.0
            ylabel2 = ylabel + "\n" + "Percent Error"

        plt.scatter(x, diff, s=1, alpha=alpha, label=label)
        plt.legend(loc="upper left")
        plt.ylabel(ylabel2)

    def plot_radii(self, x, vlines, vline_labels):
        if (
            not np.any(x < 0.0) and vlines is not None
        ):  # Confirm that this is spherical coordinates
            ax = plt.gca()
            i = 0
            colors = ["r", "b", "g"]
            for vline in vlines:
                line = plt.axvline(vline, c=colors[i])
                if vline_labels is not None:
                    x_data = line.get_xdata()
                    line.get_ydata()
                    plt.annotate(
                        vline_labels[i],
                        xy=(
                            x_data[-1],
                            10
                            ** (
                                (np.log10(ax.viewLim.ymax) - np.log10(ax.viewLim.ymin))
                                / 3.0
                            )
                            + np.log10(ax.viewLim.ymin),
                        ),
                        rotation="vertical",
                        c=line.get_color(),
                        textcoords="data",
                    )
                i += 1

    def plot_box_and_whisker(self, y, y_pred, label=None, percent=True):
        # https://plotly.com/python/box-plots/ -- Rainbow Box Plots section

        import numpy as np
        import plotly.graph_objects as go

        y = np.array(y)
        y_pred = np.array(y_pred)
        if percent:
            diff = np.abs((y - y_pred) / y)
        else:
            diff = y - y_pred

        boxes = []
        for i in range(0, len(diff)):
            boxes.append(go.Box(y=diff[i, :], name=label[i]))
        fig = go.Figure(data=boxes)
        return fig
