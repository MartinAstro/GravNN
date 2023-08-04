import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from GravNN.Support.transformations import cart2sph, check_fix_radial_precision_errors
from GravNN.Visualization.VisualizationBase import VisualizationBase


class MapBase(VisualizationBase):
    def __init__(self, unit="m/s^2", **kwargs):
        """Visualization class responsible for plotting maps of the gravity field.

        Args:
            unit (str, optional): Acceleration unit (ex. 'mGal' or 'm/s^2').
        """
        super().__init__(**kwargs)
        if unit == "mGal":
            # https://en.wikipedia.org/wiki/Gal_(unit)
            # 1 Gal == 0.01 m/s^2
            # 1 mGal == 1E-2 * 10E-3 = 10E-5 or 10000 mGal per m/s^2
            self.scale = 10000.0
        elif unit == "m/s^2":
            self.scale = 1.0
        pass
        self.tick_interval = [30, 30]
        self.unit = unit

    def new_map(
        self,
        grid,
        **kwargs,
    ):  # vlim=None, log_scale=False, alpha=None, cmap=None):
        vlim = kwargs.get("vlim", None)
        log_scale = kwargs.get("log_scale", False)
        alpha = kwargs.get("alpha", None)
        cmap = kwargs.get("cmap", None)
        labels = kwargs.get("labels", True)
        xlabel = kwargs.get("xlabel", None)

        ticks = kwargs.get("ticks", True)
        if ticks:
            yticks = np.linspace(
                -90,
                90,
                num=180 // self.tick_interval[1] + 1,
                endpoint=True,
                dtype=int,
            )
            xticks = np.linspace(
                0,
                360,
                num=360 // self.tick_interval[0] + 1,
                endpoint=True,
                dtype=int,
            )

            xloc = np.linspace(
                0,
                len(grid) - 1,
                num=len(xticks),
                endpoint=True,
                dtype=int,
            )
            yloc = np.linspace(
                0,
                len(grid[0]),
                num=len(yticks),
                endpoint=True,
                dtype=int,
            )

            xticks_labels = [r"$" + str(xtick) + r"^\circ$" for xtick in xticks]
            yticks_labels = [r"$" + str(ytick) + r"^\circ$" for ytick in yticks]

            try:
                plt.xticks(xloc, labels=xticks_labels, fontsize=11)
                plt.yticks(yloc, labels=yticks_labels, fontsize=11)
            except Exception:
                plt.xticks([])
                plt.yticks([])
        else:
            plt.xticks([])
            plt.yticks([])

        ax = plt.gca()

        if labels:
            ax.set_xlabel("Longitude", fontsize=11)
            ax.set_ylabel("Latitude", fontsize=11)

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=11)

        grid = np.transpose(grid)  # imshow takes (MxN)
        if log_scale:
            if vlim is not None:
                norm = SymLogNorm(linthresh=1e-4, vmin=vlim[0], vmax=vlim[1])
            else:
                norm = SymLogNorm(1e-4)
        else:
            norm = None

        if vlim is not None:
            im = plt.imshow(
                grid * self.scale,
                vmin=vlim[0],
                vmax=vlim[1],
                norm=norm,
                alpha=alpha,
                cmap=cmap,
            )
        else:
            im = plt.imshow(grid * self.scale, norm=norm, alpha=alpha, cmap=cmap)
        return im

    def add_colorbar(self, im, label, **kwargs):
        extend = kwargs.get("extend", "neither")
        vlim = kwargs.get("vlim", None)
        loc = kwargs.get("loc", "right")
        orientation = kwargs.get("orientation", "vertical")
        pad = kwargs.get("pad", 0.05)

        ax = plt.gca()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(loc, size="5%", pad=pad)
        if self.unit == "mGal":
            format = "$%.2f$"
        else:
            format = "$%.4f$"
        if vlim is not None:
            t = np.linspace(vlim[0], vlim[1], 5)
            format = kwargs.get("format", format)
            cBar = plt.colorbar(
                im,
                cax=cax,
                ticks=t,
                format=format,
                extend=extend,
                orientation=orientation,
            )

        else:
            cBar = plt.colorbar(
                im,
                cax=cax,
                format=format,
                extend=extend,
                orientation=orientation,
            )
        if orientation == "vertical":
            cBar.ax.set_ylabel(label)
        else:
            cBar.ax.set_xlabel(label)
            cBar.ax.xaxis.set_ticks_position("top")
            cBar.ax.xaxis.set_label_position("top")

        if loc == "left":
            cBar.ax.yaxis.set_ticks_position("left")
            cBar.ax.yaxis.set_label_position("left")

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    def plot_grid(self, grid, label, **kwargs):  # vlim=None, cmap=None):
        # TODO: Consider depreciating this and shifting just to plot_map
        new_fig = kwargs.get("new_fig", True)
        colorbar = kwargs.get("colorbar", True)
        title = kwargs.get("title", None)

        if new_fig:
            fig, ax = self.newFig()
        else:
            fig = plt.gcf()
            ax = plt.gca()
        im = self.new_map(grid, **kwargs)
        if title:
            plt.title(title)
        if colorbar:
            self.add_colorbar(im, label, **kwargs)
        return fig, ax

    def plot_trajectory(self, trajectory):
        pos_sph = cart2sph(np.array(trajectory.positions))
        pos_sph = check_fix_radial_precision_errors(pos_sph)
        cmap = get_cmap("Spectral")

        dataLim = plt.gcf().axes[0].dataLim
        xpix_per_deg = (dataLim.max[0] - dataLim.min[0]) / 360.0  # x
        ypix_per_deg = (dataLim.max[1] - dataLim.min[1]) / 180.0  # x

        plt.sca(plt.gcf().axes[0])
        colors = (pos_sph[:, 0] - np.min(pos_sph[:, 0])) / (
            np.max(pos_sph[:, 0]) - np.min(pos_sph[:, 0])
        )
        plt.scatter(
            pos_sph[:, 1] * xpix_per_deg,
            pos_sph[:, 2] * ypix_per_deg,
            c=cmap(colors),
            s=1,
            zorder=10,
        )
