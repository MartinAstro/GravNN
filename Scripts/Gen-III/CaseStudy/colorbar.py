import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, SymLogNorm

from GravNN.Visualization.VisualizationBase import VisualizationBase


class StandaloneColorbar(VisualizationBase):
    def __init__(
        self,
        cmap="viridis",
        bounds=(0, 1),
        orientation="vertical",
        label="",
        log=False,
        reverse_cmap=False,
        figsize=(2, 8),
        tick_and_label_position="left",
    ):
        super().__init__()
        self.cmap = cmap
        if reverse_cmap:
            self.cmap = self.cmap.reversed()
        self.bounds = bounds
        self.orientation = orientation
        self.log = log
        self.label = label
        self.fig_size = figsize
        self.tick_and_label_position = tick_and_label_position

    def plot(self):
        # Create a figure with specified size
        fig, ax = self.newFig()

        # Set the colormap
        if self.log:
            norm = SymLogNorm(
                linthresh=1e-4,
                vmin=self.bounds[0],
                vmax=self.bounds[1],
            )
        else:
            norm = Normalize(vmin=self.bounds[0], vmax=self.bounds[1])

        # Create colorbar base
        cbar = ColorbarBase(ax, cmap=self.cmap, norm=norm, orientation=self.orientation)

        # Set the label
        cbar.set_label(self.label)

        # Change tick + label positions
        cbar.ax.yaxis.set_ticks_position(self.tick_and_label_position)
        cbar.ax.yaxis.set_label_position(self.tick_and_label_position)

        # rotate tick labels
        # if self.orientation == "vertical":
        # plt.setp(cbar.ax.get_yticklabels(), rotation=35)

        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.tight_layout(pad=0)
        return fig, cbar


if __name__ == "__main__":
    base_vis = VisualizationBase()
    fig_size = (base_vis.w_full / 9, base_vis.w_tri * 2)

    vis = StandaloneColorbar(
        cmap="jet",
        bounds=(0.01, 10),
        orientation="vertical",
        label="Acceleration Error",
        log=True,
        figsize=fig_size,
    )
    vis.plot()
    vis.save(plt.gcf(), "primary_colorbar")

    fig_size = (base_vis.w_full, base_vis.w_full * 0.1)
    vis = StandaloneColorbar(
        cmap="jet",
        bounds=(0.01, 10),
        orientation="horizontal",
        label="Acceleration Error",
        log=True,
        figsize=fig_size,
    )
    vis.plot()
    vis.save(plt.gcf(), "primary_colorbar_horizontal")
    plt.show()
