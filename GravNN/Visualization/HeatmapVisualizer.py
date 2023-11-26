import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.Visualization.VisualizationBase import VisualizationBase

plt.rc("text", usetex=True)


class HeatmapVisualizer(VisualizationBase):
    def __init__(self, df, **kwargs):
        self.df = df
        super().__init__(**kwargs)

    def plot(self, x, y, z, query=None, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if kwargs.get("newFig", True):
            self.newFig()

        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # query the dataframe
        if query is not None:
            df = self.df.query(query)
        else:
            df = self.df.copy()

        data_df = df.pivot_table(index=x, columns=y, values=z, fill_value=0)
        data = data_df.values

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(np.unique(data_df.columns))
        ax.set_yticklabels(np.unique(data_df.index))

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        plt.grid(False)

        return im, cbar


class Heatmap3DVisualizer(VisualizationBase):
    def __init__(self, df, **kwargs):
        self.df = df
        super().__init__(**kwargs)
        plt.rc("font", size=6.0)
        self.fig_size = (self.w_tri, self.w_tri)

    def plot(
        self,
        x,
        y,
        z,
        query=None,
        vmin=None,
        vmax=None,
        x_base2=True,
        y_base2=True,
        **kwargs,
    ):
        if kwargs.get("newFig", True):
            fig, ax = self.new3DFig()
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_zlabel(None)
        # query the dataframe
        if query is not None:
            df = self.df.query(query)
        else:
            df = self.df.copy()

        data_df = df.pivot_table(index=x, columns=y, values=z, fill_value=np.nan)

        df_stacked = data_df.stack()
        index = np.array(df_stacked.index.tolist()).astype(float)

        x_data, y_data = index[:, 0], index[:, 1]
        if x_base2:
            x_data = np.log2(x_data)
        if y_base2:
            y_data = np.log2(y_data)

        dx = np.diff(np.unique(x_data))[0]
        dy = np.diff(np.unique(y_data))[0]
        # if x_base2:
        #     dx = np.ones_like(x_data) * 1
        # if y_base2:
        #     dy = np.ones_like(y_data) * 1

        z_data = vmin
        if vmin is None:
            vmin = data_df.values.min()

        # difference the value from the minimum
        dz = df_stacked.values - vmin
        if vmax is not None:
            # dz = 0 should be green, dz = vmax should be red
            # the maximum dz should be vmax - vmin
            dz = np.clip(dz, 0, vmax - vmin)

        errors = df_stacked.values

        cmap = mpl.cm.RdYlGn.reversed()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(errors))
        ax = plt.gca()
        ax.bar3d(
            x_data - dx / 2,
            y_data - dy / 2,
            z_data,
            dx,
            dy,
            dz,
            color=colors,
        )

        ax.set_zlim([vmin, vmax])

        el = kwargs.get("elev", 15)
        az = kwargs.get("azim", 45)
        ax.view_init(elev=el, azim=az)

        ax.set_xticks(np.unique(x_data))
        ax.set_yticks(np.unique(y_data))

        if x_base2:
            ax.set_xticklabels(
                ["$2^{" + str(int(i)) + "}$" for i in np.unique(x_data)],
            )
        if y_base2:
            ax.set_yticklabels(
                ["$2^{" + str(int(i)) + "}$" for i in np.unique(y_data)],
            )

        if kwargs.get("annotate_key", None) is not None:
            annotate_key = kwargs.get("annotate_key")
            annotate_df = df.pivot_table(
                index=x,
                columns=y,
                values=annotate_key,
                fill_value=np.nan,
            )
            annotate_df = annotate_df.stack()

            bbox = dict(boxstyle="round", fc="0", alpha=0.2, ec=None)
            for i in range(len(dz)):
                text = annotate_df.values[i]
                ax.text(
                    x_data[i],
                    y_data[i],
                    dz[i],
                    text,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=3,
                    bbox=bbox,
                )

            plt.gcf().text(
                0.85,
                0.85,
                s=annotate_key,
                horizontalalignment="right",
                verticalalignment="top",
                bbox=bbox,
                fontsize=3,
            )

        # move the ticks closer to the axis
        ax.tick_params(pad=-5)

        x_formatted = " ".join(x.split("_"))
        y_formatted = " ".join(y.split("_"))

        ax.set_xlabel(x_formatted, labelpad=-10, loc="right")
        ax.set_ylabel(y_formatted, labelpad=-10, loc="top")
        plt.gcf().tight_layout(pad=0.0)

        return


def main():
    df_file = "Data/Dataframes/epochs_N_search_all_metrics.data"
    df = pd.read_pickle(df_file)

    df["percent_mean"] = df["percent_mean"] * 100
    v_min = df["percent_mean"].min()
    v_max = df["percent_mean"].max()

    vis = HeatmapVisualizer(df)
    query = "num_units == 10"
    vis.plot(
        x="epochs",
        y="N_train",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )

    vis = Heatmap3DVisualizer(df)
    query = "num_units == 10"
    vis.plot(
        x="epochs",
        y="N_train",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )

    plt.show()


if __name__ == "__main__":
    main()
