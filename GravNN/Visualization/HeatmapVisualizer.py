import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from GravNN.Visualization.VisualizationBase import VisualizationBase
import os
plt.rc('text', usetex=True)


class HeatmapVisualizer(VisualizationBase):
    def __init__(self, df, **kwargs):
        self.df = df
        super().__init__(**kwargs)

    def plot(self, x, y, z, query=None, ax=None,
                cbar_kw=None, cbarlabel="", **kwargs):
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

        data_df = df.pivot_table(
                index=x, 
                columns=y,
                values=z, 
                fill_value=0)
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
        ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                rotation_mode="anchor")


        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        plt.grid(False)

        return im, cbar


class Heatmap3DVisualizer(VisualizationBase):
    def __init__(self, df, **kwargs):
        self.df = df
        super().__init__(**kwargs)

    def plot(self,x, y, z, query=None, vmin=None, vmax=None, **kwargs):
        
        if kwargs.get("newFig", True):
            plt.figure()

        # query the dataframe
        if query is not None:
            df = self.df.query(query)
        else:
            df = self.df.copy()

        data_df = df.pivot_table(
                index=x, 
                columns=y,
                values=z, 
                fill_value=0)


        df_stacked = data_df.stack()
        index = np.array(df_stacked.index.tolist()).astype(float)
        x_data, y_data = np.log2(index[:,0]), np.log2(index[:,1])

        z_data = vmin
        dx = np.ones_like(x_data)*1
        dy = np.ones_like(y_data)*1

        dz = df_stacked.values - vmin
        if vmax is not None:
            dz = np.clip(df_stacked.values, 0, vmax) - vmin

        errors = df_stacked.values

        plt.gcf().add_subplot(111, projection='3d')

        cmap = mpl.cm.RdYlGn.reversed()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(errors))
        ax = plt.gca()
        ax.bar3d(
            x_data, y_data, z_data,
            dx, dy, dz,
            color=colors
        )

        ax.set_zlim([vmin, vmax])
        ax.view_init(elev=15, azim=45)

        ax.set_xticklabels(["$2^{" + str(int(i)) + "}$" for i in np.unique(x_data)])
        ax.set_yticklabels(["$2^{" + str(int(i)) + "}$" for i in np.unique(y_data)])

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)

        return 


def main():
    df_file = "Data/Dataframes/epochs_N_search_all_metrics.data"
    df = pd.read_pickle(df_file)

    v_min = df['percent_mean'].min()
    v_max = df['percent_mean'].max()


    vis = HeatmapVisualizer(df)
    query = "num_units == 10"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )



    vis = Heatmap3DVisualizer(df)
    query = "num_units == 10"
    vis.plot(
            x='epochs', 
            y='N_train', 
            z='percent_mean', 
            vmin=v_min, 
            vmax=v_max, 
            query=query
        )
   
    plt.show()


if __name__ == "__main__":
    main()