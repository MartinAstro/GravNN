import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import os
plt.rc('text', usetex=True)

def heatmap3d(df, vmin, vmax):

    df_stacked = df.stack()
    index = np.array(df_stacked.index.tolist()).astype(float)
    x, y = np.log2(index[:,0]), np.log2(index[:,1])

    z = vmin# np.zeros_like(x) 
    dx = np.ones_like(x)*1
    dy = np.ones_like(y)*1
    dz = df_stacked.values - vmin

    errors = df_stacked.values

    # scaler = MinMaxScaler()
    # scaler.fit([vmin, vmax])
    # colors = scaler.transform(errors)

    # colors = df.values
    plt.gcf().add_subplot(111, projection='3d')

    cmap = mpl.cm.RdYlGn.reversed()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(errors))
    ax = plt.gca()
    ax.bar3d(
        x, y, z,
        dx, dy, dz,
        color=colors
    )

    ax.set_zlim([vmin, vmax])
    ax.view_init(elev=15, azim=45)

    ax.set_xticklabels(["$2^{" + str(int(i)) + "}$" for i in np.unique(x)])
    ax.set_yticklabels(["$2^{" + str(int(i)) + "}$" for i in np.unique(y)])

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Data")
    ax.set_zlabel("Mean Percent Error", rotation=180)



    return 

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
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

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def make_grid(df, query):
    if query is not None:
        sub_df = df.query(query)
    else:
        sub_df = df.copy()
    sub_df = sub_df.pivot_table(columns='N_train', index='epochs', 
                            values='percent_mean', fill_value=0)
    return sub_df

def plot(df, query, **kwargs):
    sub_df = make_grid(df, query)
    plt.figure()
    im, cbar = heatmap(sub_df.values,
                    row_labels=sub_df.index,
                    col_labels=sub_df.columns,
                    **kwargs
                    )
    return plt.gcf()

def plot3d(df, query, **kwargs):
    sub_df = make_grid(df, query)
    plt.figure()
    heatmap3d(sub_df,
            **kwargs
                    )
    return plt.gcf()

def save(name):
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')



def main():
    df_file = "Data/Dataframes/epochs_N_search_all_metrics.data"
    df = pd.read_pickle(df_file)

    v_min = df['percent_mean'].min()
    v_max = df['percent_mean'].max()

    query = "fourier_features == 20 & freq_decay == False"

    os.makedirs("Plots/PINNIII/", exist_ok=True)
    query = "num_units == 10"
    plot3d(df, query, vmin=v_min, vmax=v_max)
    save("Plots/PINNIII/NvE_10.pdf")

    query = "num_units == 20"
    plot3d(df, query, vmin=v_min, vmax=v_max)
    save("Plots/PINNIII/NvE_20.pdf")

    query = "num_units == 40"
    plot3d(df, query, vmin=v_min, vmax=v_max)
    save("Plots/PINNIII/NvE_40.pdf")
    
    query = "num_units == 80"
    plot3d(df, query, vmin=v_min, vmax=v_max)
    save("Plots/PINNIII/NvE_80.pdf")

    plt.show()


if __name__ == "__main__":
    main()