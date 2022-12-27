import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def make_grid(df):    
    sigmas = np.array(df['fourier_sigma'].values.tolist()).astype(float)
    metric = np.array(df['percent_mean'].values.tolist()).astype(float)

    df['FF_1'] = sigmas[:,0]
    df['FF_2'] = sigmas[:,1]

    new_df = df.pivot_table(columns='FF_1', index='FF_2', 
                            values='percent_mean', fill_value=0)
    return new_df

def main():
    df_file = "Data/Dataframes/multiFF_hparams_sigma_metrics.data"
    df = pd.read_pickle(df_file)

    v_min = df['percent_mean'].min()
    v_max = df['percent_mean'].max()
    vlim = [v_min, v_max]

    mask = df['fourier_features'] == 20

    plt.figure()
    plt.subplot(1,2,1)
    new_df = make_grid(df[mask])
    heatmap(new_df.values, row_labels=new_df.index, col_labels=new_df.columns, vmin=v_min, vmax=v_max)
    plt.title("20 Features")

    plt.subplot(1,2,2)
    new_df = make_grid(df[~mask])
    heatmap(new_df.values, row_labels=new_df.index, col_labels=new_df.columns, vmin=v_min, vmax=v_max)
    plt.title("40 Features")
    plt.show()


if __name__ == "__main__":
    main()