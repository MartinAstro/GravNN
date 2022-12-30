import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(df, x_label, y_label, z_label, log_x_2=False, log_y_2=False):
    plt.figure()
    x = np.unique(df[z_label])
    markers_list = ["o", "^", "+", "D", "x"]
    for i, z_i in enumerate(x):
        sub_df = df[df[z_label]==z_i]
        plt.plot(
                sub_df[x_label],
                sub_df[y_label],
                marker=markers_list[i],
                label=f'{z_label}={str(z_i)}'
                )
    plt.legend()
    plt.grid()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if log_x_2:
        plt.gca().set_xscale('log', basex=2)
    if log_y_2:
        plt.gca().set_yscale('log', basey=2)

def main():

    df = pd.read_pickle("Data/Dataframes/epochs_N_search_metrics.data")
    plot(df, 'epochs', 'percent_mean', 'N_train', True, False)
    plot(df, 'N_train', 'percent_mean', 'epochs', True, False)
        
    df = pd.read_pickle("Data/Dataframes/epochs_N_search_20_metrics.data")
    plot(df, 'epochs', 'percent_mean', 'N_train', True, False)
    plot(df, 'N_train', 'percent_mean', 'epochs', True, False)

    plt.show()


if __name__ == "__main__":
    main()