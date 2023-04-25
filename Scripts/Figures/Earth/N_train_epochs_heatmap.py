import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Visualization.HeatmapVisualizer import Heatmap3DVisualizer


def PINN_III():
    df_file = "Data/Dataframes/epochs_N_search_all_metrics.data"
    df = pd.read_pickle(df_file)
    df = df.rename(columns={"epochs": "Epochs", "N_train": "Samples"})
    df.percent_mean = df.percent_mean * 100

    v_min = df["percent_mean"].min()
    v_max = df["percent_mean"].max()

    vis = Heatmap3DVisualizer(df)
    query = "num_units == 10"
    vis.plot(
        x="Epochs",
        y="Samples",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )
    vis.save(plt.gcf(), "PINNIII/NvE_10.pdf")

    query = "num_units == 20"
    vis.plot(
        x="Epochs",
        y="Samples",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )
    vis.save(plt.gcf(), "PINNIII/NvE_20.pdf")

    query = "num_units == 40"
    vis.plot(
        x="Epochs",
        y="Samples",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )
    vis.save(plt.gcf(), "PINNIII/NvE_40.pdf")

    query = "num_units == 80"
    vis.plot(
        x="Epochs",
        y="Samples",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )
    vis.save(plt.gcf(), "PINNIII/NvE_80.pdf")


def PINN_I():
    df_file = "Data/Dataframes/epochs_N_search_PINN_I_metrics.data"
    df = pd.read_pickle(df_file)
    df = df.rename(columns={"epochs": "Epochs", "N_train": "Samples"})
    df.percent_mean = df.percent_mean * 100

    df_file_PINN_III = "Data/Dataframes/epochs_N_search_all_metrics.data"
    df_PINN_III = pd.read_pickle(df_file_PINN_III)
    df_PINN_III.percent_mean = df_PINN_III.percent_mean * 100

    # scale by PINN III Results
    v_min = df_PINN_III["percent_mean"].min()
    v_max = df_PINN_III["percent_mean"].max()

    vis = Heatmap3DVisualizer(df)
    query = "num_units == 10"
    vis.plot(
        x="Epochs",
        y="Samples",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )
    vis.save(plt.gcf(), "PINNI/NvE_10.pdf")

    query = "num_units == 20"
    vis.plot(
        x="Epochs",
        y="Samples",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )
    vis.save(plt.gcf(), "PINNI/NvE_20.pdf")

    query = "num_units == 40"
    vis.plot(
        x="Epochs",
        y="Samples",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )
    vis.save(plt.gcf(), "PINNI/NvE_40.pdf")

    query = "num_units == 80"
    vis.plot(
        x="Epochs",
        y="Samples",
        z="percent_mean",
        vmin=v_min,
        vmax=v_max,
        query=query,
    )
    vis.save(plt.gcf(), "PINNI/NvE_80.pdf")


def main():
    PINN_III()
    PINN_I()
    plt.show()


if __name__ == "__main__":
    main()
