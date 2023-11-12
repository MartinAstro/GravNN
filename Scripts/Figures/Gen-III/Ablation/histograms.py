import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Visualization.HeatmapVisualizer import Heatmap3DVisualizer
from GravNN.Visualization.VisualizationBase import VisualizationBase


def width_depth():
    df_file = "Data/Dataframes/all_ablation_width_depth.data"
    df = pd.read_pickle(df_file)

    # Identify the length of the layers list
    df["layers_length"] = df["layers"].apply(lambda x: len(x) - 2)

    df = df.rename(
        columns={
            "num_units": "Nodes",
            "layers_length": "Layers",
            # "val_loss" : "Percent Error",
        },
    )

    df.val_loss = df.val_loss.astype(float) * 100
    # df.time_delta = df.time_delta.astype(int)
    v_min = 0  # np.min(df.val_loss)
    v_max = 3

    vis = Heatmap3DVisualizer(df)
    vis.fig_size = (vis.w_half, vis.w_half)

    vis.plot(
        x="Nodes",
        y="Layers",
        z="val_loss",
        vmin=v_min,
        vmax=v_max,
        y_base2=False,
        annotate_key="params",
    )

    file_name = df_file.split("/")[-1].split("all_ablation_")[-1].split(".")[0]
    vis.save(plt.gcf(), file_name)


def batch_learning():
    df_file = "Data/Dataframes/all_ablation_batch_learning.data"
    df = pd.read_pickle(df_file)

    df = df.rename(
        columns={
            "learning_rate": "Learning Rate",
            "batch_size": "Batch Size",
        },
    )

    df.val_loss = df.val_loss.astype(float) * 100
    df.time_delta = df.time_delta.astype(int)
    v_min = 0  # np.min(df.val_loss)
    v_max = 5.6

    vis = Heatmap3DVisualizer(df)
    vis.fig_size = (vis.w_half, vis.w_half)

    vis.plot(
        x="Learning Rate",
        y="Batch Size",
        z="val_loss",
        vmin=v_min,
        vmax=v_max,
        azim=-45,
        annotate_key="time_delta",
    )

    file_name = df_file.split("/")[-1].split("all_ablation_")[-1].split(".")[0]
    vis.save(plt.gcf(), file_name)


def data_epochs(df_file):
    df = pd.read_pickle(df_file)
    df = df.rename(
        columns={
            "N_train": "Data",
            "epochs": "Epochs",
            # "val_loss" : "Percent Error",
        },
    )

    df.val_loss = df.val_loss.astype(float) * 100
    df.time_delta = df.time_delta.astype(int)
    v_min = 0  # np.min(df.val_loss)
    v_max = 5.6

    vis = Heatmap3DVisualizer(df)
    vis.fig_size = (vis.w_half, vis.w_half)

    vis.plot(
        x="Data",
        y="Epochs",
        z="val_loss",
        vmin=v_min,
        vmax=v_max,
        annotate_key="time_delta",
    )
    file_name = df_file.split("/")[-1].split("all_ablation_")[-1].split(".")[0]
    vis.save(plt.gcf(), file_name)


def noise_loss(df_file, linestyle="-"):
    df = pd.read_pickle(df_file)
    # df = df.rename(columns={
    #     "N_train": "Data",
    #     "epochs": "Epochs",
    #     # "val_loss" : "Percent Error",
    #     })

    df.val_loss = df.val_loss.astype(float) * 100
    df.time_delta = df.time_delta.astype(int)
    df.sort_values(by=["N_train"], inplace=True)

    pinn_a = df[df["PINN_constraint_fcn"] == "pinn_a"]
    pinn_al = df[df["PINN_constraint_fcn"] == "pinn_al"]

    pinn_a.sort_values(by=["N_train"], inplace=True)
    pinn_al.sort_values(by=["N_train"], inplace=True)

    plt.plot(pinn_a.N_train, pinn_a.val_loss, label="PINN-A", linestyle=linestyle)
    plt.plot(pinn_al.N_train, pinn_al.val_loss, label="PINN-AL", linestyle=linestyle)
    plt.legend()
    plt.xlabel("Data")
    plt.ylabel("Percent Error")

    # set x-axis as base2
    ax = plt.gca()
    ax.set_xscale("log", base=2)

    vis = VisualizationBase()
    vis.save(plt.gcf(), "noise_loss")


def main():
    width_depth()
    batch_learning()
    data_epochs("Data/Dataframes/all_ablation_data_epochs_small.data")
    data_epochs("Data/Dataframes/all_ablation_data_epochs_large.data")

    # vis = VisualizationBase()
    # vis.newFig()
    # noise_loss("Data/Dataframes/all_ablation_noise_loss_small.data")
    # noise_loss("Data/Dataframes/all_ablation_noise_loss_large.data", linestyle='--')
    plt.show()


if __name__ == "__main__":
    main()
