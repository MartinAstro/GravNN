import os

import matplotlib.pyplot as plt
import pandas as pd

import GravNN
from GravNN.Visualization.VisualizationBase import VisualizationBase


def format_model_name(df):
    def append_size(row):
        small_in_name = "Small" in row["model_name"]
        large_in_name = "Large" in row["model_name"]

        # remove small or large from model name
        if small_in_name or large_in_name:
            row["model_name"] = (
                row["model_name"].replace("_Small", "").replace("_Large", "")
            )
            row["model_name"] = (
                row["model_name"].replace("Small", "").replace("Large", "")
            )

        # categorize model
        pm_model = "PM" in row["model_name"]
        small_model = row["num_params"] < 1e3

        # if point mass model, no large or small label
        if pm_model:
            return row["model_name"] + " -"
        elif small_model:
            return row["model_name"] + " S"
        else:
            return row["model_name"] + " L"

    def abbreviate_poly(row):
        if "POLYHEDRAL" in row["model_name"]:
            row["model_name"] = row["model_name"].replace("POLYHEDRAL", "POLY")
        return row["model_name"]

    # if _ in name, remove it
    df["model_name"] = df.apply(lambda row: append_size(row), axis=1)
    df["model_name"] = df.apply(lambda row: abbreviate_poly(row), axis=1)
    df["model_name"] = df["model_name"].str.replace("_", " ")
    return df


def unpack_df(df):
    # unpack all values from list container where possible
    for col in df.columns:
        try:
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
        except:
            pass
    return df


def add_model_size(df):
    # categories into big or small models
    df["model_size"] = 0
    df[df["num_params"] < 1e3].index
    big_models = df[df["num_params"] > 1e3].index
    df.loc[big_models, "model_size"] = 1
    return df


def plot_training_time(df):
    # average the time the different acc_noise configs
    df = df.groupby(["model_name", "N_train"]).mean()
    df.reset_index(inplace=True)

    # drop polyhedral
    df = df[~df["model_name"].str.contains("POLY")]
    df = df[~df["model_name"].str.contains("PM")]

    # rename models
    df["model_name"] = df["model_name"].str.replace(" S", "").str.replace(" L", "")

    data = {}
    for N_train in df["N_train"].unique():
        for model_size in df["model_size"].unique():
            label = str(N_train) + " " + str(model_size)
            raw_data = df[(df["N_train"] == N_train) & (df["model_size"] == model_size)]
            raw_data.sort_values("model_name", inplace=True)
            data[label] = raw_data["train_duration"].values
            index = raw_data.model_name.values

    data["model_name"] = index
    df = pd.DataFrame(data)

    # sort values by model_name, then by N_train
    df.sort_values(["50000 1.0"], ascending=[False], inplace=True)

    vis = VisualizationBase()
    vis.fig_size = (vis.w_full, vis.h_tri)
    df.plot.bar(x="model_name")  # , color=colors)

    plt.ylabel("Training Time")
    plt.xlabel("Model")
    plt.yscale("log")

    plt.xticks(rotation=70)
    plt.xlabel("Model")
    plt.ylabel("Time (s)")


def plot_evaluation_time(df, category):
    # average the time for each model
    df = df.groupby("model_name").mean()

    # sort the dataframe by parameters first then by time
    df = df.sort_values(["model_size", category], ascending=[False, False])

    df[category] /= 1e3  # number of samples
    df[category] *= 1e3  # time in ms
    y_label = "Time (ms)"

    vis = VisualizationBase()
    vis.fig_size = (vis.w_full, vis.h_tri)
    vis.newFig()

    # color based on model size
    colors = ["blue" if x == 0 else "red" for x in df["model_size"]]
    plt.bar(df.index, df[category], color=colors)

    # add a legend for the colors
    plt.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="blue", label="Small"),
            plt.Rectangle((0, 0), 1, 1, color="red", label="Large"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color="black",
                alpha=0.25,
                label="\# of Params",
            ),
        ],
        loc="upper right",
    )

    plt.ylabel("Inference Time")
    plt.xlabel("Model")
    plt.yscale("log")

    # overlay the num_params on top of the bars
    for i, v in enumerate(df["num_params"]):
        plt.text(
            i,
            0.002,
            str(int(v)),
            ha="center",
            va="bottom",
            rotation=90,
            color="white",
            bbox=dict(facecolor="black", alpha=0.25, boxstyle="round,pad=0.2"),
        )

    plt.ylim([0.001, 100])
    plt.xticks(rotation=70)
    plt.xlabel("Model")
    plt.ylabel(y_label)

    plt.gca().grid(axis="x")


def main():
    time_category = "dt_a_batch"
    time_category = "dt_a_single"

    directory = os.path.dirname(GravNN.__file__)
    # df = pd.read_pickle(directory + "/../Data/Comparison/all_comparisons.data")
    df = pd.read_pickle(
        directory + "/../Data/Comparison/comparison_metrics_072324.data",
    )
    df = unpack_df(df)
    df = add_model_size(df)

    df = format_model_name(df)
    df = df.set_index("model_name")

    plot_evaluation_time(df, time_category)
    vis = VisualizationBase()
    vis.save(plt.gcf(), "inference_time")

    # plot_training_time(df)


if __name__ == "__main__":
    main()
    plt.show()
