import os

import matplotlib.pyplot as plt
import pandas as pd

import GravNN
from GravNN.Visualization.VisualizationBase import VisualizationBase


def format_model_name(df):
    def append_size(row):
        small_in_name = "Small" in row["model_name"]
        large_in_name = "Large" in row["model_name"]

        small_model = row["num_params"] < 1e3
        point_mass_model = "PM" in row["model_name"]
        if small_in_name or large_in_name:
            row["model_name"] = (
                row["model_name"].replace("_Small", "").replace("_Large", "")
            )
            row["model_name"] = (
                row["model_name"].replace("Small", "").replace("Large", "")
            )

        if point_mass_model:
            return row["model_name"] + " -"

        if small_model:
            return row["model_name"] + " S"  # " Small"
        else:
            return row["model_name"] + " L"  # " Large"

    def abbreviate_poly(row):
        if "POLYHEDRAL" in row["model_name"]:
            row["model_name"] = row["model_name"].replace("POLYHEDRAL", "POLY")
        return row["model_name"]

    df["model_name"] = df.apply(lambda row: append_size(row), axis=1)
    df["model_name"] = df.apply(lambda row: abbreviate_poly(row), axis=1)
    return df


def main(N_train, category):
    directory = os.path.dirname(GravNN.__file__)
    df = pd.read_pickle(directory + "/../Data/Comparison/all_comparisons.data")

    # drop all columns with acc_noise != 0
    df = df[df["acc_noise"] == 0]

    # get only the N_train 500
    df = df[df["N_train"] == N_train]

    df = format_model_name(df)

    # set the model_name as index
    df = df.set_index("model_name")

    # categories into big or small models
    big_models = df[df["num_params"] > 1e3].index
    df[df["num_params"] < 1e3].index

    df["model_size"] = 0
    df.loc[big_models, "model_size"] = 1

    # sort the dataframe by parameters first then by time
    df = df.sort_values(["model_size", category], ascending=[False, False])

    # df = df.sort_values(category, ascending=False)

    # plot the dt_a_batch data from the df as a histogram
    # keep the model_name as the x-axis
    vis = VisualizationBase()
    vis.newFig()

    if "dt_" in category:
        df[category] /= 1e3  # number of samples
        df[category] *= 1e3  # time in ms
        y_label = "Time (ms)"
    else:
        y_label = category

    df[category].plot(kind="bar")
    # overlay the num_params on top of the bars
    for i, v in enumerate(df["num_params"]):
        # convert i to the index of the x-axis

        plt.text(i, df[category].iloc[i], str(int(v)), ha="center", va="bottom")

    # make the x-axis the model names
    plt.xticks(rotation=70)
    plt.xlabel("Model")
    plt.ylabel(y_label)

    df["time_per_param"] = df[category] / df["num_params"]
    vis.newFig()
    df["time_per_param"].plot(kind="bar")
    plt.xticks(rotation=70)
    plt.xlabel("Model")
    plt.ylabel("Time per Parameter (ms)")


if __name__ == "__main__":
    # main(50000, "train_duration")
    # main(500, "train_duration")
    # main(50000, "dt_a_single")
    main(500, "dt_a_single")
    # main(50000, "dt_a_batch")
    # main(500, "dt_a_batch")
    plt.show()
