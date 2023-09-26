import os

import matplotlib.pyplot as plt
import pandas as pd

import GravNN
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer


def main():
    # df = pd.read_pickle("Data/Dataframes/eros_pinn_II_III.data")
    directory = os.path.dirname(GravNN.__file__) + "/../"
    df = pd.read_pickle(f"{directory}Data/Dataframes/eros_pinn_II_III_warm_start.data")

    model_id = df["id"].values[0]  # PINN-II - A
    model_id = df["id"].values[2]  # PINN-II - ALC
    print(model_id)
    config, model = load_config_and_model(df, model_id)
    extrapolation_exp = ExtrapolationExperiment(model, config, 500)
    extrapolation_exp.run()
    vis = ExtrapolationVisualizer(
        extrapolation_exp,
        save_directory=os.path.abspath(".") + "/Plots/Eros/",
        annotate=False,
    )
    vis.fig_size = vis.tri_page
    vis.plot_interpolation_percent_error()
    vis.plot_extrapolation_percent_error()
    plt.gca().set_yscale("log")
    # plt.gca().set_ylim([0, 100])
    # vis.save(plt.gcf(), "PINN_II_extrapolation.pdf")

    # model_id = df["id"].values[1] # PINN-III - A
    model_id = df["id"].values[3]  # PINN-III - ALC
    # model_id = df["id"].values[4] # PINN-III - A -- wrong
    print(model_id)
    config, model = load_config_and_model(df, model_id)
    extrapolation_exp = ExtrapolationExperiment(model, config, 500)
    extrapolation_exp.run()
    vis = ExtrapolationVisualizer(
        extrapolation_exp,
        save_directory=os.path.abspath(".") + "/Plots/Eros/",
        annotate=False,
    )
    vis.fig_size = vis.tri_page
    vis.plot_interpolation_percent_error()
    vis.plot_extrapolation_percent_error()
    plt.gca().set_yscale("log")
    # plt.gca().set_ylim([0, 100])

    # vis.save(plt.gcf(), "PINN_III_extrapolation.pdf")

    plt.show()


if __name__ == "__main__":
    main()
