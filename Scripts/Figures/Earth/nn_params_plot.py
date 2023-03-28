import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from GravNN.CelestialBodies.Planets import Earth
from GravNN.Visualization.FigureSupport import nn_pareto_curve, sh_pareto_curve
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)


def filter_dataframe(file_name, max_deg=None, radius_max=None):
    df = pd.read_pickle(file_name)
    if max_deg is not None:
        sub_df = df.loc[:max_deg]
    if radius_max is not None:
        sub_df = df[df["radius_max"] == radius_max].sort_values(by="params")
    return sub_df


def conference_compactness():
    planet = Earth()
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/OneOff/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    # ! Spherical Harmonics Results
    df = filter_dataframe("Data/Dataframes/sh_stats_DH_Brillouin.data", max_deg=None)
    sh_pareto_curve(df)
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    args = [
        ("N_1000000_Rand_Study.data", "--", None, "NN_Brill_Params"),
        ("N_1000000_PINN_study.data", None, "o", "NN_Brill_PINN_Params"),
        ("N_1000000_PINN_study_opt.data", None, "v", "NN_Brill_PINN_Opt_Params"),
    ]

    for df_file, linestyle, marker, plot_name in args:
        df = filter_dataframe(df_file, radius_max=planet.radius + 420000)
        nn_pareto_curve(df, orbit_name="Brillouin", linestyle=linestyle, marker=marker)
        vis.save(fig, f"{plot_name}.pdf")


def journal_compactness():
    planet = Earth()
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/OneOff/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    # Journal Figures
    df = filter_dataframe(
        "Data/Dataframes/sh_stats_Brillouin.data",
        max_deg=None,
    )
    sh_pareto_curve(
        df,
    )
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    args = [
        ("traditional_nn_df.data", "--", None, "NN_Brill_Params"),
        ("pinn_df.data", None, "o", "NN_Brill_PINN_Params"),
    ]

    for df_file, linestyle, marker, plot_name in args:
        df = filter_dataframe(df_file, radius_max=planet.radius + 420000)
        nn_pareto_curve(df, orbit_name="Brillouin", linestyle=linestyle, marker=marker)
        vis.save(fig, f"{plot_name}.pdf")


def main():
    # conference_compactness()
    journal_compactness()
    plt.show()


if __name__ == "__main__":
    main()
