import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from GravNN.CelestialBodies.Planets import Earth
from GravNN.Visualization.FigureSupport import nn_pareto_curve, sh_pareto_curve
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)


def conference_compactness():
    planet = Earth()
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/OneOff/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    # ! Spherical Harmonics Results
    sh_pareto_curve("Data/Dataframes/sh_stats_DH_Brillouin.data", max_deg=None)
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    # ! Neural Network Results
    nn_pareto_curve(
        "Data/Dataframes/N_1000000_Rand_Study.data",
        radius_max=planet.radius + 420000,
        orbit_name="Brillouin",
        linestyle="--",
    )
    vis.save(fig, "NN_Brill_Params.pdf")

    # ! PINN Neural Network Results
    nn_pareto_curve(
        "Data/Dataframes/N_1000000_PINN_study.data",
        radius_max=planet.radius + 420000,
        orbit_name="Brillouin",
        marker="o",
    )
    vis.save(fig, "NN_Brill_PINN_Params.pdf")

    # ! Optimized PINN Neural Network Results
    nn_pareto_curve(
        "Data/Dataframes/N_1000000_PINN_study_opt.data",
        radius_max=planet.radius + 420000,
        orbit_name="Brillouin",
        marker="v",
    )
    vis.save(fig, "NN_Brill_PINN_Opt_Params.pdf")


def journal_compactness():
    planet = Earth()
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/OneOff/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    # Journal Figures
    sh_pareto_curve("Data/Dataframes/sh_stats_Brillouin.data", max_deg=None)
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    # ! Neural Network Results
    nn_pareto_curve(
        "Data/Dataframes/traditional_nn_df.data",
        radius_max=planet.radius + 420000,
        orbit_name="Brillouin",
        linestyle="--",
    )
    vis.save(fig, "NN_Brill_Params.pdf")

    nn_pareto_curve(
        "Data/Dataframes/pinn_df.data",
        radius_max=planet.radius + 420000,
        orbit_name="Brillouin",
        marker="o",
    )
    vis.save(fig, "NN_Brill_PINN_Params.pdf")


def PINN_III_compactness():
    planet = Earth()
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/OneOff/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    # Journal Figures
    sh_pareto_curve("Data/Dataframes/sh_stats_Brillouin.data", max_deg=None)
    plt.legend()
    # vis.save(fig, "Brill_Params.pdf")

    # ! PINN I Neural Network Results
    nn_pareto_curve(
        "Data/Dataframes/pinn_df.data",
        radius_max=planet.radius + 420000,
        orbit_name="Brillouin",
        marker="--",
    )

    # ! PINN III Neural Network Results
    nn_pareto_curve(
        "Data/Dataframes/earth_revisited_032723.data",
        radius_max=planet.radius + 420000,
        orbit_name="Brillouin",
        marker="--",
    )
    # vis.save(fig, "NN_Brill_PINN_Params.pdf")


def main():
    # conference_compactness()
    # journal_compactness()
    PINN_III_compactness()
    plt.show()


if __name__ == "__main__":
    main()
