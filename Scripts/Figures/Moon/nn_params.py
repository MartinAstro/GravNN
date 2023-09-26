import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Moon
from GravNN.Visualization.FigureSupport import nn_pareto_curve, sh_pareto_curve
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)


def journal_compactness():
    # TODO: Need to generate network dataframes
    planet = Moon()
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/Moon/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    sh_pareto_curve(
        "Data/Dataframes/sh_stats_moon_Brillouin.data",
        max_deg=None,
        sigma=2,
    )
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    # ! Neural Network Results
    nn_pareto_curve(
        "Data/Dataframes/moon_traditional_nn_df.data",
        radius_max=planet.radius + 50000,
        orbit_name="Brillouin",
        linestyle="--",
        sigma=2,
    )
    vis.save(fig, "NN_Brill_Params.pdf")

    # ! PINN Neural Network Results
    nn_pareto_curve(
        "Data/Dataframes/moon_pinn_df.data",
        radius_max=planet.radius + 50000,
        orbit_name="Brillouin",
        marker="o",
        sigma=2,
    )
    vis.save(fig, "NN_Brill_PINN_Params.pdf")


def main():
    journal_compactness()
    plt.show()


if __name__ == "__main__":
    main()
