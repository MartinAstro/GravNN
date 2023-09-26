import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)


def main():
    Earth()

    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/OneOff/")

    # ! RSE MEAN Full Size
    # Brillouin 0 km
    fig, ax = vis.newFig(fig_size=vis.full_page)

    def sh_pareto_curve(file_name):
        poly_df = pd.read_pickle(file_name)
        plt.semilogx(poly_df.index, poly_df["rse_mean"], label=r"MSE($\mathcal{S}$)")
        # plt.semilogx(poly_df.index, poly_df['sigma_2_mean'], label=r'MSE($\mathcal{F}$)')
        # plt.semilogx(poly_df.index, poly_df['sigma_2_c_mean'], label=r'MSE($\mathcal{C}$)')

        plt.ylabel("Mean RSE")
        plt.xlabel("Parameters")

        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    def nn_pareto_curve(file_name, orbit_name, linestyle=None, marker=None):
        nn_df = pd.read_pickle(file_name)
        sub_df = nn_df.sort_values(
            by="params",
        )  # [nn_df['radius_max'] == planet.radius + 1000.0]
        plt.gca().set_prop_cycle(None)
        plt.semilogx(
            sub_df["params"],
            sub_df[orbit_name + "_rse_mean"],
            linestyle=linestyle,
            marker=marker,
        )
        # plt.semilogx(sub_df['params'], sub_df[orbit_name+'_sigma_2_mean'], linestyle=linestyle, marker=marker)
        # plt.semilogx(sub_df['params'], sub_df[orbit_name+'_sigma_2_c_mean'], linestyle=linestyle, marker=marker)
        plt.legend()

    sh_pareto_curve("poly_stats_eros_surface.data")

    # ! Neural Network Results
    nn_pareto_curve(
        "N_100000_rand_eros_study.data",
        orbit_name="Surface",
        linestyle="--",
    )
    # vis.save(fig, "NN_Brill_Params.pdf")

    # ! PINN Neural Network Results
    nn_pareto_curve(
        "N_100000_rand_eros_PINN_study.data",
        orbit_name="Surface",
        marker="o",
    )
    vis.save(fig, "NN_Surface_PINN_Params.pdf")

    plt.show()


if __name__ == "__main__":
    main()
