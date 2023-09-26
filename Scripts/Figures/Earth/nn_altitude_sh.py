import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.ExponentialDist import ExponentialDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Visualization.SphHarmEquivalenceVisualizer import (
    SphHarmEquivalenceVisualizer,
)
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["blue", "green", "red", "blue", "green", "red"],
)


def extract_sub_df(trajectory, df):
    sub_df = df[df["distribution"] == trajectory.__class__]
    sub_df = sub_df[
        (sub_df["num_units"] == 20)
        | (sub_df["num_units"] == 40)
        | (sub_df["num_units"] == 80)
    ].sort_values(by="num_units", ascending=False)
    return sub_df


def main():
    vis = VisualizationBase(save_directory=os.path.abspath(".") + "/Plots/")

    planet = Earth()
    points = 5000000
    radius_bounds = [planet.radius, planet.radius + 420000.0]

    trad_df = pd.read_pickle("Data/Dataframes/traditional_nn_df.data")
    pinn_df = pd.read_pickle("Data/Dataframes/pinn_df.data")
    df = pd.concat([trad_df, pinn_df])

    # * Random Brillouin
    bounds = [0, 270]
    traj = RandomDist(planet, radius_bounds, points)

    # Find all networks that trained on that trajectory
    sub_df = extract_sub_df(traj, df)

    vis = SphHarmEquivalenceVisualizer(sub_df)
    vis.plot(sub_df, statistic="param_rse_mean", legend=True, bounds=bounds)
    vis.save(plt.gcf(), "Generalization/Random_nn_sh_altitude_equivalence.pdf")

    vis.plot(sub_df, statistic="param_sigma_2_mean", legend=True, bounds=bounds)
    vis.save(plt.gcf(), "Generalization/Random_sigma_nn_sh_altitude_equivalence.pdf")

    vis.plot(sub_df, statistic="param_sigma_2_c_mean", legend=True, bounds=bounds)
    vis.save(plt.gcf(), "Generalization/Random_comp_nn_sh_altitude_equivalence.pdf")

    # * Exponential
    vis.fig_size = vis.half_page
    bounds = [0, 160]
    exp_df = pd.read_pickle("Data/Dataframes/exponential_invert_dist_all.data")
    traj = ExponentialDist(
        planet,
        radius_bounds,
        points=1200000,
        scale_parameter=[420000.0 / 3.0],
        invert=[True],
    )
    exp_df = extract_sub_df(traj, df)

    vis.plot(exp_df, statistic="param_rse_mean", legend=False, bounds=bounds)
    vis.save(plt.gcf(), "Generalization/Exp_nn_sh_altitude_equivalence.pdf")

    vis.plot(exp_df, statistic="param_sigma_2_mean", legend=False, bounds=bounds)
    vis.save(plt.gcf(), "Generalization/Exp_sigma_nn_sh_altitude_equivalence.pdf")

    vis.plot(exp_df, statistic="param_sigma_2_c_mean", legend=False, bounds=bounds)
    vis.save(plt.gcf(), "Generalization/Exp_comp_nn_sh_altitude_equivalence.pdf")

    # * Exponential Narrower
    exp_df = pd.read_pickle("Data/Dataframes/exponential_invert_dist_v10.data")
    traj = ExponentialDist(
        planet,
        radius_bounds,
        points=1200000,
        scale_parameter=[420000.0 / 10.0],
        invert=[True],
    )
    exp_df = extract_sub_df(traj, df)

    vis.plot(exp_df, statistic="param_rse_mean", legend=False, bounds=bounds)
    vis.save(plt.gcf(), "Generalization/Exp_narrow_nn_sh_altitude_equivalence.pdf")

    vis.plot(exp_df, statistic="param_sigma_2_mean", legend=False, bounds=bounds)
    vis.save(
        plt.gcf(),
        "Generalization/Exp_narrow_sigma_nn_sh_altitude_equivalence.pdf",
    )

    vis.plot(exp_df, statistic="param_sigma_2_c_mean", legend=False, bounds=bounds)
    vis.save(plt.gcf(), "Generalization/Exp_narrow_comp_nn_sh_altitude_equivalence.pdf")

    plt.show()


if __name__ == "__main__":
    main()
