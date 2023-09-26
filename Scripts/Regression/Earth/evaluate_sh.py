import os

import numpy as np
import pandas as pd
import tensorflow as tf
from utils import compute_stats

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Support.StateObject import StateObject
from GravNN.Trajectories import FibonacciDist

np.random.seed(1234)
tf.random.set_seed(0)


def get_sh_data(trajectory, gravity_file, **kwargs):
    # Handle cases where the keyword wasn't properly wrapped as a list []
    try:
        max_deg = int(kwargs["max_deg"][0])
        deg_removed = int(kwargs["deg_removed"][0])
    except:
        max_deg = int(kwargs["max_deg"])
        deg_removed = int(kwargs["deg_removed"])

    Call_r0_gm = SphericalHarmonics(gravity_file, degree=max_deg, trajectory=trajectory)
    accelerations = Call_r0_gm.load(override=kwargs["override"]).accelerations
    potentials = Call_r0_gm.potentials

    Clm_r0_gm = SphericalHarmonics(
        gravity_file,
        degree=deg_removed,
        trajectory=trajectory,
    )
    accelerations_Clm = Clm_r0_gm.load(override=kwargs["override"]).accelerations
    potentials_Clm = Clm_r0_gm.potentials

    x = Call_r0_gm.positions  # position (N x 3)
    a = accelerations - accelerations_Clm
    u = np.array(potentials - potentials_Clm).reshape((-1, 1))  # potential (N x 1)

    return x, a, u


def compute_sh_regression_statistics(sh_df, sh_df_stats_file, trajectory, grid_true):
    directory = os.path.join(
        os.path.abspath("."),
        "GravNN",
        "Files",
        "GravityModels",
        "Regressed",
        "Earth",
    )
    df_all = pd.DataFrame()
    for i in range(len(sh_df)):
        row = sh_df.iloc[i]
        file_name = row["model_identifier"]
        deg = int(file_name.split("_")[1])

        # * Predict the value at the training data
        x_est, a_est, u_est = get_sh_data(
            trajectory,
            directory + "\\" + file_name,
            max_deg=deg,
            deg_removed=2,
            override=True,
        )
        grid_pred = StateObject(trajectory=trajectory, accelerations=a_est)

        entries = compute_stats(grid_true, grid_pred)
        entries.update({"deg": [deg], "params": [deg * (deg + 1)]})

        df = pd.DataFrame().from_dict(entries)
        df_all = df_all.append(df)

    df_all.index = sh_df.index
    sh_df = sh_df.join(df_all)
    sh_df.to_pickle(sh_df_stats_file)


def main():
    """Given the regressed spherical harmonic models
    (estimate_sh_BLLS.py), compute the associated error of these
    regressed representations and store in new regress_stats dataframe.
    """
    planet = Earth()
    sh_file = planet.sh_file
    max_deg = 1000

    # * Generate the true acceleration
    trajectory = FibonacciDist(planet, planet.radius, 250000)
    file_name = "Data/Dataframes/Regression/Earth_SH_regression_9500_v1"
    file_name = "Data/Dataframes/Regression/Earth_SH_regression_5000_v1"
    file_name = "Data/Dataframes/Regression/Earth_SH_regression_2500_v1"
    file_name = "Data/Dataframes/Regression/Earth_SH_regression_18000_v1"
    file_name = "Data/Dataframes/Regression/Earth_SH_regression_5000_v2"

    sh_df = pd.read_pickle(file_name + ".data")
    sh_df_stats_file = file_name + "_stats.data"

    x_truth, a_truth, u_truth = get_sh_data(
        trajectory,
        sh_file,
        max_deg=max_deg,
        deg_removed=2,
        override=False,
    )
    grid_true = StateObject(trajectory=trajectory, accelerations=a_truth)

    compute_sh_regression_statistics(sh_df, sh_df_stats_file, trajectory, grid_true)


if __name__ == "__main__":
    main()
