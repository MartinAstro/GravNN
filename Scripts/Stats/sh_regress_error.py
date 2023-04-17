import os

import numpy as np
import pandas as pd
import tensorflow as tf

import GravNN
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Networks.Data import training_validation_split
from GravNN.Regression.Regression import Regression
from GravNN.Support.StateObject import StateObject
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from GravNN.Trajectories import DHGridDist, RandomDist

np.random.seed(1234)
tf.random.set_seed(0)


def get_regress_data(model_file):
    # Generate the data
    planet = Earth()
    trajectory = RandomDist(planet, [planet.radius, planet.radius + 420000.0], 1000000)
    x, a, u = get_sh_data(trajectory, planet.sh_file, 1000, 2)
    x, a, u, x_val, a_val, u_val = training_validation_split(x, a, u, 9500, 500)
    return x, a


def regress_model(x, a, max_deg):
    planet = Earth()
    gravnn_path = os.path.dirname(GravNN.__file__)
    grav_file = gravnn_path + f"/Files/GravityModels/Regressed/regress_{max_deg}.csv"
    regressor = Regression(max_deg, planet, x, a)
    regressor.perform_regression(remove_deg=True)
    regressor.save(grav_file)


def main():
    planet = Earth()
    model_file = planet.sh_file
    density_deg = 180
    max_deg = 1000
    model_deg = 33
    gravnn_path = os.path.dirname(GravNN.__file__)
    # * Generate the true acceleration
    df_file = "Data/Dataframes/sh_regress_stats_" + str(model_deg) + "_Brillouin.data"
    trajectory = DHGridDist(planet, planet.radius, degree=density_deg)
    x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2)
    grid_true = StateObject(trajectory=trajectory, accelerations=a)

    deg_list = np.arange(3, model_deg, 1, dtype=int)
    df_all = pd.DataFrame()

    for deg in deg_list:
        # * Predict the value at the training data
        x_est, a_est, u_est = get_sh_data(
            trajectory,
            gravnn_path + f"/Files/GravityModels/Regressed/regress_{deg}.csv",
            max_deg=deg,
            deg_removed=2,
        )
        grid_pred = StateObject(trajectory=trajectory, accelerations=a_est)

        # * Difference and stats
        diff = grid_pred - grid_true

        # This ensures the same features are being evaluated independent of what degree
        # is taken off at beginning
        sigma_2_mask, sigma_2_mask_compliment = sigma_mask(grid_true.total, 2)
        sigma_3_mask, sigma_3_mask_compliment = sigma_mask(grid_true.total, 3)

        data = diff.total
        rse_stats = mean_std_median(data, prefix="rse")
        sigma_2_stats = mean_std_median(data, sigma_2_mask, "sigma_2")
        sigma_2_c_stats = mean_std_median(data, sigma_2_mask_compliment, "sigma_2_c")
        sigma_3_stats = mean_std_median(data, sigma_3_mask, "sigma_3")
        sigma_3_c_stats = mean_std_median(data, sigma_3_mask_compliment, "sigma_3_c")

        extras = {
            "deg": [deg],
            "max_error": [np.max(diff.total)],
        }

        entries = {
            **rse_stats,
            **sigma_2_stats,
            **sigma_2_c_stats,
            **sigma_3_stats,
            **sigma_3_c_stats,
            **extras,
        }
        # for d in (rse_stats, percent_stats, percent_rel_stats, sigma_2_stats,
        # sigma_2_c_stats, sigma_3_stats, sigma_3_c_stats): entries.update(d)

        df = pd.DataFrame().from_dict(entries).set_index("deg")
        df_all = df_all.append(df)

    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()
