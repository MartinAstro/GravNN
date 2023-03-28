import os
import pickle

import numpy as np
import pandas as pd

import GravNN
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Support.StateObject import StateObject
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from GravNN.Trajectories import FibonacciDist


def nearest_analytic(map_stat_series, value):
    i = 0
    if value < map_stat_series.iloc[i]:
        while value < map_stat_series.iloc[i]:
            i += 1
            if i >= len(map_stat_series) - 1:
                return -1
    else:
        return -1
    upper_y = map_stat_series.iloc[i - 1]
    lower_y = map_stat_series.iloc[i]

    upper_x = map_stat_series.index[i - 1]  # x associated with upper bound
    lower_x = map_stat_series.index[i]  # x associated with lower bound

    slope = (lower_y - upper_y) / (lower_x - upper_x)

    line_x = np.linspace(upper_x, lower_x, 100)
    line_y = slope * (line_x - upper_x) + upper_y

    i = 0
    while value < line_y[i]:
        i += 1

    nearest_param = np.round(line_x[i])
    return nearest_param


def diff_map_and_stats(
    name,
    trajectory,
    acc_sh,
    acc_pinn,
    stat_types=["mean", "std", "median"],
    percent=False,
):
    state_obj_true = StateObject(trajectory=trajectory, accelerations=acc_sh)
    state_obj_pred = StateObject(trajectory=trajectory, accelerations=acc_pinn)
    diff = state_obj_pred - state_obj_true

    if percent:
        diff = (state_obj_pred - state_obj_true) / state_obj_true * 100

    # This ensures the same features are being evaluated independent of what degree is
    # taken off at beginning
    one_sigma_mask, one_sigma_mask_compliment = sigma_mask(state_obj_true.total, 1)
    two_sigma_mask, two_sigma_mask_compliment = sigma_mask(state_obj_true.total, 2)
    three_sigma_mask, three_sigma_mask_compliment = sigma_mask(state_obj_true.total, 3)

    rse_stats = mean_std_median(diff.total, prefix=name + "_rse", stat_types=stat_types)

    args = [
        (one_sigma_mask, "sigma_1"),
        (one_sigma_mask_compliment, "sigma_1_c"),
        (two_sigma_mask, "sigma_2"),
        (two_sigma_mask_compliment, "sigma_2_c"),
        (three_sigma_mask, "sigma_3"),
        (three_sigma_mask_compliment, "sigma_3_c"),
    ]

    results = {}
    results.update(rse_stats)

    # get the stats for each mask
    for mask, suffix in args:
        stats_mask = mean_std_median(
            diff.total,
            mask,
            f"{name}_{suffix}",
            stat_types=stat_types,
        )
        results.update(stats_mask)

    return diff, results


class PlanetAnalyzer:
    def __init__(self, model, config):
        """Analysis class used to evaluate the performance of a PINN gravity model
        trained on a celestial body typically modeled with spherical harmonics.

        The class is responsible for determining the closest spherical harmonic degree
        equivalent given the networks mean RSE both at the Brillouin sphere and at a
        specified altitude.

        Args:
            model (PINNGravityModel): compiled keras model containing network
            config (dict): hyperparameter and configuration dictionary for model
        """
        self.config = config
        self.model = model
        self.x_transformer = config["x_transformer"][0]
        self.u_transformer = config["u_transformer"][0]
        self.a_transformer = config["a_transformer"][0]
        self.body_type = "Planet"

    def compute_nearest_analytic(self, name, map_stats):
        # Compute nearest SH degree
        analytic_path = self.config["analytic_truth"][0]
        with open(f"Data/Dataframes/{analytic_path}{name}.data", "rb") as f:
            stats_df = pickle.load(f)

        keys = [
            "rse_mean",
            "rse_median",
            "sigma_1_mean",
            "sigma_1_median",
            "sigma_1_c_mean",
            "sigma_1_c_median",
            "sigma_2_mean",
            "sigma_2_median",
            "sigma_2_c_mean",
            "sigma_2_c_median",
            "sigma_3_mean",
            "sigma_3_median",
            "sigma_3_c_mean",
            "sigma_3_c_median",
        ]
        param_stats = {}
        for key in keys:
            entry_key = f"{name}_param_{key}"
            entry_value = nearest_analytic(stats_df[key], map_stats[f"{name}_{key}"])
            param_stats.update(
                {
                    entry_key: [entry_value],
                },
            )

        return param_stats

    def compute_rse_stats(self, test_trajectories, percent=False):
        stats = {}

        for traj_name, traj_data in test_trajectories.items():
            # SH Data and NN Data
            x, acc_sh, u = get_sh_data(
                traj_data,
                self.config["grav_file"][0],
                **self.config,
            )
            acc_pinn = self.model.compute_acceleration(x)

            # Generate map statistics on sets A, F, and C (2 and 3 sigma)
            diff, diff_stats = diff_map_and_stats(
                traj_name,
                traj_data,
                acc_sh,
                acc_pinn,
                percent=percent,
            )
            map_stats = {
                **diff_stats,
                traj_name + "_max_error": [np.max(diff.total)],
            }

            # Calculate the spherical harmonic degree that yields approximately
            # the same statistics
            analytic_neighbors = self.compute_nearest_analytic(traj_name, map_stats)
            stats.update(map_stats)
            stats.update(analytic_neighbors)
        return stats

    def compute_alt_stats(self, planet, altitudes, points, sh_alt_df, model_id):
        stats = {}
        df_all = pd.DataFrame()

        for alt in altitudes:
            trajectory = FibonacciDist(planet, planet.radius + alt, points)
            model_file = trajectory.celestial_body.sh_file
            x, acc_sh, u = get_sh_data(trajectory, model_file, **self.config)
            acc_pinn = self.model.compute_acceleration(x)

            diff, diff_stats = diff_map_and_stats(
                "",
                trajectory,
                acc_sh,
                acc_pinn,
                "mean",
            )
            extras = {
                "alt": [alt],
                "max_error": [np.max(diff.total)],
            }
            entries = {
                **diff_stats,
                **extras,
            }
            stats.update(entries)

            # Check for the nearest SH in altitude

            keys = [
                "rse_mean",
                "rse_median",
                "sigma_1_mean",
                "sigma_1_c_mean",
                "sigma_2_mean",
                "sigma_2_c_mean",
                "sigma_3_mean",
                "sigma_3_c_mean",
            ]
            analytic_neighbors = {}
            df_alt = sh_alt_df.loc[alt]
            for key in keys:
                entry_key = f"param_{key}"
                entry_value = nearest_analytic(df_alt[key], entries[f"_{key}"])
                analytic_neighbors.update(
                    {
                        entry_key: [entry_value],
                    },
                )

            stats.update(analytic_neighbors)
            df = pd.DataFrame().from_dict(stats).set_index("alt")
            df_all = df_all.append(df)
        print(df_all)

        df_all.to_pickle(
            os.path.dirname(GravNN.__file__)
            + "/../Data/Networks/"
            + str(model_id)
            + "/rse_alt.data",
        )
        return df_all
