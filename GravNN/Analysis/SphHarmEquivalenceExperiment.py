import os
import pickle
from pprint import pprint

import numpy as np
import pandas as pd

import GravNN
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Networks.Model import load_config_and_model
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


def compute_stats(
    state_obj_true,
    state_obj_pred,
    stat_types=["mean", "std", "median"],
    percent=False,
):
    da = state_obj_pred - state_obj_true

    if percent:
        da = da / state_obj_true * 100

    # calculate the masks
    sigma_1_mask, sigma_1_mask_compliment = sigma_mask(state_obj_true.total, 1)
    sigma_2_mask, sigma_2_mask_compliment = sigma_mask(state_obj_true.total, 2)
    sigma_3_mask, sigma_3_mask_compliment = sigma_mask(state_obj_true.total, 3)

    rse_stats = mean_std_median(da.total, prefix="rse", stat_types=stat_types)

    args = [
        (sigma_1_mask, "sigma_1"),
        (sigma_1_mask_compliment, "sigma_1_c"),
        (sigma_2_mask, "sigma_2"),
        (sigma_2_mask_compliment, "sigma_2_c"),
        (sigma_3_mask, "sigma_3"),
        (sigma_3_mask_compliment, "sigma_3_c"),
    ]

    results = {}
    results.update(rse_stats)

    # get the stats for each mask
    for mask, suffix in args:
        stats = mean_std_median(da.total, mask, f"{suffix}", stat_types=stat_types)
        results.update(stats)

    return results


class SphHarmEquivalenceExperiment:
    def __init__(self, model, config, truth_file):
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
        self.planet = config["planet"][0]
        self.body_type = "Planet"

        self.truth_df = truth_file
        if not isinstance(truth_file, pd.DataFrame):
            self.truth_df = pd.read_pickle(truth_file)

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

    def compute_alt_stats(self, points):
        df_all = pd.DataFrame()

        altitudes = self.truth_df.index
        for alt in altitudes:
            trajectory = FibonacciDist(self.planet, self.planet.radius + alt, points)
            sh_file = trajectory.celestial_body.sh_file
            x, acc_sh, u = get_sh_data(trajectory, sh_file, **self.config)
            acc_pinn = self.model.compute_acceleration(x)

            state_obj_true = StateObject(trajectory=trajectory, accelerations=acc_sh)
            state_obj_pred = StateObject(trajectory=trajectory, accelerations=acc_pinn)

            stats = compute_stats(state_obj_true, state_obj_pred)
            extras = {
                "alt": [alt],
                # "max_error": [np.max(da.total)],
            }
            stats.update(extras)

            # Check for the nearest SH in altitude
            keys = [
                "rse_mean",
                # "sigma_1_mean",
                # "sigma_1_c_mean",
                "sigma_2_mean",
                "sigma_2_c_mean",
                "sigma_3_mean",
                "sigma_3_c_mean",
            ]
            analytic_neighbors = {}
            df_alt = self.truth_df.loc[alt]
            for key in keys:
                nearest_sh = nearest_analytic(df_alt["param_" + key], stats[f"{key}"])
                analytic_neighbors.update({f"param_{key}": [nearest_sh]})

            stats.update(analytic_neighbors)
            df = pd.DataFrame().from_dict(stats).set_index("alt")
            df_all = df_all.append(df)
        pprint(df_all)

        rse_path = f"/../Data/Networks/{model_id}/rse_alt.data"
        df_all.to_pickle(os.path.dirname(GravNN.__file__) + rse_path)
        return df_all

    def run(self, points=5000000):
        self.compute_alt_stats(points)


if __name__ == "__main__":
    df = pd.read_pickle("Data/Dataframes/earth_trainable_FF.data")
    truth_df = pd.read_pickle("Data/Dataframes/sh_stats_earth_altitude_v2.data")
    model_id = df.id[-1]
    config, model = load_config_and_model(df, model_id)
    exp = SphHarmEquivalenceExperiment(model, config, truth_df)
    exp.run(points=500)
