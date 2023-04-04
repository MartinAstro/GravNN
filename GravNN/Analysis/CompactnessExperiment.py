import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.utils import update_df_row
from GravNN.Support.StateObject import StateObject
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from GravNN.Trajectories.FibonacciDist import FibonacciDist


class PINN_m_C22:
    def __init__(self, PINN, sh_info, max_deg):
        self.model = PINN
        self.SH_model = SphericalHarmonics(sh_info, max_deg)
        self.max_deg = max_deg

    def compute_acceleration(self, x):
        PINN_acc = self.model.compute_acceleration(x)
        SH_acc = self.SH_model.compute_acceleration(x)
        return PINN_acc - SH_acc


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
    for mask, column in args:
        stats = mean_std_median(da.total, mask, column, stat_types=stat_types)
        results.update(stats)

    return results


class CompactnessExperiment:
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
            entry_value = nearest_analytic(self.truth_df[key], map_stats[key])
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

            state_obj_true = StateObject(trajectory=traj_data, accelerations=acc_sh)
            state_obj_pred = StateObject(trajectory=traj_data, accelerations=acc_pinn)

            # Generate map statistics on sets A, F, and C (2 and 3 sigma)
            stats = compute_stats(state_obj_true, state_obj_pred)

            # Calculate the spherical harmonic degree that yields approximately
            # the same statistics
            analytic_neighbors = self.compute_nearest_analytic(traj_name, stats)
            stats.update(analytic_neighbors)
        return stats

    def run(self, test_trajectories):
        stats = self.compute_rse_stats(test_trajectories)
        return stats


if __name__ == "__main__":
    df_name = "Data/Dataframes/earth_PINN_III_FF_040423.data"
    df = pd.read_pickle(df_name)
    # truth_df = pd.read_pickle("Data/Dataframes/sh_stats_Brillouin_deg_n1.data")
    truth_df = pd.read_pickle("Data/Dataframes/sh_stats_Brillouin.data")

    planet = Earth()
    df_name_truth = "Brillouin_deg_n1"  # no deg removed
    test_trajectories = {
        df_name_truth: FibonacciDist(planet, planet.radius, points=50000),
    }

    for idx in range(len(df)):
        model_id = df.id.values[idx]
        planet = df.planet.values[idx]

        config, model = load_config_and_model(model_id, df)
        config["deg_removed"] = [2]
        PINN_m_C22_model = PINN_m_C22(model, planet.sh_file, 2)
        exp = CompactnessExperiment(PINN_m_C22_model, config, truth_df)
        stats = exp.run(test_trajectories)
        df = update_df_row(model_id, df, stats, save=False)
    df.to_pickle(df_name)
