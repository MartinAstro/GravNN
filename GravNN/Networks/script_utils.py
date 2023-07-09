import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.Networks.utils import update_df_row


def get_altitude_list(planet):
    if planet.__class__ == Earth().__class__:
        sh_stats_df = pd.read_pickle("Data/Dataframes/sh_stats_earth_altitude_v2.data")
        alt_list = np.linspace(
            0,
            500000,
            50,
            dtype=float,
        )  # Every 10 kilometers above surface
        window = np.array([5, 15, 45, 100, 300])  # Close to surface distribution
        alt_list = np.concatenate([alt_list, window, 420000 + window, 420000 - window])
        altitudes = np.sort(np.unique(alt_list))
    elif planet.__class__ == Moon().__class__:
        sh_stats_df = pd.read_pickle("Data/Dataframes/sh_stats_moon_altitude.data")
        altitudes = np.linspace(
            0,
            50000,
            50,
            dtype=float,
        )  # Every 1 kilometers above surface
        altitudes = np.concatenate(
            [altitudes, np.linspace(50000, 55000, 2, dtype=float)[1:]],
        )
    elif planet.__class__ == Bennu().__class__:
        exit("Not implemented yet")
    else:
        exit("Selected planet not implemented yet")
    return sh_stats_df, altitudes


def save_analysis(df_file, results):
    df = pd.read_pickle(df_file)
    for result in results:
        model_id = result[0]
        rse_entries = result[1]
        if model_id is None:
            continue
        df = update_df_row(model_id, df, rse_entries, save=False)
    df.to_pickle(df_file)


def save_training(df_file, configs):
    for config in configs:
        config = dict(sorted(config.items(), key=lambda kv: kv[0]))
        config["PINN_constraint_fcn"] = [
            config["PINN_constraint_fcn"][0],
        ]  # Can't have multiple args in each list
        df = pd.DataFrame().from_dict(config).set_index("timetag")

        try:
            df_all = pd.read_pickle(df_file)
            df_all = df_all.append(df)
            df_all.to_pickle(df_file)
        except Exception:
            df.to_pickle(df_file)
