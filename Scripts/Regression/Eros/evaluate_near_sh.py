import glob
import os

import numpy as np
import pandas as pd

import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Trajectories import RandomDist
from GravNN.Trajectories.SurfaceDist import SurfaceDist


def evaluate_sh(trajectory, shape_file, model):
    max_deg = int(os.path.basename(model).split("_")[1])
    x, a_true, u = get_poly_data(trajectory, shape_file, point_mass_removed=[False])
    x, a, u = get_sh_data(
        trajectory,
        model,
        max_deg=max_deg,
        deg_removed=-1,
        override=[True],
    )
    a_error = np.linalg.norm(a - a_true, axis=1) / np.linalg.norm(a_true, axis=1) * 100
    return np.average(a_error)


def main(df_path):
    planet = Eros()
    R = planet.radius
    N_samples = 2000  # 20000
    df = pd.read_pickle(df_path)

    for i in range(len(df)):
        model = df.iloc[i]["file_name"]

        outer_trajectory = RandomDist(
            planet,
            [R, 3 * R],
            N_samples,
            shape_model=planet.obj_8k,
        )
        outer_avg_error = evaluate_sh(outer_trajectory, planet.obj_8k, model)

        inner_trajectory = RandomDist(
            planet,
            [0, R],
            N_samples,
            shape_model=planet.obj_8k,
        )
        inner_avg_error = evaluate_sh(inner_trajectory, planet.obj_8k, model)

        surface_trajectory = SurfaceDist(planet, planet.obj_8k)
        surface_avg_error = evaluate_sh(surface_trajectory, planet.obj_8k, model)

        error_dict = {
            "outer_avg_error": outer_avg_error,
            "inner_avg_error": inner_avg_error,
            "surface_avg_error": surface_avg_error,
        }
        # update dataframe in place to include these new columns
        df.loc[df.index[i], error_dict.keys()] = error_dict.values()

    new_df_path = df_path.split(".data")[0] + "_metrics.data"
    df.to_pickle(new_df_path)


if __name__ == "__main__":
    gravnn_dir = os.path.dirname(GravNN.__file__) + "/../"
    search = f"{gravnn_dir}Data/Dataframes/eros_sh_regression_*.data"
    files = glob.glob(search)
    for file in files:
        if "metric" in file:
            continue
        main(file)
