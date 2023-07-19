import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Trajectories import RandomDist, SurfaceDist


def evaluate_nn(trajectory, model_file, model):
    x, a_true, u = get_poly_data(trajectory, model_file, point_mass_removed=[False])
    a = model.compute_acceleration(trajectory.positions.astype(np.float32))
    a_error = np.linalg.norm(a - a_true, axis=1) / np.linalg.norm(a_true, axis=1) * 100
    return np.average(a_error)


def main(df_path):
    df = pd.read_pickle(df_path)

    planet = Eros()
    R = planet.radius
    N_samples = 2000  # 20000

    for i in range(len(df)):
        id = df["id"].values[i]
        config, model = load_config_and_model(id, df)

        outer_trajectory = RandomDist(
            planet,
            [R, 3 * R],
            N_samples,
            shape_model=planet.obj_8k,
        )
        outer_avg_error = evaluate_nn(outer_trajectory, planet.obj_8k, model)

        inner_trajectory = RandomDist(
            planet,
            [0, R],
            N_samples,
            shape_model=planet.obj_8k,
        )
        inner_avg_error = evaluate_nn(inner_trajectory, planet.obj_8k, model)

        surface_trajectory = SurfaceDist(planet, planet.obj_8k)
        surface_avg_error = evaluate_nn(surface_trajectory, planet.obj_8k, model)

        error_dict = {
            "outer_avg_error": outer_avg_error,
            "inner_avg_error": inner_avg_error,
            "surface_avg_error": surface_avg_error,
        }
        # update dataframe in place to include these new columns
        df.loc[df.index[i], error_dict.keys()] = error_dict.values()

    new_df_path = df_path.split(".data")[0] + "_metrics.data"
    df.to_pickle(new_df_path)
    plt.show()


if __name__ == "__main__":
    gravnn_dir = os.path.dirname(GravNN.__file__) + "/../"
    search = f"{gravnn_dir}Data/Dataframes/eros_regression_*.data"
    files = glob.glob(search)
    for file in files:
        if "metric" in file:
            continue
        main(file)
