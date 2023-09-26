import glob
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Trajectories import RandomDist
from GravNN.Trajectories.SurfaceDist import SurfaceDist


def evaluate_sh(trajectory, shape_file, model):
    max_deg = int(os.path.basename(model).split("_")[1])
    x, a_true, u = get_hetero_poly_data(
        trajectory,
        shape_file,
        point_mass_removed=[False],
    )
    x, a, u = get_sh_data(
        trajectory,
        model,
        max_deg=max_deg,
        deg_removed=-1,
        override=[True],
    )
    a_error = np.linalg.norm(a - a_true, axis=1) / np.linalg.norm(a_true, axis=1) * 100
    return np.average(a_error)


def evaluate_nn(trajectory, obj_file, model):
    x, a_true, u = get_hetero_poly_data(
        trajectory,
        obj_file,
        point_mass_removed=[False],
    )
    a = model.compute_acceleration(trajectory.positions.astype(np.float32))
    a_error = np.linalg.norm(a - a_true, axis=1) / np.linalg.norm(a_true, axis=1) * 100
    return np.average(a_error)


def evaluate_model(i, df, evaluate_method):
    planet = Eros()
    R = planet.radius
    N_samples = 20000
    model = df.iloc[i]["file_name"]

    outer_trajectory = RandomDist(
        planet,
        [R, 3 * R],
        N_samples,
        obj_file=planet.obj_8k,
    )
    outer_avg_error = evaluate_method(outer_trajectory, planet.obj_8k, model)

    inner_trajectory = RandomDist(
        planet,
        [0, R],
        N_samples,
        obj_file=planet.obj_8k,
    )
    inner_avg_error = evaluate_method(inner_trajectory, planet.obj_8k, model)

    surface_trajectory = SurfaceDist(planet, planet.obj_8k)
    surface_avg_error = evaluate_method(surface_trajectory, planet.obj_8k, model)

    error_dict = {
        "outer_avg_error": outer_avg_error,
        "inner_avg_error": inner_avg_error,
        "surface_avg_error": surface_avg_error,
    }
    # update dataframe in place to include these new columns
    return df.index[i], error_dict


def main(df_path, method):
    df = pd.read_pickle(df_path)
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            evaluate_model,
            [(i, df, method) for i in range(len(df))],
        )

    # update the original df with the results
    for i, error_dict in results:
        df.loc[i, error_dict.keys()] = error_dict.values()

    new_df_path = df_path.split(".data")[0] + "_metrics.data"
    df.to_pickle(new_df_path)


if __name__ == "__main__":
    gravnn_dir = os.path.dirname(GravNN.__file__) + "/../"

    # Evaluate SH
    search = f"{gravnn_dir}Data/Dataframes/eros_sh_regression_*.data"
    files = glob.glob(search)
    for file in files:
        if "metric" in file:
            continue
        main(file, evaluate_sh)

    # Evaluate NN
    search = f"{gravnn_dir}Data/Dataframes/eros_regression_*.data"
    files = glob.glob(search)
    for file in files:
        if "metric" in file:
            continue
        main(file, evaluate_nn)
