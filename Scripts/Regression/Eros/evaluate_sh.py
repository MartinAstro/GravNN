import glob

import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Trajectories import RandomDist, SurfaceDist


def get_file_info(model_path):
    directories = model_path.split("/")
    directories = directories[-4:]
    N = int(directories[0].split("_")[1])
    M = int(directories[1].split("_")[1])
    N_data = int(directories[2])
    entries = directories[3].split("_")[1].split(".")
    noise = float(entries[0] + "." + entries[1])
    return N, M, N_data, noise


def evaluate_sh_suite(trajectory, dist_name):
    models = glob.glob(
        "GravNN/Files/GravityModels/Regressed/Eros/RandomDist/BLLS/**/**/**/**.csv",
    )
    planet = Eros()
    x, a_true, u = get_poly_data(
        trajectory,
        planet.obj_200k,
        point_mass_removed=[False],
    )
    df = pd.DataFrame()
    for model in models:
        N, M, N_data, noise = get_file_info(model)
        x, a_regress, u = get_sh_data(
            trajectory,
            model,
            max_deg=N,
            deg_removed=-1,
            override=[True],
        )
        a_error = (
            np.linalg.norm(a_regress - a_true, axis=1)
            / np.linalg.norm(a_true, axis=1)
            * 100
        )
        a_avg = np.average(a_error)
        a_std = np.std(a_error)
        a_max = np.max(a_error)
        row = pd.DataFrame.from_dict(
            {
                "N": [N],
                "M": [M],
                "N_data": [N_data],
                "noise": [noise],
                "mean_error": [a_avg],
                "std_error": [a_std],
                "max_error": [a_max],
            },
        )
        df = df.append(row)
    df.to_pickle("Data/Dataframes/BLLS_" + dist_name + "_stats.data")


def main():
    planet = Eros()
    min_radius = planet.radius
    max_radius = planet.radius * 3
    trajectory = RandomDist(planet, [min_radius, max_radius], 20000, planet.obj_200k)
    dist_name = "r_outer"
    evaluate_sh_suite(trajectory, dist_name)

    min_radius = 0
    max_radius = planet.radius
    trajectory = RandomDist(planet, [min_radius, max_radius], 20000, planet.obj_200k)
    dist_name = "r_inner"
    evaluate_sh_suite(trajectory, dist_name)

    trajectory = SurfaceDist(planet, planet.obj_200k)
    dist_name = "r_surface"
    evaluate_sh_suite(trajectory, dist_name)


if __name__ == "__main__":
    main()
