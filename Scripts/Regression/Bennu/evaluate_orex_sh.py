import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Trajectories import RandomDist
from GravNN.Trajectories.SurfaceDist import SurfaceDist
from GravNN.Visualization.VisualizationBase import VisualizationBase


def get_file_info(file_path):
    directories = os.path.dirname(file_path).split("/")
    model_name = os.path.basename(file_path).split(".")[0]
    samples = int(model_name)
    max_deg_dir = directories[-3]
    max_deg = int(max_deg_dir.split("_")[1])
    return samples, max_deg


def evaluate_sh(planet, N, trajectory, dist_name, sampling_interval, hoppers=False):
    models = glob.glob(
        "GravNN/Files/GravityModels/Regressed/Bennu/EphemerisDist/BLLS/N_%d/**/%s/*.csv"
        % (N, str(hoppers)),
    )
    x, a_true, u = get_poly_data(
        trajectory,
        planet.stl_200k,
        point_mass_removed=[False],
    )

    sample_list = np.array([])
    error_list = np.array([])

    models.sort()
    for model in models:
        samples, max_deg = get_file_info(model)
        x, a, u = get_sh_data(
            trajectory,
            model,
            max_deg=max_deg,
            deg_removed=-1,
            override=[True],
        )
        a_error = (
            np.linalg.norm(a - a_true, axis=1) / np.linalg.norm(a_true, axis=1) * 100
        )

        sample_list = np.hstack((sample_list, samples))
        error_list = np.hstack((error_list, np.average(a_error)))

    # Sort models based on cumulative sample count
    idx = np.argsort(sample_list)
    sample_list = np.take_along_axis(sample_list, idx, axis=0)
    error_list = np.take_along_axis(error_list, idx, axis=0)

    vis = VisualizationBase()
    vis.fig_size = vis.half_page
    vis.newFig()
    plt.semilogy(sample_list * sampling_interval / (86400), error_list)
    plt.xlabel("Days Since Insertion")
    plt.ylabel("Average Acceleration Error")
    plt.ylim(1e0, np.max(error_list))

    directory = os.path.abspath(".") + "/Plots/Asteroid/Regression/"
    os.makedirs(directory, exist_ok=True)
    vis.save(
        plt.gcf(),
        directory
        + "sh_error_orex_shoemaker_"
        + str(max_deg)
        + "_"
        + dist_name
        + ".pdf",
    )

    data_directory = os.path.dirname(models[-1]) + "/Data"
    os.makedirs(data_directory, exist_ok=True)
    with open(data_directory + "/sh_estimate_" + dist_name + ".data", "wb") as f:
        pickle.dump(sample_list, f)
        pickle.dump(error_list, f)


def evaluate_sh_suite(trajectory, sampling_interval, dist_name, hoppers):
    planet = Bennu()
    dist_name += "_" + str(sampling_interval)
    evaluate_sh(planet, 4, trajectory, dist_name, sampling_interval, hoppers)
    evaluate_sh(planet, 8, trajectory, dist_name, sampling_interval, hoppers)
    evaluate_sh(planet, 16, trajectory, dist_name, sampling_interval, hoppers)


def main():
    planet = Bennu()
    hoppers = True
    trajectory = RandomDist(
        planet,
        [planet.radius, planet.radius * 3],
        20000,
        planet.stl_200k,
    )
    dist_name = "r_outer"
    sampling_interval = 10 * 60
    evaluate_sh_suite(trajectory, sampling_interval, dist_name, hoppers)

    min_radius = 0
    max_radius = planet.radius
    trajectory = RandomDist(planet, [min_radius, max_radius], 20000, planet.stl_200k)
    dist_name = "r_inner"
    sampling_interval = 10 * 60
    evaluate_sh_suite(trajectory, sampling_interval, dist_name, hoppers)

    trajectory = SurfaceDist(planet, planet.stl_200k)
    dist_name = "r_surface"
    sampling_interval = 10 * 60
    evaluate_sh_suite(trajectory, sampling_interval, dist_name, hoppers)

    plt.show()


if __name__ == "__main__":
    main()
