import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
import pandas as pd
from GravNN.Trajectories import DHGridDist, SurfaceDist, RandomAsteroidDist
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks.Model import CustomModel, load_config_and_model
from GravNN.Visualization.VisualizationBase import VisualizationBase

def evaluate_network_error(model, trajectory, a_true,):
    df = pd.read_pickle(model)
    ids = df['id']
    model_name = os.path.basename(model).split('.')[0]
    samples = int(model_name.split("_")[-1])
    config, model = load_config_and_model(ids[-1], df)
    a = model.generate_acceleration(trajectory.positions.astype(np.float32))
    a_error = np.linalg.norm(a - a_true, axis=1)/np.linalg.norm(a_true, axis=1)*100
    return samples, np.average(a_error)


def evaluate_nn(trajectory, model_file, models):
    planet = Eros()
    x, a_true, u = get_poly_data(trajectory, model_file, point_mass_removed=[False])

    sample_list = []
    error_list = []
    for model in models:
        samples, error = evaluate_network_error(model, trajectory, a_true)
        sample_list.append(samples)
        error_list.append(error)
    sample_list = np.array(sample_list)
    error_list = np.array(error_list)
    idx = np.argsort(sample_list)
    sample_list = np.take_along_axis(sample_list,idx, axis=0)
    error_list = np.take_along_axis(error_list, idx, axis=0)
    return sample_list, error_list



def generate_figure(directory, sample_list, error_list):
    vis = VisualizationBase()
    vis.newFig()
    plt.semilogy(sample_list, error_list)
    plt.xlabel("Samples")
    plt.ylabel("Average Acceleration Error")
    vis.save(plt.gcf(), directory + "nn_error_near_shoemaker.pdf")

def get_constraint_str(prefix):
    words = prefix.split("_")
    function = words[0].lower() + "_" + words[1].lower()
    return function

def main():
    prefix_list = ['pinn_A_None', 'pinn_ALC_None']
    # (pinn_A, pinn_ALC)_(None, 5000)
    for seed in range(0,5):
        for file_prefix in prefix_list: 
            constraint = get_constraint_str(file_prefix)
            model_directory = os.path.curdir + "/GravNN/Files/GravityModels/Regressed/" \
                                                "Eros/EphemerisDist/"+ constraint+ "/" \
                                                "seed_" + str(seed) + "/"
            models = glob.glob( model_directory +file_prefix + "**.data") # PINN_A, PINN_ALC
            models.sort()

            planet = Eros()
            sampling_interval = 10*60
            plot_path = os.path.abspath('.') +"/Plots/Asteroid/Regression/" + file_prefix + "_"

            min_radius = planet.radius
            max_radius = planet.radius*3
            trajectory = RandomAsteroidDist(planet, [
                    min_radius, max_radius], 
                    20000, 
                    planet.obj_200k)
            dist_name = "r_outer_" + str(sampling_interval)
            sample_list, error_list = evaluate_nn(trajectory, planet.obj_200k, models)
            generate_figure(plot_path, sample_list, error_list)
            with open(model_directory + "nn_estimate_" + dist_name + ".data", 'wb') as f:
                pickle.dump(sample_list, f)
                pickle.dump(error_list, f)


            min_radius = 0
            max_radius = planet.radius 
            trajectory = RandomAsteroidDist(planet, [
                    min_radius, max_radius], 
                    20000, 
                    planet.obj_200k)
            dist_name = "r_inner_" + str(sampling_interval)
            sample_list, error_list = evaluate_nn(trajectory, planet.obj_200k, models)
            generate_figure(plot_path, sample_list, error_list)
            with open(model_directory + "nn_estimate_" + dist_name + ".data", 'wb') as f:
                pickle.dump(sample_list, f)
                pickle.dump(error_list, f)


            trajectory = SurfaceDist(planet, planet.obj_200k)
            dist_name = "r_surface_" + str(sampling_interval)
            sample_list, error_list = evaluate_nn(trajectory, planet.obj_200k, models)
            generate_figure(plot_path, sample_list, error_list)
            with open(model_directory + "nn_estimate_" + dist_name + ".data", 'wb') as f:
                pickle.dump(sample_list, f)
                pickle.dump(error_list, f)

    plt.show()

if __name__ == "__main__":
    main()