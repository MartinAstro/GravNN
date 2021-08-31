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


def get_trajectory_and_acceleration(model_file, min_radius, max_radius):
    planet = Eros()
    trajectory = RandomAsteroidDist(planet, [
        min_radius, max_radius], 
        20000, 
        model_file)
    x, a_true, u = get_poly_data(trajectory, model_file, point_mass_removed=[False])
    return trajectory, a_true


def evaluate_nn(min_radius, max_radius, model_file, models):
    trajectory, a_true = get_trajectory_and_acceleration(model_file, min_radius, max_radius)
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



def main():

    # (pinn_A, pinn_ALC)_(None, 5000)
    file_prefix = "pinn_A_None"
    #file_prefix = "pinn_ALC_None"
    # file_prefix = "pinn_A_5000"
    #file_prefix = "pinn_ALC_5000"  
    models = glob.glob("GravNN/Files/GravityModels/Regressed/Eros/EphemerisDist/"+file_prefix+"**.data") # PINN_A, PINN_ALC
    models.sort()

    sampling_interval = 10*60
    planet = Eros()
    min_radius = planet.radius
    max_radius = planet.radius*3
    dist_name = "r_outer_" + str(sampling_interval)
    sample_list, error_list = evaluate_nn(min_radius, max_radius, planet.obj_200k, models)

    plot_path = os.path.abspath('.') +"/Plots/Asteroid/Regression/" + file_prefix + "_"
    generate_figure(plot_path, sample_list, error_list)

    data_directory = os.path.abspath('.') + "/GravNN/Files/Regression/" + file_prefix + "/"
    os.makedirs(data_directory, exist_ok=True)
    with open(data_directory + "nn_estimate_" + dist_name + "_" + ".data", 'wb') as f:
        pickle.dump(sample_list, f)
        pickle.dump(error_list, f)



    # sampling_interval = 1*60
    # evaluate_nn_suite(min_radius, max_radius, sampling_interval, directory, dist_name)




    # min_radius = 0
    # max_radius = planet.radius 
    # dist_name = "r_inner"
    # sampling_interval = 10*60
    # evaluate_nn_suite(min_radius, max_radius, sampling_interval, directory, dist_name)
    # sampling_interval = 1*60
    # evaluate_nn_suite(min_radius, max_radius, sampling_interval, directory, dist_name)

    plt.show()

if __name__ == "__main__":
    main()