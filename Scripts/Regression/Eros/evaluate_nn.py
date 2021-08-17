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
def evaluate_nn(planet, models, trajectory, sampling_interval, dist_name):
    directory = os.path.abspath('.') +"/Plots/Asteroid/Regression/"
    os.makedirs(directory, exist_ok=True)
    vis = VisualizationBase()
    x, a_true, u = get_poly_data(trajectory, planet.model_potatok, point_mass_removed=[False])

    models = glob.glob("GravNN/Files/GravityModels/Regressed/Eros/RandomAsteroidDist/*.data")

    sample_list = []
    error_list = []
    for model in models:
        df = pd.read_pickle(model)
        ids = df['id']
        model_name = os.path.basename(model).split('.')[0]
        samples = int(model_name.split("_")[2])
        config, model = load_config_and_model(ids[-1], df)
        a = model.generate_acceleration(trajectory.positions.astype(np.float32))

        a_error = np.linalg.norm(a - a_true, axis=1)/np.linalg.norm(a_true, axis=1)*100

        #a_error = np.sqrt(np.mean(np.square(a - a_true)))
        error_list.append(np.average(a_error))
        sample_list.append(samples)

    sample_list = np.array(sample_list)
    error_list = np.array(error_list)
    idx = np.argsort(sample_list)
    sample_list = np.take_along_axis(sample_list,idx, axis=0)
    error_list = np.take_along_axis(error_list, idx, axis=0)

    vis.newFig()
    plt.semilogy(sample_list, error_list)
    plt.xlabel("Samples")
    plt.ylabel("Average Acceleration Error")
    vis.save(plt.gcf(), directory + "nn_error_near_shoemaker"+ dist_name +".pdf")

    data_directory = os.path.abspath('.') + "/GravNN/Files/Regression/"
    os.makedirs(data_directory,exist_ok=True)
    with open(data_directory + "nn_estimate" + dist_name+ "_" + str(config['num_units'][0]) + ".data", 'wb') as f:
        pickle.dump(sample_list, f)
        pickle.dump(error_list, f)

    plt.show()

def evaluate_nn_suite(min_radius, max_radius, sampling_interval, dist_name):
    directory = "GravNN/Files/GravityModels/Regressed/Eros/"
    planet = Eros()
    trajectory = RandomAsteroidDist(planet, [
        min_radius, max_radius], 
        2500, 
        planet.model_potatok)

    dist_name += "_"+str(sampling_interval)

    models = glob.glob(directory + "EphemerisDist/PINN_20*_*"+str(sampling_interval)+".data")
    evaluate_nn(planet, models, trajectory, sampling_interval, dist_name)

    models = glob.glob(directory + "EphemerisDist/PINN_40*_*"+str(sampling_interval)+".data")
    evaluate_nn(planet, models, trajectory, sampling_interval, dist_name)

    models = glob.glob(directory + "EphemerisDist/PINN_80*_*"+str(sampling_interval)+".data")
    evaluate_nn(planet, models, trajectory, sampling_interval, dist_name)

    plt.show()

def main():
    planet = Eros()
    min_radius = planet.radius
    max_radius = planet.radius + 10000.0
    dist_name = "r_outer"
    sampling_interval = 10*60
    evaluate_nn_suite(min_radius, max_radius, sampling_interval, dist_name)
    # sampling_interval = 1*60
    # evaluate_nn_suite(min_radius, max_radius, sampling_interval, dist_name)

    # min_radius = 0
    # max_radius = planet.radius 
    # dist_name = "r_inner"
    # sampling_interval = 10*60
    # evaluate_nn_suite(min_radius, max_radius, sampling_interval, dist_name)
    # sampling_interval = 1*60
    # evaluate_nn_suite(min_radius, max_radius, sampling_interval, dist_name)


if __name__ == "__main__":
    main()