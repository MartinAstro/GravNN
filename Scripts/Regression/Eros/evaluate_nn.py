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
def evaluate_nn(min_radius, max_radius, sampling_interval, directory, dist_name, pinn_type):
    planet = Eros()
    trajectory = RandomAsteroidDist(planet, [
        min_radius, max_radius], 
        2500, 
        planet.model_potatok)

    models = glob.glob(directory + "/PINN_*"+str(sampling_interval)+".data")
    models.sort()

    vis = VisualizationBase()
    x, a_true, u = get_poly_data(trajectory, planet.model_potatok, point_mass_removed=[False])

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
        error_list.append(np.average(a_error))
        sample_list.append(samples)

    sample_list = np.array(sample_list)
    error_list = np.array(error_list)
    idx = np.argsort(sample_list)
    sample_list = np.take_along_axis(sample_list,idx, axis=0)
    error_list = np.take_along_axis(error_list, idx, axis=0)


    plot_directory = os.path.abspath('.') +"/Plots/Asteroid/Regression/" + pinn_type + "/"
    data_directory = os.path.abspath('.') + "/GravNN/Files/Regression/" + pinn_type + "/"
    dist_name += "_"+str(sampling_interval) # ex. r_outer_60

    os.makedirs(plot_directory, exist_ok=True)
    os.makedirs(data_directory, exist_ok=True)

    vis.newFig()
    plt.semilogy(sample_list, error_list)
    plt.xlabel("Samples")
    plt.ylabel("Average Acceleration Error")
    vis.save(plt.gcf(), plot_directory + "nn_error_near_shoemaker"+ dist_name +".pdf")

    with open(data_directory + "nn_estimate_" + dist_name+ "_" + str(config['num_units'][0]) + ".data", 'wb') as f:
        pickle.dump(sample_list, f)
        pickle.dump(error_list, f)


def main():
    pinn_type = 'transformer_pinn_alc' # pinn_a, pinn_alc, transformer_pinn_a, transformer_pinn_alc

    for pinn_type in ["pinn_a", "pinn_alc", "transformer_pinn_a", "transformer_pinn_alc"]:
        directory = "GravNN/Files/GravityModels/Regressed/" + pinn_type + "/Eros/EphemerisDist/"
        os.makedirs(os.path.abspath('.') +"/Plots/Asteroid/Regression/" + pinn_type + "/", exist_ok=True)
        os.makedirs(os.path.abspath('.') + "/GravNN/Files/Regression/" + pinn_type + "/", exist_ok=True)

        planet = Eros()
        min_radius = planet.radius
        max_radius = planet.radius + 10000.0
        dist_name = "r_outer"
        sampling_interval = 10*60
        evaluate_nn(min_radius, max_radius, sampling_interval, directory, dist_name, pinn_type)
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