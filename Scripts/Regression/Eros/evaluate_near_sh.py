from GravNN.Trajectories import RandomAsteroidDist
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Visualization.VisualizationBase import VisualizationBase

def get_file_info(file_path):
    model_name = os.path.basename(file_path).split('.')[0]
    samples = int(model_name.split("_")[3])
    max_deg = int(model_name.split("_")[1])
    try: 
        time = int(model_name.split("_")[4])
    except:
        time = None
    return samples, max_deg, time

def evaluate_sh(planet, models, trajectory, dist_name, sampling_interval):
    x, a_true, u = get_poly_data(trajectory, planet.obj_200k, point_mass_removed=[False])

    sample_list = np.array([])
    error_list = np.array([])
    time_list = np.array([])

    models.sort()
    for model in models:
        samples, max_deg, time = get_file_info(model)
        x, a, u = get_sh_data(trajectory, model, max_deg=max_deg, deg_removed=-1,override=[True])
        a_error = np.linalg.norm(a - a_true, axis=1)/np.linalg.norm(a_true,axis=1)*100

        sample_list = np.hstack((sample_list, samples))
        error_list = np.hstack((error_list, np.average(a_error)))
        time_list = np.hstack((time_list, time))

    # Sort models based on cumulative sample count
    idx = np.argsort(sample_list)
    sample_list = np.take_along_axis(sample_list,idx, axis=0)
    error_list = np.take_along_axis(error_list, idx, axis=0)

    vis = VisualizationBase()
    vis.fig_size = vis.half_page
    vis.newFig()
    plt.semilogy(sample_list*sampling_interval/(86400), error_list)
    plt.xlabel("Days Since Insersion")
    plt.ylabel("Average Acceleration Error")
    plt.ylim(1E0, np.max(error_list))

    # file_name = "GravNN/Files/GravityModels/Regressed/Eros/true.csv"
    # x, a, u = get_sh_data(trajectory, file_name, max_deg=4, deg_removed=-1, override=[True])
    # a_error = np.linalg.norm(a - a_true)/np.linalg.norm(a_true)*100

    directory = os.path.abspath('.') + "/Plots/Asteroid/Regression/"
    os.makedirs(directory, exist_ok=True)
    vis.save(plt.gcf(), directory + "sh_error_near_shoemaker_"+ str(max_deg) + "_" + dist_name + ".pdf")

    data_directory = os.path.abspath('.') + "/GravNN/Files/GravityModels/Regressed/Eros/EphemerisDist/BLLS/"
    os.makedirs(data_directory,exist_ok=True)
    with open(data_directory + "sh_estimate_" + dist_name + "_" + str(max_deg) + ".data", 'wb') as f:
        pickle.dump(sample_list, f)
        pickle.dump(error_list, f)




def evaluate_sh_suite(min_radius, max_radius, sampling_interval, dist_name):
    directory = "GravNN/Files/GravityModels/Regressed/Eros/EphemerisDist/BLLS/"
    planet = Eros()
    trajectory = RandomAsteroidDist(planet, [
        min_radius, max_radius], 
        20000, 
        planet.obj_200k)

    dist_name += "_"+str(sampling_interval)
    models = glob.glob(directory + "BLLS_4*_0*"+str(sampling_interval)+".csv")
    evaluate_sh(planet, models, trajectory, dist_name, sampling_interval)

    models = glob.glob(directory + "BLLS_8*_0*"+str(sampling_interval)+".csv")
    evaluate_sh(planet, models, trajectory, dist_name, sampling_interval)

    models = glob.glob(directory + "BLLS_16*_0*"+str(sampling_interval)+".csv")
    evaluate_sh(planet, models, trajectory, dist_name, sampling_interval)

    plt.show()


def main():
    planet = Eros()
    min_radius = planet.radius
    max_radius = planet.radius * 3
    dist_name = "r_outer"
    sampling_interval = 10*60
    evaluate_sh_suite(min_radius, max_radius, sampling_interval, dist_name)

    min_radius = 0
    max_radius = planet.radius 
    dist_name = "r_inner"
    sampling_interval = 10*60
    evaluate_sh_suite(min_radius, max_radius, sampling_interval, dist_name)
    # sampling_interval = 1*60
    # evaluate_sh_suite(min_radius, max_radius, sampling_interval, dist_name)



if __name__ == "__main__":
    main()