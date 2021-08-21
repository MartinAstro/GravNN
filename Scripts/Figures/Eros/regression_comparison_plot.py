from logging import error
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
from GravNN.Trajectories import DHGridDist, SurfaceDist
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Trajectories.utils import generate_near_orbit_trajectories
from GravNN.Support.transformations import cart2sph
def get_sh_file_info(file_path):
    model_name = os.path.basename(file_path).split('.')[0]
    sampling_interval = int(model_name.split("_")[4])
    max_deg = int(model_name.split("_")[5])
    return sampling_interval, max_deg

def get_nn_file_info(file_path):
    model_name = os.path.basename(file_path).split('.')[0]
    sampling_interval = int(model_name.split("_")[4])
    num_units = int(model_name.split("_")[5])
    return sampling_interval, num_units

def plot_orbits_as_violins():
    trajectories = generate_near_orbit_trajectories(10*60)
    radial_dists = []
    orbit_start_times = []
    t0 = trajectories[0].times[0]
    for trajectory in trajectories:
        x = cart2sph(trajectory.positions)
        r = x[:,0]
        t = trajectory.times
        radial_dists.append(r/1000)
        orbit_start_times.append((t[0]-t0)/86400)
    
    plt.violinplot(radial_dists, positions=orbit_start_times, widths=10)

def plot_regression_error(data_directory, dist_name, sampling_interval, linestyle):
    files = glob.glob(data_directory + "sh_estimate_"+ dist_name + "_" + str(sampling_interval) + "_*.data")
    files.sort()
    for file in files:
        sampling_interval, max_deg = get_sh_file_info(file)
        with open(file, 'rb') as f:
            sh_samples = pickle.load(f)
            sh_errors = pickle.load(f)
        plt.semilogy(sh_samples*sampling_interval/86400, sh_errors, label='SH ' + str(sampling_interval) + ' ' + str(max_deg), linestyle=linestyle)


    files = glob.glob(data_directory+ "nn_estimate_"+ dist_name + "_" + str(sampling_interval) + "*.data")
    files.sort()
    for file in files:
        sampling_interval, num_units = get_nn_file_info(file)
        with open(file, 'rb') as f:
            nn_samples = pickle.load(f)
            nn_errors = pickle.load(f)
        plt.semilogy(nn_samples*sampling_interval/86400, nn_errors, label='PINN ' + str(sampling_interval) + ' ' + str(num_units), linestyle=linestyle)


def main():
    directory = os.path.abspath('.') +"/Plots/Asteroid/Regression/"
    os.makedirs(directory, exist_ok=True)

    planet = Eros()
    trajectory = DHGridDist(planet, planet.radius*2, 90)
    x, a_true, u = get_poly_data(trajectory, planet.model_potatok, point_mass_removed=[False])
    vis = VisualizationBase()
    vis.fig_size = vis.full_page
    data_directory = os.path.abspath('.') + "/GravNN/Files/Regression/"

    
    dist_name = 'r_outer'    
    sampling_interval = 600
    linestyle = '-'
    vis.newFig()
    plot_regression_error(data_directory, dist_name, sampling_interval, linestyle)



    # plt.gca().set_prop_cycle(None)
    # dist_name = 'r_outer'    
    # sampling_interval = 60
    # linestyle = '--'
    # plot_regression_error(data_directory, dist_name, sampling_interval, linestyle)


    plt.xlabel("Days Since Insersion")
    plt.ylabel("Average Acceleration Error")
    plt.legend()
    plt.ylim(1E-1, 1E7)

    plt.twinx()
    plot_orbits_as_violins()
    plt.ylabel("Radius (km)")

    vis.save(plt.gcf(), directory + "regression_error_near_shoemaker.pdf")

    plt.show()

if __name__ == "__main__":
    main()