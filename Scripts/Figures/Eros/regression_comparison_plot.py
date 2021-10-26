from logging import error
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
import numpy as np
import glob
import os
import pickle
from GravNN.Trajectories import DHGridDist, SurfaceDist
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Trajectories.utils import generate_near_orbit_trajectories, generate_near_hopper_trajectories
from GravNN.Support.transformations import cart2sph

def get_sh_file_info(file_path):
    model_name = os.path.basename(file_path).split('.')[0]
    directories = os.path.dirname(file_path).split('/')
    sampling_interval = int(model_name.split("_")[4])
    max_deg_dir = directories[-4]
    max_deg = int(max_deg_dir.split("_")[1])
    return sampling_interval, max_deg

def get_nn_file_info(file_path):
    model_name = os.path.basename(file_path).split('.')[0]
    sampling_interval = int(model_name.split("_")[-1])
    return sampling_interval

def convert_type_to_latex(name, network_type):
    words = name.split("_")
    if 'Transformer' in network_type:
        output =  "PIT" + " " + words[1].upper() 
    else:
        output = words[0].upper() + " " + words[1].upper() 
    return output


def compute_confidence_interval_and_average(y):
    y = np.array(y)
    avg_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0)
    # 95% confidence interval
    ci = 1.96 * std_y/avg_y
    return ci, avg_y

def plot_orbits_as_violins(trajectories, near_trajectories, color='black'):
    radial_dists = []
    orbit_start_times = []
    t0 = near_trajectories[0].times[0]
    for i in range(0,len(trajectories)):
        x = cart2sph(trajectories[i].positions)
        r = x[:,0]
        t = near_trajectories[i].times
        radial_dists.append(r/1000)
        orbit_start_times.append((t[0]-t0)/86400)
    
    bodies = plt.violinplot(radial_dists, positions=orbit_start_times, widths=10)
    for pc in bodies['bodies']:
        pc.set_color(color)
    bodies['cmaxes'].set_color(color)
    bodies['cmins'].set_color(color)
    bodies['cbars'].set_color(color)
    # bodies.set_color('black')

def plot_sh_error(data_directory, dist_name, sampling_interval, linestyle):
    files = glob.glob(data_directory + "Data/sh_estimate_"+ dist_name + "*.data")
    files.sort()
    for file in files:
        sampling_interval, max_deg = get_sh_file_info(file)
        with open(file, 'rb') as f:
            sh_samples = pickle.load(f)
            sh_errors = pickle.load(f)
        plt.semilogy(sh_samples*sampling_interval/86400, sh_errors, label='SH ' + str(max_deg), linestyle=linestyle)

def plot_nn_error(network_type, pinn_type, hoppers, dist_name, sampling_interval, linestyle):
    data_directory = os.path.abspath('.') + \
        "/GravNN/Files/GravityModels/Regressed/Eros/EphemerisDist/" \
            "%s/%s/%s/**/Data/nn_estimate_%s_%d*.data" % \
        (network_type, pinn_type, str(hoppers) ,dist_name, sampling_interval)
    files = glob.glob(data_directory)
    nn_samples = []
    nn_errors = []
    files.sort()
    for file in files:
        sampling_interval = get_nn_file_info(file)
        with open(file, 'rb') as f:
            nn_samples.append(pickle.load(f))
            nn_errors.append(pickle.load(f))
    
    ci, avg_y = compute_confidence_interval_and_average(nn_errors)
    plt.semilogy(nn_samples[0]*sampling_interval/86400, avg_y, label=convert_type_to_latex(pinn_type, network_type), linestyle=linestyle)
    plt.gca().fill_between(nn_samples[0]*sampling_interval/86400, (avg_y-ci), (avg_y+ci), alpha=.1)


def plot_error(dist_name, hoppers, linestyle):
    sampling_interval = 600
    sh_directory = os.path.abspath('.') + "/GravNN/Files/GravityModels/Regressed/Eros/EphemerisDist/BLLS/**/**/%s/" % str(hoppers)
    plot_sh_error(sh_directory, dist_name, sampling_interval, linestyle)

    network_type = "SphericalPinesTransformerNet"
    # pinn_type = 'pinn_a'
    # plot_nn_error(network_type, pinn_type, hoppers, dist_name, sampling_interval, linestyle)

    pinn_type = 'pinn_alc'
    plot_nn_error(network_type, pinn_type, hoppers,  dist_name, sampling_interval, linestyle)
    plt.gca().set_prop_cycle(None)


def main():
    directory = os.path.abspath('.') +"/Plots/Asteroid/Regression/"
    os.makedirs(directory, exist_ok=True)


    vis = VisualizationBase()
    vis.fig_size = vis.full_page
    vis.newFig()

    hoppers=False

    plot_error("r_outer", hoppers, '-')
    plot_error("r_inner", hoppers, '--')
    plot_error("r_surface", hoppers, ':')
    
    plt.xlabel("Days Since Insertion")
    plt.ylabel("Average Acceleration Error")
    plt.ylim(1E-1, 1E3)

    lines = plt.gca().get_lines()
    # TODO: change index if you include pinn_a
    legend1 = plt.legend(handles=lines[0:4],loc='upper left')

    exterior_line = mlines.Line2D([], [], color='black', marker='',
                            markersize=15, linestyle='-', label='Exterior')
    interior_line = mlines.Line2D([], [], color='black', marker='',
                            markersize=15, linestyle='--', label='Interior')
    surface_line = mlines.Line2D([], [], color='black', marker='',
                            markersize=15, linestyle=':', label='Surface')
    plt.legend(handles=[exterior_line, interior_line, surface_line], loc='upper right')
    plt.gca().add_artist(legend1)


    plt.twinx()
    near_trajectories = generate_near_orbit_trajectories(60*10)
    hopper_trajectories = generate_near_hopper_trajectories(60*10)
    plot_orbits_as_violins(near_trajectories, near_trajectories, color='black')
    
    # Add rectangle patch which shows the min and max radii of the asteroid
    poly_gm = Polyhedral(Eros(), Eros().obj_200k)
    min_radius = np.min(np.linalg.norm(poly_gm.mesh.vertices, axis=1))
    max_radius = np.max(np.linalg.norm(poly_gm.mesh.vertices, axis=1))
    rect = Rectangle(xy=(0,min_radius), height=max_radius - min_radius, width=350, alpha=0.3, color='skyblue')
    plt.gca().add_patch(rect)

    if hoppers: 
        plot_orbits_as_violins(hopper_trajectories, near_trajectories, color='magenta')
    else:
        plt.gca().annotate("Permissible Altitudes Below Brillouin Radius", xy=(0.5,0.5), xytext=(0.25,0.25), xycoords=rect, textcoords=rect, color='dodgerblue', fontsize=6)

    plt.ylabel("Radius (km)")



    vis.save(plt.gcf(), directory + "transformer_regression_error_near_shoemaker_%s.pdf" % str(hoppers))
    # vis.save(plt.gcf(), directory + "regression_error_near_shoemaker.pdf")
    
    plt.show()


if __name__ == "__main__":
    main()