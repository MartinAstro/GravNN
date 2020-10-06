from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Clustering.clustering import kmeans, DBSCAN_labels
from GravNN.Preprocessors.Transformations import *
from GravNN.Preprocessors.Projections import * 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import cm


import os
import pyshtools
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pickle
from random import randint, seed
seed(0)


mapUnit = "m/s^2"
mapUnit = 'mGal'
os.path.splitext(os.path.basename(__file__))[0]

map_vis = VisualizationBase(save_directory=os.path.splitext(__file__)[0]  + "/../../../Plots/" +os.path.splitext(os.path.basename(__file__))[0] + "/")

planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
density_deg = 175
max_deg = 1000

trajectory_reduced = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.25)
#trajectory_reduced = DHGridDist(planet, radius, degree=density_deg)

Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_reduced)
Call_r0_grid = Grid(gravityModel=Call_r0_gm)
C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory_reduced)
C20_r0_grid = Grid(gravityModel=C20_r0_gm)
R0_pert_grid = Call_r0_grid - C20_r0_grid


def generate_colors(data_labels):
    N = len(np.unique(data_labels))
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                        name)
                    for name, color in mcolors.XKCD_COLORS.items())
    names = [name for hsv, name in by_hsv]
    interval = int(np.floor(len(by_hsv)/N))
    color_list = []
    for i in range(0,N):
        color_list.append(mcolors.XKCD_COLORS[names[interval*i]])
    
    # Shift data labels to all be positive
    data_labels = data_labels - np.min(data_labels)
    data_colors = []
    for i in range(0, len(data_labels)):
        data_colors.append(color_list[data_labels[i]])
    return data_colors



def plot_3D_surface(x, y, z, axis_labels=None):
    """Plot the acceleration components
    """
    fig, ax = map_vis.new3DFig()

    label_colors = None
    colors_hex = []

    #X, Y = np.meshgrid(x, y)
    #z = np.reshape(z, np.shape(X))
    ax.plot_trisurf(x, y, z, cmap =cm.YlGnBu)
    #ax.plot_surface(X,Y, z)
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])


def plot_3D(x, y, z, axis_labels=None, data_labels=None):
    """Plot the acceleration components
    """
    fig, ax = map_vis.new3DFig()

    label_colors = None
    colors_hex = []
    if data_labels is not None:
        label_colors = generate_colors(data_labels)
    
    ax.scatter(x, y, z, s=1,c=label_colors)
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])


if __name__ == "__main__":
    ext = ".png"

    # Scatter plot of each acceleration plotted over phi and theta
    theta = R0_pert_grid.positions[:,1]
    phi = R0_pert_grid.positions[:,2]

    data = minmax_pos_standard_acc(R0_pert_grid)
    #data = project_36DOF(data)

    labels = None
    labels = kmeans(data, 5)
    #labels = DBSCAN_labels(data, 0.5)

    # %% 
    # Plot the acceleration maps in 3D
    # plot_3D_surface(theta, phi, R0_pert_grid.acceleration[:,0], axis_labels=[r'$\theta$', r'$\phi$', r'$a_{r}$'])
    # plt.gca().view_init(60, 30)
    # map_vis.save(plt.gcf(), '3D_a_r_iso_triangle' + ext)

    # plot_3D_surface(theta, phi, R0_pert_grid.acceleration[:,1], axis_labels=[r'$\theta$', r'$\phi$', r'$a_{\theta}$'])
    # plt.gca().view_init(60, 30)
    # map_vis.save(plt.gcf(), '3D_a_theta_iso_triangle' + ext)

    # plot_3D_surface(theta, phi, R0_pert_grid.acceleration[:,2], axis_labels=[r'$\theta$', r'$\phi$', r'$a_{\phi}$'])
    # plt.gca().view_init(60, 30)
    # map_vis.save(plt.gcf(), '3D_a_phi_iso_triangle' + ext)


    # %% 
    # Plot the acceleration maps in 3D
    ext = "_kmeans_minmax_std_5.png"

    plot_3D(theta, phi, R0_pert_grid.acceleration[:,0], axis_labels=[r'$\theta$', r'$\phi$', r'$a_{r}$'], data_labels=labels)
    plt.gca().view_init(60,30)
    map_vis.save(plt.gcf(), '3D_a_r_iso' + ext)

    plot_3D(theta, phi, R0_pert_grid.acceleration[:,1], axis_labels=[r'$\theta$', r'$\phi$', r'$a_{\theta}$'], data_labels=labels)
    plt.gca().view_init(60,30)
    map_vis.save(plt.gcf(), '3D_a_theta_iso' + ext)

    plot_3D(theta, phi, R0_pert_grid.acceleration[:,2], axis_labels=[r'$\theta$', r'$\phi$', r'$a_{\phi}$'], data_labels=labels)
    plt.gca().view_init(60,30)
    map_vis.save(plt.gcf(), '3D_a_phi_iso' + ext)

    # %% 
    # Plot the acceleration clusters in 3D acceleration space 

    # Scatter plot of the accelerations with no scaling
    # plot_3D(R0_pert_grid.acceleration[:,0], R0_pert_grid.acceleration[:,1], R0_pert_grid.acceleration[:,2], axis_labels=[r'$\theta$', r'$\phi$', r'$a_{r}$'], data_labels=None)
    # map_vis.save(plt.gcf(), '3D_acc_all_unlabeled'+ext)

    # # Scatter plot of the accelerations with labels
    # plot_3D(R0_pert_grid.acceleration[:,0], R0_pert_grid.acceleration[:,1], R0_pert_grid.acceleration[:,2], axis_labels=[r'$\theta$', r'$\phi$', r'$a_{r}$'], data_labels=labels)
    # map_vis.save(plt.gcf(), '3D_acc_all_labeled'+ext)


    #plot_3d_kmeans_3DOF()
    plt.show()

