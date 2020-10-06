from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Preprocessors.Transformations import *
from GravNN.Preprocessors.Projections import * 
from GravNN.Clustering.clustering import kmeans, DBSCAN_labels, optics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits.axes_grid1 import make_axes_locatable


import os
import pyshtools
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import randint, seed
import umap.plot
seed(0)


mapUnit = "m/s^2"
mapUnit = 'mGal'
map_vis = VisualizationBase(save_directory=os.path.splitext(__file__)[0]  + "/../../../Plots/" +os.path.splitext(os.path.basename(__file__))[0] + "/",halt_formatting=True)

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

def plot_2D_scatter(x, y, labels=None):
    fig, ax = map_vis.newFig()
    plt.scatter(x, y, c=labels, s=1)
    ax.set_aspect('equal', 'datalim')

def plot_2D(x,y, z, labels):
    """Plot an image with colorbar
    """
    fig, ax = map_vis.newFig()
    x_unique = np.unique(x.round(decimals=4))
    y_unique = np.unique(y.round(decimals=4))
    Z = np.reshape(z, (len(x_unique), len(y_unique)))
    plt.imshow(np.transpose(Z))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(label=labels[2], cax=cax)


if __name__ == "__main__":
    ext = ".png"

    # Scatter plot of each acceleration plotted over phi and theta
    theta = R0_pert_grid.positions[:,1]
    phi = R0_pert_grid.positions[:,2]

    # %% 
    # Scale the data

    transform = minmax_pos_standard_acc_latlon
    data = transform(R0_pert_grid)
    #data = minmax_all(R0_pert_grid)

    # %% 
    # Project the data into higher or lower dimension

    #data = project_36DOF(data)
    
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.0, n_epochs=1000, target_metric="l2")
    data = reducer.fit_transform(data)

    # %% 
    # Label the data 
    clusters = 10
    #labels = kmeans(data, clusters)
    labels = DBSCAN_labels(data, 0.1)
    #labels = None
    #labels=optics(data, 10)
  
    ext = "_42DOF" + "_kmeans_" + str(clusters) + "_"+transform.__name__ + ".png"
    plot_2D(theta, phi, labels, labels=[r'$\theta$', r'$\phi$', r'Classes'])
    #map_vis.save(plt.gcf(), '2D_a_classes' + ext)

    plot_2D_scatter(data[:,0], data[:,1], labels=labels)


    # %% 
    # plot_2D(theta, phi, R0_pert_grid.acceleration[:,0], labels=[r'$\theta$', r'$\phi$', r'$a_{r}$'])
    # map_vis.save(plt.gcf(), '2D_a_r_top' + ext)
    
    # plot_2D(theta, phi, R0_pert_grid.acceleration[:,1], labels=[r'$\theta$', r'$\phi$', r'$a_{\theta}$'])
    # map_vis.save(plt.gcf(), '2D_a_theta_top' + ext)

    # plot_2D(theta, phi, R0_pert_grid.acceleration[:,2], labels=[r'$\theta$', r'$\phi$', r'$a_{\phi}$'])
    # map_vis.save(plt.gcf(), '2D_a_phi_top' + ext)



    plt.show()

