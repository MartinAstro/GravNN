import os
import pickle
from random import randint, seed

import matplotlib.pyplot as plt
import numpy as np
import pyshtools
import umap.plot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from GravNN.CelestialBodies.Planets import Earth
from GravNN.Clustering.clustering import DBSCAN_labels, kmeans
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Preprocessors.Projections import *
from GravNN.Preprocessors.Transformations import *
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.Reduced3dDist import Reduced3dDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.Trajectories.RandomDist import RandomDist

from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.StateObject import StateObject
from GravNN.Visualization.VisualizationBase import VisualizationBase

seed(0)


mapUnit = "m/s^2"
mapUnit = 'mGal'
vis = VisualizationBase(save_directory=os.path.splitext(__file__)[0]  + "/../../../Plots/" +os.path.splitext(os.path.basename(__file__))[0] + "/",halt_formatting=True)

planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
density_deg = 175
max_deg = 1000

#trajectory = Reduced3dDist(planet, radiusBounds=[radius, radius*1.05], layers=5, degree=density_deg, reduction=0.25)
#trajectory = ReducedRandDist(planet, radiusBounds=[radius, radius*1.05], points=15488, degree=density_deg, reduction=0.25)

#trajectory = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.25)
#trajectory = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.125)

#trajectory = ReducedGridDist(planet, radius*1.05, degree=density_deg, reduction=0.25)
#trajectory = DHGridDist(planet, radius, degree=density_deg)

point_count = 259200 # 0.5 Deg
trajectory = RandomDist(planet, [radius , radius*1.05], point_count) #R0

def computeStateObject(trajectory, degree):
    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory)

    Call_r0_state_obj = StateObject(gravityModel=Call_r0_gm)
    C20_r0_state_obj = StateObject(gravityModel=C20_r0_gm)
    R0_pert_state_obj = Call_r0_state_obj - C20_r0_state_obj
    return R0_pert_state_obj, Call_r0_gm



if __name__ == "__main__":
    ext = ".png"

    # Get the data 
    R0_pert_state_obj, Call_r0_gm = computeStateObject(trajectory, max_deg)
    
    data = np.hstack((R0_pert_state_obj.positions, R0_pert_state_obj.accelerations))
    #data = Call_r0_gm.compute_acc_components()
    #data = np.hstack((R0_pert_state_obj.positions, data))
    print(data.shape)

    # %% 
    # Project the data into higher or lower dimension
    projection = project_none
    #data = projection(data)

    transformation = minmax_pos_standard_acc_components
    data = transformation(data)

    # %% 
    # UMAP and Plot
    neighbors = np.floor(np.exp([2, 3]))
    min_distances = [0.0, 0.1, 0.5]
    iterations = 1000
    target_metric = 'l2'
    ext = "_deg"+str(max_deg) + ".png"
    for neighbor in neighbors:
        for min_distance in min_distances:
            name = trajectory.trajectory_name + "/" + projection.__name__ + "/" + transformation.__name__ + "/" + str(int(neighbor)) + "_" + str(min_distance).replace(".","_") + "_" + str(iterations) + "_"+ target_metric + ext

            reducer = umap.UMAP(n_neighbors=int(neighbor), min_dist=min_distance,n_epochs=iterations, target_metric=target_metric)
            embedding = reducer.fit_transform(data)
            umap.plot.diagnostic(reducer, diagnostic_type='pca')
            vis.save(plt.gcf(), "UMAP2D/" + name)
            plt.close()

            reducer3D = umap.UMAP(n_neighbors=int(neighbor), min_dist=min_distance, n_components=3, n_epochs=iterations, target_metric=target_metric)
            embedding = reducer3D.fit_transform(data)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], s=0.1)
            vis.save(plt.gcf(), "UMAP3D/" + name)
            plt.close()


    #umap.plot.diagnostic(reducer, diagnostic_type='pca')
    
    plt.show()
