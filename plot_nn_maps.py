import os
from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization
from GravityModels.SphericalHarmonics import SphericalHarmonics
from CelestialBodies.Planets import Earth
from Trajectories.DHGridDist import DHGridDist
from GravityModels.NN_Base import NN_Base
from keras.utils.layer_utils import count_params

import pyshtools
import matplotlib.pyplot as plt
import numpy as np

from Trajectories.UniformDist import UniformDist
from Trajectories.RandomDist import RandomDist

from Trajectories.DHGridDist import DHGridDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics
from Preprocessors.MinMaxTransform import MinMaxTransform
from Support.transformations import sphere2cart, cart2sph, project_acceleration

from GravityModels.GravityModelBase import GravityModelBase

from GravityModels.NN_Base import NN_Base
from GravityModels.NNSupport.NN_DeepLayers import *
from GravityModels.NNSupport.NN_MultiLayers import *
from GravityModels.NNSupport.NN_SingleLayers import *

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta
import copy
from plot_sh_maps import phase_2
from talos import Restore

def round_stats(grid):
    avg = np.round(np.average(np.abs(grid)),2)
    med =  np.round(np.median(np.abs(grid)),2)
    return avg, med
def print_nn_params(nn, true_grid, grid):
    error_grid = copy.deepcopy(grid)
    error_grid -= true_grid
    error_grid = (error_grid/true_grid) * 100.0
    dr, dr_median = round_stats(error_grid.r)
    dtheta, dtheta_median = round_stats(error_grid.theta)
    dphi, dphi_median = round_stats(error_grid.phi)
    dtotal, dtotal_median = round_stats(error_grid.total)
    
    # if not type(nn.opt) == type(Adadelta()):
    #     lr =  str(nn.lr)
    # else:
    #     lr =  "N/A" 
    # print(nn.model_func.__name__ + "\t&\t" + str(len(nn.x_train)) + "\t&\t" + str(nn.batch_size) + "\t&\t" + str(nn.opt.__class__.__name__) + "\t&\t" +  str(lr) + "\t&\t" + str(dr_median) + "\t&\t" + str(dtheta_median) + "\t&\t" + str(dphi_median) + "\t&\t" + str(dtotal_median) + "\\\\")

    return dtotal_median

def main():
    """
    After running through hyperparameter optimization, load the best model, and plot the corresponding map. 
    """
    # Define training data
    planet = Earth()
    point_count = 1000
    trajectory = UniformDist(planet, planet.radius, point_count)
    #trajectory = RandomDist(planet, [planet.radius, planet.radius*1.1], point_count)

    gravityModel = SphericalHarmonics(planet.sh_file, degree=None, trajectory=trajectory)
    gravityModel.load() 
    gravityModelC20 = SphericalHarmonics(planet.sh_file, degree=2, trajectory=trajectory)
    gravityModelC20.load()

    pos_sphere = cart2sph(trajectory.positions)
    acc_proj = project_acceleration(pos_sphere, gravityModel.accelerations)
    acc_projC20 = project_acceleration(pos_sphere, gravityModelC20.accelerations)
    acc_proj = acc_proj - acc_projC20

    preprocessor = MinMaxTransform()
    preprocessor.percentTest = 0.0
    preprocessor.split(pos_sphere, acc_proj)
    preprocessor.fit()
    x_train, y_train = preprocessor.apply_transform()

    # Initialize test data set (true grid with C20 removed)
    density_deg = 175
    max_deg = 1000
    sh_file = planet.sh_hf_file
    map_grid = DHGridDist(planet, planet.radius*1.00, degree=density_deg)

    sh_all_gravityModel = SphericalHarmonics(sh_file, degree=max_deg, trajectory=map_grid)
    sh_C20_gravityModel = SphericalHarmonics(sh_file, degree=2, trajectory=map_grid)
    true_grid = Grid(gravityModel=sh_all_gravityModel)
    sh_20_grid = Grid(gravityModel=sh_C20_gravityModel)
    true_grid -= sh_20_grid #these values are projected


    # Load the model and its training data
    #r = Restore('third_iteration.zip') # has properties results and details
    r = Restore('uniform_first.zip') 
    
    # Initialize NN
    nn = NN_Base(r.model, preprocessor, test_traj=map_grid)

    # Plot NN Results
    map_viz = MapVisualization()
    grid = Grid(gravityModel=nn, override=True)
    fig, ax = map_viz.percent_maps(true_grid,grid, param="total", vlim=[0,100])
    #map_viz.save(fig, nn.file_directory+"NN_Rel_Error.pdf")
    median_err = print_nn_params(nn, true_grid, grid)
    count_params(nn.model.trainable_weights)
    print(median_err)
    plt.show()

    # x_formatted = np.array(nn.x_train).reshape((len(nn.x_train),3))
    # nn.compute_percent_error(nn.model.predict(x_formatted), nn.y_train) # Not going to work, need to be properly transformed
    # phase_2()
    # plt.show()

if __name__ == "__main__":
    main()
