import os
import pickle
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talos
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.NN_Base import NN_Base
from GravNN.GravityModels.NNSupport.NN_hyperparam import NN_hyperparam
from GravNN.GravityModels.NNSupport.SupportFunc import *
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Preprocessors.MaxAbsTransform import MaxAbsTransform
from GravNN.Preprocessors.MinMaxStandardTransform import \
    MinMaxStandardTransform
from GravNN.Preprocessors.MinMaxTransform import MinMaxTransform
from GravNN.Preprocessors.RobustTransform import RobustTransform
from GravNN.Preprocessors.StandardTransform import StandardTransform
from GravNN.Support.transformations import (cart2sph,
                                            check_fix_radial_precision_errors,
                                            project_acceleration, sphere2cart)
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from keras.utils.layer_utils import count_params
from numpy.random import seed
from sklearn.model_selection import train_test_split
from talos import Analyze, Deploy, Evaluate, Predict, Reporting, Restore, Scan
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

seed(1)

def main():
    """
    Run an arbitrary NN configuration and 
    """
    name = "20_20_2_0.001_300"
    name = "13_0_0.001_300"
    name = "141_0_0.001_300" # BS 1
    name = "5_5_2_0.001_300" #BS 1
    name = "68_68_2_0.001_10000_BS100" #BS 1



    planet = Earth()
    preprocessor = MinMaxStandardTransform()
    point_count = 259200 
    #point_count = 180*360

    # trajectory = UniformDist(planet, planet.radius, point_count)
    trajectory = RandomDist(planet, [planet.radius, planet.radius+5000.0], point_count)
    #trajectory = RandomDist(planet, [planet.radius+330.0*1000-2500 , planet.radius + 330.0*1000+2500], point_count) #LEO
    plot_maps = True

    experiment_dir = generate_experiment_dir(trajectory, preprocessor)
    save_location = "./Files/Test_NN/" + experiment_dir + "/" + name + "/"

    params = {}
    params['kernel_initializer'] = 'glorot_normal'
    params['kernel_regularizer'] = None # 'l2'
    params['first_unit'] = 68#5
    params['first_neuron'] = 68 # 5 #128
    params['hidden_layers'] = 2
    params['dropout'] = 0.0
    params['batch_size'] = 100
    params['lr'] = 0.001
    params['epochs'] = 10000
    params['optimizer'] = Adadelta
    params['shapes'] = 'brick'
    params['losses'] = 'mean_squared_error'# 'mean_absolute_error'
    params['activation'] = 'relu' #LeakyReLU()# 'relu' #

    sh_file = planet.sh_hf_file
    max_deg = 1000

    gravityModelMap = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    gravityModelMap.load() 
    gravityModelMapC20 = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
    gravityModelMapC20.load() 
    gravityModelMap.accelerations -= gravityModelMapC20.accelerations

    pos_sphere = cart2sph(trajectory.positions)
    pos_sphere = check_fix_radial_precision_errors(pos_sphere)
    acc_proj = project_acceleration(pos_sphere, gravityModelMap.accelerations)
    
    preprocessor.percentTest = 0.3
    preprocessor.split(pos_sphere, acc_proj)
    preprocessor.fit()
    x_train, x_val, y_train, y_val = preprocessor.apply_transform()

    hist, model = NN_hyperparam(x_train, y_train, x_val, y_val, params, verbose=1, save_location=save_location, lr_norm=False)

    plot_metrics(hist)
    compute_error(model, 
                                x_train, y_train,
                                x_val, y_val
                                , preprocessor)

    # Plot NN Results
    if plot_maps:
        map_grid = DHGridDist(planet, planet.radius, degree=175)
        sh_all_gravityModel = SphericalHarmonics(sh_file, degree=max_deg, trajectory=map_grid)
        sh_C20_gravityModel = SphericalHarmonics(sh_file, degree=2, trajectory=map_grid)
        true_grid = Grid(trajectory=map_grid, accelerations=sh_all_gravityModel.load()
        sh_20_grid = Grid(trajectory=map_grid, accelerations=sh_C20_gravityModel.load()
        true_grid -= sh_20_grid #these values are projected
        nn = NN_Base(model, preprocessor, test_traj=map_grid)

        gravityModelMap = SphericalHarmonics(sh_file, degree=100, trajectory=map_grid)
        gravityModelMap.load() 
        C100_grid = Grid(trajectory=map_grid, accelerations=gravityModelMap.load()
        C100_grid -= sh_20_grid

        map_viz = MapVisualization(unit = 'mGal')
        grid = Grid(trajectory=map_grid, accelerations=nn.load(), override=True)
        fig, ax = map_viz.plot_grid_rmse(grid, true_grid,vlim=[0, 40])

        # std = np.std(true_grid.total)
        # mask = true_grid.total > 3*std
        M_params = count_params(nn.model.trainable_weights)
        print("Params: " + str(M_params))

        #map_viz.save(fig, nn.file_directory+"NN_Rel_Error.pdf")
        # coefficient_list.append(M_params)
        # rmse_list.append(np.average(np.sqrt(np.square(grid.total - true_grid.total))))
        # rmse_feat_list.append(np.average(np.sqrt(np.square((grid.total - true_grid.total))),weights=mask))
        plt.show()



if __name__ == '__main__':
    main()
