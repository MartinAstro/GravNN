
import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"

import copy
import pickle
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Analysis import Analysis
from GravNN.Networks.Callbacks import CustomCallback
from GravNN.Networks.Compression import (cluster_model, prune_model,
                                         quantize_model)
from GravNN.Networks.Model import CustomModel, load_config_and_model
from GravNN.Networks.Networks import (DenseNet, InceptionNet, ResNet,
                                      TraditionalNet)
from GravNN.Networks.Plotting import Plotting
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import (cart2sph, project_acceleration,
                                            sphere2cart)
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.FibonacciDist import FibonacciDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.SurfaceDist import SurfaceDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.random.seed(1234)
tf.random.set_seed(0)


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():


    df_file = "Data/Dataframes/temp_spherical.data"

    df = pd.read_pickle(df_file)#[5:]
    ids = df['id'].values

    points = 64800

    planet = Earth()
    alt_list = np.arange(0, 500000, 10000, dtype=float)
    window = np.array([0, 5, 10, 15, 25, 35, 45, 100, 200, 300, 400]) 
    alt_list = np.concatenate([alt_list, window, 420000+window, 420000-window])
    altitudes = np.sort(np.unique(alt_list))
    test_trajectories = {
        "Brillouin" : FibonacciDist(planet, planet.radius, points),
        #"LEO" : FibonacciDist(planet, planet.radius+420000.0, points),
        }
    
    # #* Asteroid config
    # planet = Eros()
    # altitudes = np.arange(0, 10000, 1000, dtype=float) 
    # test_trajectories = {
    #     "Brillouin" : DHGridDist(planet, planet.radius, degree=density_deg),
    #     "Surface" : SurfaceDist(planet, planet.model_25k),
    #     "LBO" : DHGridDist(planet, planet.radius+10000.0, degree=density_deg),
    #     }

    for model_id in ids:
        tf.keras.backend.clear_session()

        alt_df_file = os.path.abspath('.') +"/Data/Networks/"+str(model_id)+"/rse_alt.data"
        config, model = load_config_and_model(model_id, df_file)

        # Analyze the model
        analyzer = Analysis(model, config)
        rse_entries = analyzer.compute_rse_stats(test_trajectories)
        utils.update_df_row(model_id, df_file, rse_entries)
        alt_df = analyzer.compute_alt_stats(planet, altitudes, points)
        alt_df.to_pickle(alt_df_file)


if __name__ == '__main__':
    main()
