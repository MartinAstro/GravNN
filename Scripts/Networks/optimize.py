
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
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Analysis import Analysis
from GravNN.Networks.Callbacks import CustomCallback
from GravNN.Networks.Compression import (cluster_model, prune_model,
                                         quantize_model)
from GravNN.Networks.Model import CustomModel
from GravNN.Networks.Networks import (DenseNet, InceptionNet, ResNet,
                                      TraditionalNet)
from GravNN.Networks.Plotting import Plotting
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import cart2sph, sphere2cart, project_acceleration
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.random.seed(1234)
tf.random.set_seed(0)

def main():
    planet = Earth()
    gravity_file = planet.sh_hf_file
    density_deg = 180
    save = True
    train = True
    df_file = 'temp.data'
    df_file = 'param_study.data'
    #df_file = "tensorflow_2_results.data"
    #df_file = "tensorflow_2_ablation.data"
    
    inception_layer = [3, 7, 11]
    dense_layer = [10, 10, 10]
    data_config = {
        'distribution' : [RandomDist],
        'N_train' : [250000], 
        'N_val' : [4000],
        'radius_min' : [planet.radius],
        'radius_max' : [planet.radius + 10.0],
        'acc_noise' : [0.00],
        'basis' : [None],# ['spherical'],
        'deg_removed' : [2],
        'include_U' : [False],
        'max_deg' : [1000], 
        'sh_truth' : ['sh_stats_']
    }
    compression_config = {
        'fine_tuning_epochs' : [25000],
        'sparsity' : [None], # None, 0.5, 0.8
        'num_w_clusters' : [None], # None, 32, 16
        'quantization' : [None] 
    }

    # ResNet -- 'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
    # DenseNet -- 'layers' : [[3, dense_layer, [10], dense_layer, [10], dense_layer, [10], dense_layer, 3]],
    # InceptionNet -- 'layers' : [[3, inception_layer, inception_layer, inception_layer, inception_layer, 1]],


    config = {}
    config.update(data_config)
    config.update(network_config)
    config.update(compression_config)

    # Dropout
    configurations = {
         "2" : config,
    }    

    for key, config in configurations.items():
        tf.keras.backend.clear_session()

        utils.check_config_combos(config)
        
        trajectory = config['distribution'][0](planet, [config['radius_min'][0], config['radius_max'][0]], points=1000000)
        x_unscaled, a_unscaled, u_unscaled = get_sh_data(trajectory, planet, gravity_file,config['max_deg'][0], config['deg_removed'][0])
        x_train, a_train, u_train, x_val, a_val, u_val = training_validation_split(x_unscaled, 
                                                                                    a_unscaled, 
                                                                                    u_unscaled, 
                                                                                    config['N_train'][0], 
                                                                                    config['N_val'][0])

        if config['basis'][0] == 'spherical':
            x_unscaled = cart2sph(x_unscaled)     
            a_unscaled = project_acceleration(x_unscaled, a_unscaled)
            x_unscaled[:,1:3] = np.deg2rad(x_unscaled[:,1:3])

        # Preprocessing
        try:
            x_transformer = config['preprocessing'][0](feature_range=(-1,1))
            a_transformer = config['preprocessing'][0](feature_range=(-1,1))
        except:
            x_transformer = config['preprocessing'][0]()
            a_transformer = config['preprocessing'][0]()

        x_train = x_transformer.fit_transform(x_train)
        a_train = a_transformer.fit_transform(a_train)

        x_val = x_transformer.transform(x_val)
        a_val = a_transformer.transform(a_val)
        
        # Initial Data

        # Add Noise if interested
        a_train = a_train + config['acc_noise'][0]*np.std(a_train)*np.random.randn(a_train.shape[0], a_train.shape[1])

        dataset = generate_dataset(x_train, a_train)
        val_dataset = generate_dataset(x_val, a_val)

        network = tf.keras.models.load_model(os.path.abspath('.') +"/Plots/"+str(init_file)+"/network")
        model = CustomModel(config, network)

        try:
            optimizer = config['optimizer'][0](learning_rate=config['lr_scheduler'][0])
        except:
            print("EXCEPTION INITIALIZING OPTIMIZER")
            optimizer = config['optimizer'][0]()

        model.compile(optimizer=optimizer, loss="mse")#, run_eagerly=True)#, metrics=["mae"])

        model, cluster_history = cluster_model(model, dataset, config)
        model, prune_history = prune_model(model, dataset, config)
        model, quantize_history = quantize_model(model, dataset, config)
    
        model.optimize(dataset)

        model.save()

        plt.close()



if __name__ == '__main__':
    main()
