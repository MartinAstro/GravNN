
import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

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
from GravNN.CelestialBodies.Asteroids import Bennu, Eros
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Networks.Configs.Default_Configs import *
from GravNN.Networks.Configs.Fast_Configs import *
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.GravityModels.SphericalHarmonics import (SphericalHarmonics,
                                                     get_sh_data)
from GravNN.Networks import utils
from GravNN.Networks.Analysis import Analysis
from GravNN.Networks.Callbacks import CustomCallback
from GravNN.Networks.Compression import (cluster_model, prune_model,
                                         quantize_model)
from GravNN.Networks.Data import generate_dataset, training_validation_split
from GravNN.Networks.Model import CustomModel, load_config_and_model
from GravNN.Networks.Networks import (DenseNet, InceptionNet, ResNet,
                                      TraditionalNet)
from GravNN.Networks.Plotting import Plotting
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import (cart2sph, project_acceleration,
                                            sphere2cart)
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ExponentialDist import ExponentialDist
from GravNN.Trajectories.GaussianDist import GaussianDist
from GravNN.Trajectories.RandomAsteroidDist import RandomAsteroidDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from GravNN.Networks.Activations import leaky_relu, bent_identity

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# # TODO: Put in mixed precision training
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


np.random.seed(1234)
tf.random.set_seed(0)


def train_network(filename, config):
    tf.keras.backend.clear_session()

    utils.check_config_combos(config)
    config = utils.format_config_combos(config)
    
    # TODO: Trajectories should take keyword arguments so the inputs dont have to be standard, just pass in config.
    trajectory = config['distribution'][0](config['planet'][0], [config['radius_min'][0], config['radius_max'][0]], config['N_dist'][0], **config)
    if "Planet" in config['planet'][0].__module__:
        get_analytic_data_fcn = get_sh_data
    else:
        get_analytic_data_fcn = get_poly_data
    x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(trajectory, config['grav_file'][0], **config)


    if config['basis'][0] == 'spherical':
        x_unscaled = cart2sph(x_unscaled)     
        a_unscaled = project_acceleration(x_unscaled, a_unscaled)
        x_unscaled[:,1:3] = np.deg2rad(x_unscaled[:,1:3])

    x_train, a_train, u_train, x_val, a_val, u_val = training_validation_split(x_unscaled, 
                                                                                a_unscaled, 
                                                                                u_unscaled, 
                                                                                config['N_train'][0], 
                                                                                config['N_val'][0])

    a_train = a_train + config['acc_noise'][0]*np.std(a_train)*np.random.randn(a_train.shape[0], a_train.shape[1])


    # Preprocessing
    x_transformer = config['x_transformer'][0]
    a_transformer = config['a_transformer'][0]

    x_train = x_transformer.fit_transform(x_train)
    a_train = a_transformer.fit_transform(a_train)

    x_val = x_transformer.transform(x_val)
    a_val = a_transformer.transform(a_val)
    
    # Add Noise if interested

    dataset = generate_dataset(x_train, a_train, config['batch_size'][0])
    val_dataset = generate_dataset(x_val, a_val, config['batch_size'][0])



    if config['init_file'][0] is not None:
        network = tf.keras.models.load_model(os.path.abspath('.') +"/Data/Networks/"+str(config['init_file'][0])+"/network")
    else:
        network = config['network_type'][0](**config)

    model = CustomModel(config, network)

    optimizer = config['optimizer'][0]
    if config['mixed_precision'][0]:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    model.compile(optimizer=optimizer, loss="mse", run_eagerly=False)#, run_eagerly=True)#, metrics=["mae"])

    callback = CustomCallback()
    history = model.fit(dataset, 
                        epochs=config['epochs'][0], 
                        verbose=0,
                        validation_data=val_dataset,
                        callbacks=[callback])#,
                                    #early_stop])
    history.history['time_delta'] = callback.time_delta
    model.history = history

    # TODO: Save extra parameters like optimizer.learning_rate
    # Save network and config information
    model.config['time_delta'] = [callback.time_delta]
    model.config['x_transformer'][0] = x_transformer
    model.config['a_transformer'][0] = a_transformer
    model.save(filename)

    plt.close()

    return model, config


def main():
    df_file = 'Data/Dataframes/temp_spherical.data'
    configurations = {"fast" : get_fast_earth_config()}

    df_file = 'Data/Dataframes/new_pinn_constraints.data'
    configurations = {"fast" : get_fast_earth_pinn_config()}

    df_file = 'Data/Dataframes/new_temp.data'

    df_file = 'Data/Dataframes/new_temp_small.data'
    df_file = 'Data/Dataframes/new_temp_long.data'
    df_file = 'Data/Dataframes/dummy.data'

    configurations = {"default" : get_default_earth_config()}

    df_file = "Data/Dataframes/useless.data"
    df_file = "Data/Dataframes/basis_test.data"

    #config['PINN_flag'] = [False]
    #config['basis'] = ['spherical']
   
    for key, config in configurations.items():
        #config['basis'] = ['spherical']
        #config['init_file'] = [2459255.2569212965]
        #config['PINN_flag'] = ['gradient']

        #config['activation'] = [leaky_relu(act_slope=0.05)]
        #config['act_slope'] = [0.05]
        #config['activation'] = [bent_identity]

        config['epochs'] = [50000]
        config['mixed_precision'] = [True]
        config['layers'] = [True]
        config['dropout'] = [[0, 0.1, 0.05, 0.01, 0, 0, 0, 0]]

        #config['N_train'] = [9500]
        # config['epochs'] = [200]
        # config['N_train'] = [2000]
        # config['N_test'] = [100]

        #config['batch_size'] = [131072]
        config['layers'] = [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
        #config['batch_size'] = [int(config['batch_size'][0]/64)]
        #config['mixed_precision']=[False]
        train_network(df_file, config)


if __name__ == '__main__':
    main()
