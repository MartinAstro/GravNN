
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
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Networks import utils
from GravNN.Networks.Analysis import Analysis
from GravNN.Networks.Callbacks import CustomCallback
from GravNN.Networks.Compression import (cluster_model, prune_model,
                                         quantize_model)
from GravNN.Networks.Data import generate_dataset, training_validation_split
from GravNN.Networks.Model import CustomModel
from GravNN.Networks.Networks import (DenseNet, InceptionNet, ResNet,
                                      TraditionalNet)
from GravNN.Networks.Plotting import Plotting
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.ExponentialDist import ExponentialDist
from GravNN.Trajectories.GaussianDist import GaussianDist

from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import cart2sph, sphere2cart, project_acceleration
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler, StandardScaler



physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# TODO: Put in mixed precision training
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


#tf.keras.backend.set_floatx('float16')


np.random.seed(1234)
tf.random.set_seed(0)


def main():
    #df_file = 'Data/Dataframes/temp.data'

    # *Original Analyses
    #df_file = 'Data/Dataframes/N_1000000_study.data'
    #df_file = 'Data/Dataframes/N_1000000_PINN_study.data'
    df_file = 'Data/Dataframes/N_1000000_exp_PINN_study.data'
    #df_file = 'Data/Dataframes/N_1000000_exp_norm_study.data'
    #df_file = 'Data/Dataframes/N_1000000_exp_norm_sph_study.data'


    # *Small Sample Size
    # df_file = 'Data/Dataframes/N_10000_rand_study.data'
    # df_file = 'Data/Dataframes/N_10000_exp_study.data'

    # df_file = 'Data/Dataframes/N_10000_rand_PINN_study.data'
    # df_file = 'Data/Dataframes/N_10000_exp_PINN_study.data'

    # df_file = 'Data/Dataframes/N_10000_rand_spherical_study.data'

    # df_file = 'Data/Dataframes/N_100000_rand_eros_study.data'
    # df_file = 'Data/Dataframes/N_100000_rand_eros_PINN_study.data'

    data_config = {
        'planet' : [Earth()],
        'grav_file' : [Earth().sh_hf_file],
        'distribution' : [ExponentialDist],
        'N_dist' : [1000000],
        'N_train' : [40000], 
        'N_val' : [4000],
        'radius_min' : [Earth().radius],
        'radius_max' : [Earth().radius + 420000.0],
        'acc_noise' : [0.00],
        'basis' : [None],# ['spherical'],
        'deg_removed' : [2],
        'include_U' : [False],
        'max_deg' : [1000], 
        'dtype' : ['float32'],
        'mixedPrecision' :[True],
        'sh_truth' : ['sh_stats_']
    }
    network_config = {
        'network_type' : [TraditionalNet],
        'PINN_flag' : [False],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
        'activation' : ['tanh'],
        'init_file' : [None],#'2459192.4530671295'],
        'epochs' : [100000],
        'optimizer' : [tf.keras.optimizers.Adam()], #(learning_rate=config['lr_scheduler'][0])
        'batch_size' : [160000],
        'dropout' : [0.0], 
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'a_transformer' : [MinMaxScaler(feature_range=(-1,1))]
    }
    
    # ResNet -- 'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
    # DenseNet -- 'layers' : [[3, dense_layer, [10], dense_layer, [10], dense_layer, [10], dense_layer, 3]],
    # InceptionNet -- 'layers' : [[3, inception_layer, inception_layer, inception_layer, inception_layer, 1]],


    config = {}
    config.update(data_config)
    config.update(network_config)

    orbit = 420000.0 # Earth LEO
    #orbit = 1000.0 # Bennu Survey A 

    config['PINN_flag'] = [True]
    config['N_train'] = [950000] 
    config['batch_size'] = [40000]
    config['epochs'] = [100000]
    config['scale_parameter'] = [orbit/3.0]
    config['invert'] = [False]


    config['distribution'] = [ExponentialDist]
    config['scale_parameter'] = [orbit/10.0]
    config_exp_2_1 = copy.deepcopy(config)
    config_exp_2_1.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 1]],
                    'invert' : [False]
                    })

    
    config_exp_2_2 = copy.deepcopy(config)
    config_exp_2_2.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 1]],

                    'invert' : [True]
                    })

    config_exp_2_3 = copy.deepcopy(config)
    config_exp_2_3.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 1]],
                    'invert' : [False]
                    })

    
    config_exp_2_4 = copy.deepcopy(config)
    config_exp_2_4.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 1]],
                    'invert' : [True]
                    })

    config_exp_2_5 = copy.deepcopy(config)
    config_exp_2_5.update({'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 1]],
                    'invert' : [False]
                    })

    
    config_exp_2_6 = copy.deepcopy(config)
    config_exp_2_6.update({'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 1]],
                    'invert' : [True]
                    })
   


    configurations = {

        #  "1_exp2" : config_exp_2_1,
        #  "2_exp2" : config_exp_2_2,
        #  "3_exp2" : config_exp_2_3,
        #  "4_exp2" : config_exp_2_4,
        #  "5_exp2" : config_exp_2_5,
         "6_exp2" : config_exp_2_6,
          }  


    for key, config in configurations.items():
        tf.keras.backend.clear_session()

        utils.check_config_combos(config)
        config = utils.format_config_combos(config)
        
        
        # TODO: Trajectories should take keyword arguments so the inputs dont have to be standard, just pass in config.
        trajectory = config['distribution'][0](config['planet'][0], [config['radius_min'][0], config['radius_max'][0]], config['N_dist'][0], **config)#points=1000000)
        x_unscaled, a_unscaled, u_unscaled = get_sh_data(trajectory, config['grav_file'][0],config['max_deg'][0], config['deg_removed'][0])
        #x_unscaled, a_unscaled, u_unscaled = get_poly_data(trajectory, config['grav_file'][0])


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

        network = config['network_type'][0](config['layers'][0], config['activation'][0], dropout=config['dropout'][0], dtype=config['dtype'][0])
        model = CustomModel(config, network)
        callback = CustomCallback()
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=100, write_graph=True,
                                        write_images=False, update_freq='epoch', profile_batch=2,
                                        embeddings_freq=0, embeddings_metadata=None)


        optimizer = config['optimizer'][0]
        # TODO: Put in mixed precision training
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

        model.compile(optimizer=optimizer, loss="mse")#, run_eagerly=True)#, metrics=["mae"])

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
        model.save(df_file)

        plt.close()



if __name__ == '__main__':
    main()
