
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
#import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Analysis import Analysis
from GravNN.Networks.Callbacks import CustomCallback
from GravNN.Networks.Compression import (cluster_model, prune_model,
                                         quantize_model)
from GravNN.Networks.Model import CustomModel, load_config_and_model
from GravNN.Networks.Networks import (DenseNet, InceptionNet, ResNet,
                                      TraditionalNet)
from GravNN.Networks.Data import generate_dataset, training_validation_split
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

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)


np.random.seed(1234)
tf.random.set_seed(0)

def main():

    df_in_file = 'Data/Dataframes/N_1000000_PINN_study.data'
    df_out_file = 'Data/Dataframes/N_1000000_PINN_study_opt_1E3.data'
    #df_out_file = 'Data/Dataframes/temp_opt.data'

    compression_config = {
        'optimize' : [False],
        'optimize_epochs' : [100],
        'fine_tuning_epochs' : [25000],
        'sparsity' : [0.25], # None, 0.5, 0.8
        'num_w_clusters' : [None], # None, 32, 16
        'quantization' : [None] 
    }

    df = pd.read_pickle(df_in_file)#[-2:]
    ids = df['id'].values

    for model_id in ids:
        tf.keras.backend.clear_session()

        config, model = load_config_and_model(model_id, df_in_file)
        config['mixed_precision'] = [False]
        utils.check_config_combos(config)
        config = utils.format_config_combos(config)
        config.update(compression_config)
        
        # If optimizing, save the original network id as the init_file
        config['init_file'] = [model_id]

        trajectory = config['distribution'][0](config['planet'][0], [config['radius_min'][0], config['radius_max'][0]], config['N_dist'][0], **config)#points=1000000)
        x_unscaled, a_unscaled, u_unscaled = get_sh_data(trajectory, config['grav_file'][0],config['max_deg'][0], config['deg_removed'][0])
    
        
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
        
        # Initial Data
        dataset = generate_dataset(x_train, a_train, config['batch_size'][0])
        val_dataset = generate_dataset(x_val, a_val, config['batch_size'][0])

        network = tf.keras.models.load_model(os.path.abspath('.') +"/Data/Networks/"+str(model_id)+"/network")
        model = CustomModel(config, network)

        optimizer = config['optimizer'][0]
        #optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

        model.compile(optimizer=optimizer, loss="mse")#, run_eagerly=True)#, metrics=["mae"])

        model, cluster_history = cluster_model(model, dataset, val_dataset, config)
        model, prune_history = prune_model(model, dataset, val_dataset, config)
        model, quantize_history = quantize_model(model, dataset, val_dataset, config)
    
        model.optimize(dataset)

        model.save(df_out_file)

        plt.close()



if __name__ == '__main__':
    main()
