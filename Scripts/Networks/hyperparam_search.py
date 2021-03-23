
import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
os.environ["TF_GPU_THREAD_MODE"] ='gpu_private'
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
#import tensorflow_model_optimization as tfmot
from tensorboard.plugins.hparams import api as hp


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
from GravNN.Networks.Model import CustomModel
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
from GravNN.Networks.Activations import bent_identity

#tf.config.run_functions_eagerly(True)
mixed_precision_flag = True

if sys.platform == 'win32':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# # # TODO: Put in mixed precision training
if mixed_precision_flag:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)


#tf.profiler.experimental.server.start(6009)

np.random.seed(1234)
tf.random.set_seed(0)

def load_hparams_to_config(hparams, config):
    for key, value in hparams.items():
        name = key.name
        config[name] = [value]

    if config['activation'][0] == 'bent_identity':
        config['activation'] = [bent_identity]
    
    try:
        if 'adam' in config['optimizer'][0]:
            config['optimizer'][0] = tf.keras.optimizers.Adam()
    except:
        pass

    try: 
        if 'rmsprop' in config['optimizer'][0]:
            config['optimizer'][0] = tf.keras.optimizers.RMSprop()
    except:
        pass
        
    if 'num_units' in config:
        for i in range(1, len(config['layers'][0])-1):
            config['layers'][0][i] = config['num_units'][0]
    
    if config['network_type'][0] == 'traditional':
        config['network_type'] = [TraditionalNet]
    
    if config['network_type'][0] == 'resnet':
        config['network_type'] = [ResNet]
    #config['layers'][0][1:len(config['layers'][0])-1] = config['num_units']
    return config

def main():

    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    # df_file = 'Data/Dataframes/hyperparameter_v1.data'
    # configurations = {"Default" : get_default_earth_config() }

    # HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([20, 40, 80]))
    # HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.1))
    # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'rmsprop']))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'bent_identity'])) 
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([2048, 131072]))
    # HP_DATA_SIZE = hp.HParam('N_train', hp.Discrete([500000, 950000]))
    # HP_EPOCHS = hp.HParam('epochs', hp.Discrete([5000, 10000]))

    # df_file = 'Data/Dataframes/hyperparameter_v2.data'
    # directory = 'logs/hparam_tuning/'

    # df_file = 'Data/Dataframes/useless_board.data'
    # directory = 'logs/useless/'

    
    # df_file = 'Data/Dataframes/hyperparameter_v3.data'
    # directory = 'logs/hparam_tuning_v3/'

    # configurations = {"Default" : get_default_earth_config() }
    # HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([20, 40, 80]))
    # HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.1))
    # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'rmsprop']))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'bent_identity'])) 
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8196, 32768, 131072]))

    # HP_DATA_SIZE = hp.HParam('N_train', hp.Discrete([125000, 250000, 500000]))
    # HP_EPOCHS = hp.HParam('epochs', hp.Discrete([2500, 5000, 7500]))

    # session_num = 0
    # for epochs in HP_EPOCHS.domain.values:
    #     for num_units in HP_NUM_UNITS.domain.values:
    #         for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    #             for optimizer in HP_OPTIMIZER.domain.values:
    #                 for batch_size in HP_BATCH_SIZE.domain.values:
    #                     for data_size in HP_DATA_SIZE.domain.values:
    #                         for activation in HP_ACTIVATION.domain.values:
    #                             hparams = {
    #                                 HP_EPOCHS : epochs,
    #                                 HP_NUM_UNITS: num_units,
    #                                 HP_DROPOUT: dropout_rate,
    #                                 HP_OPTIMIZER: optimizer,
    #                                 HP_BATCH_SIZE: batch_size,
    #                                 HP_DATA_SIZE: data_size,
    #                                 HP_ACTIVATION: activation
    #                             }
    #                             run_name = "run-%d" % session_num
    #                             print('--- Starting trial: %s' % run_name)
    #                             print({h.name: hparams[h] for h in hparams})
    #                             run(df_file, directory + run_name, configurations, hparams)
    #                             session_num += 1
                            

    # df_file = 'Data/Dataframes/hyperparameter_v4.data'
    # directory = 'logs/hparam_tuning_v4/'

    # configurations = {"Default" : get_default_earth_config() }
    # HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([20, 40, 80]))
    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2, 1E-3, 1E-4]))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8196, 32768]))
    # HP_DATA_SIZE = hp.HParam('N_train', hp.Discrete([250000, 500000]))
    # HP_EPOCHS = hp.HParam('epochs', hp.Discrete([1250, 2500]))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu', 'elu', 'relu', 'swish']))

    # session_num = 0
    # for epochs in HP_EPOCHS.domain.values:
    #     for num_units in HP_NUM_UNITS.domain.values:
    #         for learning_rate in HP_LEARNING_RATE.domain.values:
    #             for batch_size in HP_BATCH_SIZE.domain.values:
    #                 for data_size in HP_DATA_SIZE.domain.values:
    #                     for activation in HP_ACTIVATION.domain.values:
    #                         hparams = {
    #                             HP_EPOCHS : epochs,
    #                             HP_NUM_UNITS: num_units,
    #                             HP_LEARNING_RATE: learning_rate,
    #                             HP_BATCH_SIZE: batch_size,
    #                             HP_DATA_SIZE: data_size,
    #                             HP_ACTIVATION: activation
    #                         }
    #                         run_name = "run-%d" % session_num
    #                         print('--- Starting trial: %s' % run_name)
    #                         print({h.name: hparams[h] for h in hparams})
    #                         run(df_file, directory + run_name, configurations, hparams)
    #                         session_num += 1
                            
    df_file = 'Data/Dataframes/hyperparameter_moon_v1.data'
    directory = 'logs/hyperparameter_moon_v1/'

    # configurations = {"Default" : get_default_moon_config() }
    # configurations['Default']['layers'] =  [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
    # configurations['Default']['N_dist'] =  [55000]
    # configurations['Default']['N_train'] =  [50000]
    # configurations['Default']['N_val'] =  [4450]
    # configurations['Default']['radius_max'] = [Moon().radius + 5000]
    # configurations['Default']['epochs'] = [50000]


    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-3, 5E-4, 5E-5]))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([2048, 8196, 32768]))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'gelu', 'relu', 'swish']))
    # HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform', 'glorot_normal']))
    # HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional', 'resnet']))


    # Just Batch Size
    df_file = 'Data/Dataframes/hyperparameter_moon_v2.data'
    directory = 'logs/hyperparameter_moon_v2/'


    configurations = {"Default" : get_default_moon_config() }
    configurations['Default']['layers'] =  [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
    configurations['Default']['N_dist'] =  [55000]
    configurations['Default']['N_train'] =  [50000]
    configurations['Default']['N_val'] =  [4450]
    configurations['Default']['radius_max'] = [Moon().radius + 5000]
    configurations['Default']['epochs'] = [50000]



    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-5]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([512, 1024, 2048]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_normal']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['resnet']))

    
    # More comprehensive Batch Size
    df_file = 'Data/Dataframes/hyperparameter_moon_v3.data'
    directory = 'logs/hyperparameter_moon_v3/'


    configurations = {"Default" : get_default_moon_config() }
    configurations['Default']['layers'] =  [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
    configurations['Default']['N_dist'] =  [55000]
    configurations['Default']['N_train'] =  [50000]
    configurations['Default']['N_val'] =  [4450]
    configurations['Default']['radius_max'] = [Moon().radius + 5000]
    configurations['Default']['epochs'] = [50000]



    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-3, 5E-5]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([512, 1024, 2048]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_normal', 'glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['resnet', 'traditional']))


    # Maybe preprocessing
    # Maybe weight initialization
    session_num = 0
    for learning_rate in HP_LEARNING_RATE.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            for activation in HP_ACTIVATION.domain.values:
                for initializer in HP_INITIALIZER.domain.values:
                    for network in HP_NETWORK.domain.values:
                        hparams = {
                            HP_LEARNING_RATE: learning_rate,
                            HP_BATCH_SIZE: batch_size,
                            HP_ACTIVATION: activation,
                            HP_INITIALIZER : initializer,
                            HP_NETWORK : network
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run(df_file, directory + run_name, configurations, hparams)
                        session_num += 1


def run(df_file, file_name, configurations, hparams):
    
    for key, config_original in configurations.items():
        tf.keras.backend.clear_session()
        config = copy.deepcopy(config_original)

        config = load_hparams_to_config(hparams, config)

        utils.check_config_combos(config)
        config = utils.format_config_combos(config)
        print(config)
        
        
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
        u_transformer = config['u_transformer'][0]

        x_train = x_transformer.fit_transform(x_train)
        a_train = a_transformer.fit_transform(a_train)
        u_train = u_transformer.fit_transform(u_train)

        x_val = x_transformer.transform(x_val)
        a_val = a_transformer.transform(a_val)
        u_val = u_transformer.transform(u_val)

        # Decide to train with potential or not
        y_train = np.hstack([u_train, a_train]) if config['use_potential'][0] else np.hstack([np.zeros(np.shape(u_train)), a_train])
        y_val = np.hstack([u_val, a_val]) if config['use_potential'][0] else np.hstack([np.zeros(np.shape(u_val)), a_val])


        dataset = generate_dataset(x_train, y_train, config['batch_size'][0])
        val_dataset = generate_dataset(x_val, y_val, config['batch_size'][0])


        if config['init_file'][0] is not None:
            network = tf.keras.models.load_model(os.path.abspath('.') +"/Data/Networks/"+str(config['init_file'][0])+"/network")
        else:
            network = config['network_type'][0](**config)        
        
        model = CustomModel(config, network)

        callback = CustomCallback()
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=file_name, histogram_freq=1000, write_graph=True)#, profile_batch='35,50')
        hyper_params = hp.KerasCallback(file_name, hparams)


        optimizer = config['optimizer'][0]
        optimizer.learning_rate = config['learning_rate'][0]
        if config['mixed_precision'][0]:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        else:
            optimizer.get_scaled_loss = lambda x: x
            optimizer.get_unscaled_gradients = lambda x: x
        model.compile(optimizer=optimizer, loss="mse")#, run_eagerly=True)#, metrics=["mae"])

        history = model.fit(dataset, 
                            epochs=config['epochs'][0], 
                            verbose=0,
                            validation_data=val_dataset,
                            callbacks=[callback, tensorboard, hyper_params])# tensorboard, hyper_params])#,
                                        #early_stop])
        history.history['time_delta'] = callback.time_delta
        model.history = history

        # TODO: Save extra parameters like optimizer.learning_rate
        # Save network and config information
        model.config['time_delta'] = [callback.time_delta]
        model.config['x_transformer'][0] = x_transformer
        model.config['a_transformer'][0] = a_transformer
        model.save(df_file)



if __name__ == '__main__':
    main()
