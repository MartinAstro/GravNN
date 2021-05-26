
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
from GravNN.CelestialBodies.Asteroids import Bennu, Eros
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Networks import utils
from GravNN.Networks.Activations import (bent_identity, leaky_relu,
                                         radial_basis_function)
from GravNN.Networks.Analysis import Analysis
from GravNN.Networks.Callbacks import CustomCallback
from GravNN.Networks.Configs.Default_Configs import *
from GravNN.Networks.Configs.Fast_Configs import *
from GravNN.Networks.Constraints import *
from GravNN.Networks.Data import (generate_dataset, get_preprocessed_data,configure_dataset,
                                  training_validation_split, standardize_output)
from GravNN.Networks.Model import CustomModel, load_config_and_model
from GravNN.Networks.Networks import (CustomNet, DenseNet, InceptionNet,
                                      ResNet, TraditionalNet, load_network)
from GravNN.Preprocessors.UniformScaler import UniformScaler
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   StandardScaler)
from tensorflow.keras.callbacks import LearningRateScheduler

#tf.config.run_functions_eagerly(True)
mixed_precision_flag = False

if sys.platform == 'win32':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

if mixed_precision_flag:
    from tensorflow.keras.mixed_precision import \
        experimental as mixed_precision

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

np.random.seed(1234)
tf.random.set_seed(0)

def configure_optimizer(config):
    optimizer = config['optimizer'][0]
    optimizer.learning_rate = config['learning_rate'][0]

    if config['mixed_precision'][0]:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    else:
        optimizer.get_scaled_loss = lambda x: x
        optimizer.get_unscaled_gradients = lambda x: x
    return optimizer

def train_network(filename, config):
    tf.keras.backend.clear_session()

    utils.check_config_combos(config)
    config = utils.format_config_combos(config)
    train_data, val_data, transformers = get_preprocessed_data(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    network = load_network(config)
    model = CustomModel(config, network)
    optimizer = configure_optimizer(config)
    model.compile(optimizer=optimizer, loss="mse", run_eagerly=False)#, run_eagerly=True)#, metrics=["mae"])

    callback = CustomCallback()
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                                        monitor='val_loss', factor=0.9, patience=500, verbose=0,
                                        mode='auto', min_delta=config['min_delta'][0], cooldown=0, min_lr=0, 
                                        )
    early_stop = tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss', min_delta=config['min_delta'][0], patience=3000, verbose=1,
                                        mode='auto', baseline=None, restore_best_weights=True
                                    )

    history = model.fit(dataset, 
                        epochs=config['epochs'][0], 
                        verbose=0,
                        validation_data=val_dataset,
                        callbacks=[callback, early_stop])#,
                        #class_weight=config['class_weight'][0])
    history.history['time_delta'] = callback.time_delta
    model.history = history

    # TODO: Save extra parameters like optimizer.learning_rate
    # Save network and config information
    # y_true = train_data[1:5]
    # y_true = np.hstack(y_true)

    # y_hat = model.output(dataset)
    # y_hat = np.hstack(y_hat)
    # diff = y_hat - y_true 

    # print(np.mean(diff, axis=0))
    # print(np.std(diff, axis=0))

    model.config['time_delta'] = [callback.time_delta]
    model.config['x_transformer'][0] = transformers['x']
    model.config['u_transformer'][0] = transformers['u']
    model.config['a_transformer'][0] = transformers['a']
    model.save(filename)

    plt.close()

    return model, config


def main():
    df_file = 'Data/Dataframes/temp_spherical.data'
    configurations = {"fast" : get_fast_earth_config()}

    df_file = 'Data/Dataframes/new_pinn_constraints.data'
    configurations = {"fast" : get_fast_earth_pinn_config()}

    df_file = "Data/Dataframes/useless_04_19_v4.data"
    configurations = {"default" : get_default_earth_config()}

    df_file = "Data/Dataframes/useless_05_07_v1.data"
    configurations = {"default" : get_default_moon_pinn_config()}
    # Test 1: No PINN without potential
    # Test 2: PINN without potential
    # Test 3: PINN with potential

    for key, config in configurations.items():
        config['network_type'] = [TraditionalNet]
        config['epochs'] = [200000]

        config['N_dist'] = [1000000]
        config['u_transformer'] = [UniformScaler(feature_range=(-1,1))]
        config['a_transformer'] = [UniformScaler(feature_range=(-1,1))]
        config['PINN_constraint_fcn'] = [pinn_A]#pinn_AP]
        config['min_delta'] = [1E-12]
        config['learning_rate'] = [0.005]
        config['scale_by'] = ['a']

        config['layers'] = [[3, 20, 20, 20, 20, 20, 20, 20, 20, 1]]

        config['N_dist'] = [5445]
        config['N_train'] = [5000]
        config['N_val'] = [408]
        config['batch_size'] = [8192]
        config['radius_max'] = [Moon().radius + 50000]
        config['activation'] = ['gelu']
        config['initializer'] = ['glorot_normal']
        #config['loss_scale'] = [[1.0/100.0, 100.0, 100.0, 100.0]]


        config['mixed_precision'] = [mixed_precision_flag]
        config['dtype'] = [tf.float32]
        config['dummy_transformer'] = [DummyScaler()]

        # config['class_weight'] = [{'0' : 1.0/100.0, 
        #                           '1' : 100.0,
        #                           '2' : 100.0,
        #                           '3' : 100.0}]

        config['class_weight'] = [[1.0, 1.0, 1.0]] #PINN_A
        # config['class_weight'] = [[1.0/1000.0, 2000, 2000, 2000]] # PINN AP
        # config['class_weight'] = [[2000, 2000, 2000, 2E6, 2E6, 2E6, 2E6]] #PINN_ALC

        train_network(df_file, config)


if __name__ == '__main__':
    main()
