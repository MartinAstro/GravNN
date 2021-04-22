
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
##import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.CelestialBodies.Asteroids import Bennu
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
from GravNN.Networks.Constraints import no_pinn, pinn_A
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
from GravNN.Preprocessors.UniformScaler import UniformScaler
from GravNN.Preprocessors.DummyScaler import DummyScaler

np.random.seed(1234)
tf.random.set_seed(0)


if sys.platform == 'win32':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def get_default_earth_config():
    data_config = {
        'planet' : [Earth()],
        'grav_file' : [Earth().sh_hf_file],
        'distribution' : [RandomDist],
        'N_dist' : [1000000],
        'N_train' : [950000], 
        'N_val' : [50000],
        'radius_min' : [Earth().radius],
        'radius_max' : [Earth().radius + 420000.0],
        'acc_noise' : [0.00],
        'basis' : [None],# ['spherical'],
        'deg_removed' : [2],
        'include_U' : [False],
        'mixed_precision' : [False],
        'max_deg' : [1000], 
        'analytic_truth' : ['sh_stats_'],
    }
    network_config = {
        'network_type' : [TraditionalNet],
        'PINN_constraint_fcn' : [no_pinn],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
        'activation' : ['tanh'],
        'init_file' : [None],#'2459192.4530671295'],
        'epochs' : [100000],
        'optimizer' : [tf.keras.optimizers.Adam()], #(learning_rate=config['lr_scheduler'][0])
        'batch_size' : [40000],
        'initializer' : ['glorot_normal'],
        'dropout' : [0.0], 
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'a_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'u_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'dtype' : [tf.float32],
        'dummy_transformer' : [DummyScaler()],
        'class_weight' : [[1.0, 1.0, 1.0]], #no_pinn and PINN_A
        'learning_rate' : [0.001]
    }
    config = {}
    config.update(data_config)
    config.update(network_config)
    return config

def get_default_moon_config():
    data_config = {
        'planet' : [Moon()],
        'grav_file' : [Moon().sh_hf_file],
        'distribution' : [RandomDist],
        'N_dist' : [1000000],
        'N_train' : [950000], 
        'N_val' : [50000],
        'radius_min' : [Moon().radius],
        'radius_max' : [Moon().radius + 50000.0],
        'acc_noise' : [0.00],
        'basis' : [None],# ['spherical'],
        'deg_removed' : [2],
        'mixed_precision' : [False],
        'max_deg' : [1000], 
        'analytic_truth' : ['sh_stats_moon_'],
    }
    network_config = {
        'network_type' : [TraditionalNet],
        'PINN_constraint_fcn' : [no_pinn],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
        'activation' : ['tanh'],
        'init_file' : [None],#'2459192.4530671295'],
        'epochs' : [100000],
        'initializer' : ['glorot_normal'],
        'optimizer' : [tf.keras.optimizers.Adam()], #(learning_rate=config['lr_scheduler'][0])
        'batch_size' : [40000],
        'dropout' : [0.0], 
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'a_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'u_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'dtype' : [tf.float32],
        'dummy_transformer' : [DummyScaler()],
        'class_weight' : [[1.0, 1.0, 1.0]] #no_pinn and PINN_A
    }
    config = {}
    config.update(data_config)
    config.update(network_config)
    return config

def get_default_eros_config():
    data_config = {
        'planet' : [Eros()],
        'grav_file' : [Eros().model_25k],
        'distribution' : [RandomAsteroidDist],
        'N_dist' : [100000],
        'N_train' : [95000], 
        'N_val' : [5000],
        'radius_min' : [0],
        'radius_max' : [Eros().radius + 10000.0],
        'acc_noise' : [0.00],
        'basis' : [None],
        'deg_removed' : [0],
        'mixed_precision' : [False],
        'dtype' :['float32'],
        'max_deg' : [1000], 
        'analytic_truth' : ['poly_stats_']
    }
    network_config = {
        'network_type' : [TraditionalNet],
        'PINN_constraint_fcn' : [no_pinn],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
        'activation' : ['tanh'],
        'init_file' : [None],
        'epochs' : [100000],
        'initializer' : ['glorot_normal'],
        'optimizer' : [tf.keras.optimizers.Adam()],
        'batch_size' : [40000],
        'dropout' : [0.0], 
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'a_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'dtype' : [tf.float32],
        'dummy_transformer' : [DummyScaler()],
        'class_weight' : [[1.0, 1.0, 1.0]] #no_pinn and PINN_A
    }

    config = {}
    config.update(data_config)
    config.update(network_config)
    return config

def get_default_earth_pinn_config():
    data_config = {
        'planet' : [Earth()],
        'grav_file' : [Earth().sh_hf_file],
        'distribution' : [RandomDist],
        'N_dist' : [1000000],
        'N_train' : [95000], 
        'N_val' : [50000],
        'radius_min' : [Earth().radius],
        'radius_max' : [Earth().radius + 420000.0],
        'initializer' : ['glorot_normal'],
        'acc_noise' : [0.00],
        'basis' : [None],# ['spherical'],
        'deg_removed' : [2],
        'mixed_precision' : [False],
        'max_deg' : [1000], 
        'analytic_truth' : ['sh_stats_']
    }
    network_config = {
        'network_type' : [TraditionalNet],
        'PINN_constraint_fcn' : [pinn_A],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 1]],
        'activation' : ['tanh'],
        'init_file' : [None],#'2459192.4530671295'],
        'epochs' : [100000],
        'optimizer' : [tf.keras.optimizers.Adam()], #(learning_rate=config['lr_scheduler'][0])
        'batch_size' : [40000],
        'dropout' : [0.0], 
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'u_transformer' : [UniformScaler(feature_range=(-1,1))],
        'a_transformer' : [UniformScaler(feature_range=(-1,1))],
        'dtype' : [tf.float32],
        'dummy_transformer' : [DummyScaler()],
        'class_weight' : [[1.0, 1.0, 1.0]], #no_pinn and PINN_A
        'learning_rate' : [0.001]
    }
    config = {}
    config.update(data_config)
    config.update(network_config)
    return config

def get_default_eros_pinn_config():
    data_config = {
        'planet' : [Eros()],
        'grav_file' : [Eros().model_25k],
        'distribution' : [RandomAsteroidDist],
        'N_dist' : [100000],
        'N_train' : [95000], 
        'N_val' : [5000],
        'radius_min' : [0],
        'radius_max' : [Eros().radius + 10000.0],
        'acc_noise' : [0.00],
        'basis' : [None],
        'deg_removed' : [0],
        'mixed_precision' : [False],
        'dtype' :['float32'],
        'max_deg' : [1000], 
        'analytic_truth' : ['poly_stats_']
    }
    network_config = {
        'network_type' : [TraditionalNet],
        'PINN_constraint_fcn' : [pinn_A],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 1]],
        'activation' : ['tanh'],
        'init_file' : [None],
        'epochs' : [100000],
        'optimizer' : [tf.keras.optimizers.Adam()],
        'batch_size' : [40000],
        'dropout' : [0.0], 
        'u_transformer' : [UniformScaler(feature_range=(-1,1))],
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'a_transformer' : [UniformScaler(feature_range=(-1,1))],
        'dtype' : [tf.float32],
        'dummy_transformer' : [DummyScaler()],
        'class_weight' : [[1.0, 1.0, 1.0]] #no_pinn and PINN_A
    }

    config = {}
    config.update(data_config)
    config.update(network_config)
    return config