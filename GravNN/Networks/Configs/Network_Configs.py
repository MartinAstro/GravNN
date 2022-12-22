from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Trajectories import RandomAsteroidDist
from GravNN.Preprocessors import DummyScaler, UniformScaler
from sklearn.preprocessing import MinMaxScaler


def PINN_I():
    network_config = {
        "basis": [None],
        "mixed_precision": [False],
        "x_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "u_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "a_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "a_bar_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "scale_by": ["a"],
        "dummy_transformer": [DummyScaler()],
        "override" : [False],
        'PINN_constraint_fcn' : ['pinn_a'],
        "layers": [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
        "activation": ["tanh"],
        "epochs": [7500],
        "initializer": ["glorot_normal"],
        "optimizer": ["adam"],
        "batch_size": [131072 // 2],
        "learning_rate": [0.001*2],
        "dropout": [0.0],
        "skip_normalization": [False],
        "lr_anneal": ['hold'],
        "beta" : [0.0],
        "input_layer": [False],
        "network_type": ["custom"],        
        "preprocessing" : [[]],
        'seed' : [0],
        'init_file' : [None],
        'jit_compile' : [False],
        'eager' : [False],
        "dtype" : ['float64'],
        "network_arch" : ['traditional'],
        "loss_fcns" : [['rms']],
    }
    return network_config


def PINN_II():
    config = PINN_I()
    network_config = {
        "x_transformer": [UniformScaler(feature_range=(-1, 1))],
        "u_transformer": [UniformScaler(feature_range=(-1, 1))],
        "a_transformer": [UniformScaler(feature_range=(-1, 1))],

        "activation": ["gelu"],
        "scale_by": ["non_dim"],
        "network_arch" : ['transformer'],
        "preprocessing" : [['pines']]
    }
    config.update(network_config)
    return config


def PINN_III():
    config = PINN_II()
    network_config = {
        "scale_by": ["non_dim_v2"],
        "loss_fcns" : [['rms', 'percent']],
        "preprocessing" : [['pines', 'r_normalize', 'r_inv']]
    }
    config.update(network_config)
    return config

