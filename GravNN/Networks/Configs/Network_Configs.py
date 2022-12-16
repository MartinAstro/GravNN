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
        "loss_fcn" : ['rms_avg'],
        "batch_size": [131072 // 2],
        "learning_rate": [0.001*2],
        "dropout": [0.0],
        "skip_normalization": [False],
        "lr_anneal": ['hold'],
        "beta" : [0.0],
        "input_layer": [False],
        "network_type": ["traditional"],
        "custom_input_layer": [None],
        'seed' : [0],
        'init_file' : [None],
        'normalization_strategy' : ['uniform'], #'radial, uniform
        'jit_compile' : [False],
        'eager' : [False],
        "dtype" : ['float64']
    }
    return network_config


def PINN_II():
    config = PINN_I()
    network_config = {
        "x_transformer": [UniformScaler(feature_range=(-1, 1))],
        "u_transformer": [UniformScaler(feature_range=(-1, 1))],
        "a_transformer": [UniformScaler(feature_range=(-1, 1))],

        "activation": ["gelu"],
        "network_type": ["sph_pines_transformer"],
        "scale_by": ["non_dim"],
    }
    config.update(network_config)
    return config


def PINN_III():
    config = PINN_II()
    network_config = {
        "network_type": ["sph_pines_transformer_v2"],
        "scale_by": ["non_dim_v2"],
        "loss_fcn" : ['avg_percent_summed_rms'],
    }
    config.update(network_config)
    return config