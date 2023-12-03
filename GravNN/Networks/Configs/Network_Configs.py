from sklearn.preprocessing import MinMaxScaler

from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Preprocessors.UniformScaler import UniformScaler


def NN():
    network_config = {
        "basis": [None],
        "mixed_precision": [False],
        "x_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "u_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "a_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "a_bar_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "scale_by": ["a"],
        "dummy_transformer": [DummyScaler()],
        "override": [False],
        "PINN_constraint_fcn": ["pinn_00"],
        "layers": [[3, 20, 20, 20, 20, 20, 20, 20, 20, 1]],
        "activation": ["tanh"],
        "epochs": [7500],
        "initializer": ["glorot_normal"],
        "optimizer": ["adam"],
        "batch_size": [131072 // 2],
        "learning_rate": [0.005],
        "dropout": [0.0],
        "skip_normalization": [False],
        "lr_anneal": [False],
        "beta": [0.0],
        "input_layer": [False],
        "network_type": ["basic"],
        "preprocessing": [[]],
        "seed": [0],
        "init_file": [None],
        "jit_compile": [True],
        "eager": [False],
        "dtype": ["float32"],
        "network_arch": ["traditional"],
        "loss_fcns": [["rms"]],
        "trainable_tanh": [False],
        "scale_nn_potential": [False],
        "fuse_models": [False],
        "enforce_bc": [False],
    }
    return network_config


def PINN_I():
    config = NN()
    network_config = {
        "PINN_constraint_fcn": ["pinn_a"],
    }
    config.update(network_config)
    return config


def PINN_II():
    config = PINN_I()
    network_config = {
        "x_transformer": [UniformScaler(feature_range=(-1, 1))],
        "u_transformer": [UniformScaler(feature_range=(-1, 1))],
        "a_transformer": [UniformScaler(feature_range=(-1, 1))],
        "network_type": ["custom"],
        "activation": ["gelu"],
        "scale_by": ["non_dim"],
        "network_arch": ["transformer"],
        "preprocessing": [["pines"]],
        "final_layer_initializer": ["glorot_uniform"],
    }
    config.update(network_config)
    return config


def PINN_III():
    config = PINN_II()
    network_config = {
        "scale_by": ["non_dim_v3"],
        "loss_fcns": [["rms", "percent"]],
        "preprocessing": [["pines", "r_inv"]],
        "freq_decay": [False],
        "lr_anneal": [False],
        "fourier_scale": [1.0],
        "final_layer_initializer": ["zeros"],
        "scale_nn_potential": [True],
        "fuse_models": [True],
        "enforce_bc": [True],
        "trainable_tanh": [True],
        "tanh_k": [0.1],
    }
    config.update(network_config)
    return config
