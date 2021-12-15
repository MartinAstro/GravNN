from GravNN.CelestialBodies.Planets import Moon
from GravNN.Trajectories import RandomDist
from GravNN.Preprocessors import DummyScaler, UniformScaler
from sklearn.preprocessing import MinMaxScaler


def get_default_moon_config():
    data_config = {
        "planet": [Moon()],
        "grav_file": [Moon().sh_file],
        "distribution": [RandomDist],
        "N_dist": [1000000],
        "N_train": [950000],
        "N_val": [50000],
        "radius_min": [Moon().radius],
        "radius_max": [Moon().radius + 50000.0],
        "acc_noise": [0.00],
        "basis": [None],
        "deg_removed": [2],
        "mixed_precision": [False],
        "max_deg": [1000],
        "analytic_truth": ["sh_stats_moon_"],
    }
    network_config = {
        "network_type": ["traditional"],
        "PINN_constraint_fcn": ["no_pinn"],
        "layers": [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
        "activation": ["tanh"],
        "init_file": [None],  #'2459192.4530671295'],
        "epochs": [100000],
        "initializer": ["glorot_normal"],
        "optimizer": ["adam"],
        "batch_size": [40000],
        "lr_anneal": [False],
        "dropout": [0.0],
        "x_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "a_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "u_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "dtype": ["float32"],
        "dummy_transformer": [DummyScaler()],
        "class_weight": [[1.0, 1.0, 1.0]],  # no_pinn and PINN_A
        "learning_rate": [0.001],
        "skip_normalization": [False],
        "seed" : [0],
    }
    config = {}
    config.update(data_config)
    config.update(network_config)
    return config


def get_default_moon_pinn_config():
    data_config = {
        "planet": [Moon()],
        "grav_file": [Moon().sh_file],
        "distribution": [RandomDist],
        "N_dist": [1000000],
        "N_train": [95000],
        "N_val": [50000],
        "radius_min": [Moon().radius],
        "radius_max": [Moon().radius + 50000.0],
        "initializer": ["glorot_normal"],
        "acc_noise": [0.00],
        "basis": [None],  # ['spherical'],
        "deg_removed": [2],
        "mixed_precision": [False],
        "max_deg": [1000],
        "analytic_truth": ["sh_stats_moon_"],
    }
    network_config = {
        "network_type": ["traditional"],
        "PINN_constraint_fcn": ["pinn_A"],
        "layers": [[3, 40, 40, 40, 40, 40, 40, 40, 40, 1]],
        "activation": ["tanh"],
        "init_file": [None],  #'2459192.4530671295'],
        "epochs": [100000],
        "optimizer": ["adam"],  # (learning_rate=config['lr_scheduler'][0])
        "batch_size": [40000],
        "dropout": [0.0],
        "lr_anneal": [False],
        "x_transformer": [MinMaxScaler(feature_range=(-1, 1))],
        "u_transformer": [UniformScaler(feature_range=(-1, 1))],
        "a_transformer": [UniformScaler(feature_range=(-1, 1))],
        "dtype": ["float32"],
        "dummy_transformer": [DummyScaler()],
        "class_weight": [[1.0, 1.0, 1.0]],  # no_pinn and PINN_A
        "learning_rate": [0.001],
        "skip_normalization": [False],
    }
    config = {}
    config.update(data_config)
    config.update(network_config)
    return config
