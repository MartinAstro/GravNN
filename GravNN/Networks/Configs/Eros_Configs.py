from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Trajectories import RandomAsteroidDist
from GravNN.Preprocessors import DummyScaler, UniformScaler
from sklearn.preprocessing import MinMaxScaler


def get_default_eros_config():
    data_config = {
        "planet": [Eros()],
        "grav_file": [Eros().obj_200k],
        "distribution": [RandomAsteroidDist],
        "N_dist": [20000],
        "N_train": [2500],
        "N_val": [1500],
        "radius_min": [0],
        "radius_max": [Eros().radius * 3],
        "acc_noise": [0.00],
        "basis": [None],
        "mixed_precision": [False],
        "dtype": ["float32"],
        "analytic_truth": ["poly_stats_"],
        "remove_point_mass": [False],  # remove point mass from polyhedral model
        "x_transformer": [UniformScaler(feature_range=(-1, 1))],
        "u_transformer": [UniformScaler(feature_range=(-1, 1))],
        "a_transformer": [UniformScaler(feature_range=(-1, 1))],
        "scale_by": ["non_dim"],
        "dummy_transformer": [DummyScaler()],
    }
    network_config = {
        "PINN_constraint_fcn": ["pinn_A"],
        "layers": [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
        "activation": ["gelu"],
        "init_file": [None],
        "epochs": [7500],
        "initializer": ["glorot_normal"],
        "optimizer": ["adam"],
        "batch_size": [131072 // 2],
        "dropout": [0.0],
        "dtype": ["float32"],
        "skip_normalization": [False],
        "lr_anneal": [False],
        "input_layer": [False],
        "network_type": ["sph_pines_traditional"],
        "custom_input_layer": [None],
        "ref_radius" : [Eros().radius]
    }

    config = {}
    config.update(data_config)
    config.update(network_config)
    return config


def get_eros_r_star_config(multiplier=1):
    config = get_default_eros_config()
    modifications = {
        "N_train": [2500 * multiplier],
        "N_val": [5000],
        "radius_min": [Eros().radius + 5000.0],
        "radius_max": [Eros().radius + 10000.0],
    }
    config.update(modifications)


def get_eros_r_bar_config(multiplier=1):
    config = get_default_eros_config()
    modifications = {
        "N_train": [2500 * multiplier],
        "N_val": [1500 * multiplier],
        "radius_min": [Eros().radius + 5000.0],
        "radius_max": [Eros().radius + 10000.0],
        "extra_distribution": [RandomAsteroidDist],
        "extra_radius_min": [0],
        "extra_radius_max": [Eros().radius + 5000.0],
        "extra_N_dist": [50000],
        "extra_N_train": [250 * multiplier],
        "extra_N_val": [5000],
    }
    config.update(modifications)
