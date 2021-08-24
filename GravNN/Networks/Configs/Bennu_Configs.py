from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.Trajectories import RandomAsteroidDist
from GravNN.Preprocessors import DummyScaler, UniformScaler
from sklearn.preprocessing import MinMaxScaler


def get_default_bennu_config():
    data_config = {
        "planet": [Bennu()],
        "grav_file": [Bennu().stl_200k],
        "distribution": [RandomAsteroidDist],
        "N_dist": [20000],
        "N_train": [2500],
        "N_val": [1500],
        "radius_min": [0],
        "radius_max": [Bennu().radius*3],
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
    }

    config = {}
    config.update(data_config)
    config.update(network_config)
    return config


def get_Bennu_r_star_config(multiplier=1):
    config = get_default_bennu_config()
    modifications = {
        "N_train": [2500 * multiplier],
        "N_val": [5000],
        "radius_min": [Bennu().radius],
        "radius_max": [Bennu().radius*5],
    }
    config.update(modifications)


def get_Bennu_r_bar_config(multiplier=1):
    config = get_default_bennu_config()
    modifications = {
        "N_train": [2500 * multiplier],
        "N_val": [1500 * multiplier],
        "radius_min": [Bennu().radius],
        "radius_max": [Bennu().radius*5],
        "extra_distribution": [RandomAsteroidDist],
        "extra_radius_min": [0],
        "extra_radius_max": [Bennu().radius],
        "extra_N_dist": [5000],
        "extra_N_train": [250 * multiplier],
        "extra_N_val": [500],
    }
    config.update(modifications)
