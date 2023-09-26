from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Preprocessors import DummyScaler, UniformScaler
from GravNN.Trajectories import RandomDist


def get_default_bennu_config():
    data_config = {
        "planet": [Bennu()],
        "obj_file": [Bennu().stl_200k],
        "distribution": [RandomDist],
        "N_dist": [20000],
        "N_train": [2500],
        "N_val": [1500],
        "radius_min": [0],
        "radius_max": [Bennu().radius * 3],
        "acc_noise": [0.00],
        "basis": [None],
        "mixed_precision": [False],
        "dtype": ["float32"],
        "analytic_truth": ["poly_stats_"],
        "gravity_data_fcn": [get_poly_data],
        "remove_point_mass": [False],  # remove point mass from polyhedral model
        "x_transformer": [UniformScaler(feature_range=(-1, 1))],
        "u_transformer": [UniformScaler(feature_range=(-1, 1))],
        "a_transformer": [UniformScaler(feature_range=(-1, 1))],
        "dummy_transformer": [DummyScaler()],
    }

    return data_config


def get_bennu_r_star_config(multiplier=1):
    config = get_default_bennu_config()
    modifications = {
        "N_train": [2500 * multiplier],
        "N_val": [5000],
        "radius_min": [Bennu().radius + 5000.0],
        "radius_max": [Bennu().radius + 10000.0],
    }
    config.update(modifications)


def get_bennu_r_bar_config(multiplier=1):
    {
        "N_train": [2500 * multiplier],
        "N_val": [1500 * multiplier],
        "radius_min": [Bennu().radius + 5000.0],
        "radius_max": [Bennu().radius + 10000.0],
        "extra_distribution": [RandomDist],
        "extra_radius_min": [0],
        "extra_radius_max": [Bennu().radius + 5000.0],
        "extra_N_dist": [1000],
        "extra_N_train": [250 * multiplier],
        "extra_N_val": [500],
    }
