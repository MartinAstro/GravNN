from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
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
        "acc_noise": [0.0],
        "basis": [None],
        "mixed_precision": [False],
        "dtype": ["float32"],
        "analytic_truth": ["poly_stats_"],
        "gravity_data_fcn" : [get_poly_data],
        "remove_point_mass": [False],  # remove point mass from polyhedral model
        "x_transformer": [UniformScaler(feature_range=(-1, 1))],
        "u_transformer": [UniformScaler(feature_range=(-1, 1))],
        "a_transformer": [UniformScaler(feature_range=(-1, 1))],
        "dummy_transformer": [DummyScaler()],
        "override" : [False],
        'ref_radius' : [Eros().radius]
    }

    return data_config