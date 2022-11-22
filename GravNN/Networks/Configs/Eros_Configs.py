from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Trajectories import RandomDist
from GravNN.Preprocessors import DummyScaler, UniformScaler
from sklearn.preprocessing import MinMaxScaler


def get_default_eros_config():
    data_config = {
        "planet": [Eros()],
        "grav_file": [Eros().obj_200k],
        "distribution": [RandomDist],
        "N_dist": [20000],
        "N_train": [2500],
        "N_val": [1500],
        "radius_min": [0],
        "radius_max": [Eros().radius * 3],
        "acc_noise": [0.0],
        "basis": [None],
        "analytic_truth": ["poly_stats_"],
        "gravity_data_fcn" : [get_poly_data],
        "remove_point_mass": [False],  # remove point mass from polyhedral model
        "override" : [False],
        "ref_radius" : [Eros().radius],
        "ref_radius_min" : [Eros().radius_min],
    }

    return data_config