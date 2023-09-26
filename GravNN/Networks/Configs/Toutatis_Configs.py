from GravNN.CelestialBodies.Asteroids import Toutatis
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Trajectories import RandomDist


def get_prototype_toutatis_config():
    data_config = {
        "planet": [Toutatis()],
        "obj_file": [Toutatis().model_lf],
        "distribution": [RandomDist],
        "N_dist": [10000],
        "N_train": [2500],
        "N_val": [150],
        "radius_min": [0],
        "radius_max": [Toutatis().radius + 5000.0],
        "acc_noise": [0.00],
        "basis": [None],
        "mixed_precision": [False],
        "dtype": ["float32"],
        "analytic_truth": ["poly_stats_"],
        "gravity_data_fcn": [get_poly_data],
    }

    return data_config
