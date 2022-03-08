from GravNN.CelestialBodies.Planets import Moon
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
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
        "gravity_data_fcn" : [get_sh_data],
        "shape_model" : [Moon().shape_model]
    }
    return data_config

