from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Trajectories import RandomDist
from GravNN.Preprocessors import DummyScaler, UniformScaler
from sklearn.preprocessing import MinMaxScaler

from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
grav_model = SphericalHarmonics(Earth().EGM2008,2)


def get_default_earth_config():
    data_config = {
        "planet": [Earth()],
        "grav_file": [Earth().sh_file],
        "distribution": [RandomDist],
        "N_dist": [1000000],
        "N_train": [950000],
        "N_val": [50000],
        "radius_min": [Earth().radius],
        "radius_max": [Earth().radius + 420000.0],
        "ref_radius" : [Earth().radius],
        "acc_noise": [0.00],
        "basis": [None],
        "deg_removed": [-1],
        "mixed_precision": [False],
        "max_deg": [1000],
        "analytic_truth": ["sh_stats_"],
        "gravity_data_fcn" : [get_sh_data],
        "shape_model" : [Earth().shape_model],
        "mu" : [Earth().mu],
        "cBar" : [grav_model.C_lm],
        "sBar" : [grav_model.S_lm]
    }
    return data_config