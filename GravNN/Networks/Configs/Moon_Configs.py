from GravNN.CelestialBodies.Planets import Moon
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Trajectories import RandomDist

grav_model = SphericalHarmonics(Moon().GRGM1200, 2)


def get_default_moon_config():
    data_config = {
        "planet": [Moon()],
        "sh_file": [Moon().sh_file],
        "distribution": [RandomDist],
        "N_dist": [1000000],
        "N_train": [950000],
        "N_val": [50000],
        "radius_min": [Moon().radius],
        "radius_max": [Moon().radius + 50000.0],
        "acc_noise": [0.00],
        "basis": [None],
        "deg_removed": [-1],
        "mixed_precision": [False],
        "max_deg": [1000],
        "analytic_truth": ["sh_stats_moon_"],
        "gravity_data_fcn": [get_sh_data],
        "obj_file": [Moon().obj_file],
        "mu": [Moon().mu],
        "cBar": [grav_model.C_lm],
        "sBar": [grav_model.S_lm],
    }
    return data_config
