
from GravNN.Networks.Configs import *

def get_r_experiment():
    hparams = {
        "grav_file" : [Eros().obj_200k],
        "radius_max" : [Eros().radius * 3],
        "radius_min" : [0],
        "N_train" : [5000],
        "acc_noise" : [0.0, 0.2],
    }
    return hparams

def get_r_star_experiment():
    hparams = {
        "grav_file" : [Eros().obj_200k],
        "radius_max" : [Eros().radius * 3],
        "radius_min" : [Eros().radius * 2],
        "N_train" : [5000],
        "acc_noise" : [0.0, 0.2],

    }
    return hparams

def get_r_bar_experiment():
    hparams = {
        "grav_file" : [Eros().obj_200k],
        "radius_max" : [Eros().radius * 3],
        "radius_min" : [Eros().radius * 2],
        "N_train" : [5000],
        "acc_noise" : [0.0, 0.2],

        "extra_distribution" : [RandomAsteroidDist],
        "extra_radius_min" : [0],
        "extra_radius_max" : [Eros().radius*2],
        "extra_N_dist" : [1000],
        "extra_N_train" : [500],
        "extra_N_val" : [500],
    }
    return hparams