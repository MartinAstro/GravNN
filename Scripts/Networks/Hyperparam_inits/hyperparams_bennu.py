    
from GravNN.Networks.Configs import *

def get_r_experiment():
    hparams = {
        "grav_file" : [Bennu().stl_200k],
        "radius_max" : [Bennu().radius * 3],
        "radius_min" : [Bennu().min_radius],
        "N_train" : [5000],
        "acc_noise" : [0.0],#, 0.2],
        "network_type" : ['sph_pines_transformer'],
        'PINN_constraint_fcn' : ['pinn_alc'],
        'normalization_strategy' : ['radial'],#, 'uniform'],
        'ref_radius_max' : [Bennu().radius],
        'ref_radius_min' : [Bennu().min_radius],
        "N_val" : [1500],
        "epochs": [7500],
        "dropout" : [0.0],

        "learning_rate": [0.001*2],
        "batch_size": [131072 // 2],
        "beta" : [0.9],

        #'batch_norm' :[True],
        "remove_point_mass" : [False], # remove point mass from polyhedral model
        "override" : [False]
    }
    return hparams