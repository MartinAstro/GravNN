from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.Trajectories import RandomDist


def get_default_eros_config():
    data_config = {
        "planet": [Eros()],
        "mu": [Eros().mu],
        "obj_file": [Eros().obj_8k],
        # "obj_file": [Eros().obj_200k],
        "distribution": [RandomDist],
        "N_dist": [50000],
        "N_train": [2500],
        "N_val": [1500],
        "radius_min": [0.0],
        "radius_max": [Eros().radius * 3],
        "acc_noise": [0.0],
        "basis": [None],
        "analytic_truth": ["poly_stats_"],
        "gravity_data_fcn": [get_hetero_poly_data],
        "remove_point_mass": [False],  # remove point mass from polyhedral model
        "override": [False],
        "ref_radius": [Eros().radius],
        "ref_radius_min": [Eros().radius_min],
        "ref_radius_max": [Eros().radius],
        "deg_removed": [-1],
    }

    return data_config
