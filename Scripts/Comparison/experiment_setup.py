from interfaces import make_experiments

from GravNN.Networks.Configs import *
from GravNN.Networks.utils import permutate_dict


def get_default_config(model_name):
    config = get_default_eros_config()
    config.update(
        {
            "obj_file": [Eros().obj_200k],
            "N_dist": [100000],
            "N_val": [4096],
        },
    )

    if "PINN_III" in model_name:
        base_config = PINN_III()
    elif "PINN_II" in model_name:
        base_config = PINN_II()
    elif "PINN_I" in model_name:
        base_config = PINN_I()
    elif "TNN" in model_name:
        base_config = NN()
    else:
        # For all non-PINN models, use PINN III as the base config
        base_config = PINN_III()
    base_config.update(ReduceLrOnPlateauConfig())
    base_config.update(
        {
            "learning_rate": [2**-10],
            "epochs": [8192 * 2],
            "batch_size": [2**11],
        },
    )
    config.update(base_config)
    return config


def setup_experiments():
    config_list = []

    config_list.append(
        permutate_dict(
            {
                "model_name": ["PM"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
            },
        ),
    )

    config_list.append(
        permutate_dict(
            {
                "model_name": ["SH"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "deg": [15, 175],
            },
        ),
    )

    config_list.append(
        permutate_dict(
            {
                "model_name": ["MASCONS"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "elements": [55, 7500],
            },
        ),
    )

    from GravNN.CelestialBodies.Asteroids import Eros

    config_list.append(
        permutate_dict(
            {
                "model_name": ["POLYHEDRAL"],
                "N_train": [500],
                "acc_noise": [0.0, 0.1],
                "shape": [Eros().obj_66, Eros().obj_10k],
            },
        ),
    )

    # Numerical
    config_list.append(
        permutate_dict(
            {
                "model_name": ["ELM"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [40, 5100],
            },
        ),
    )

    # NN
    # Small
    config_list.append(
        permutate_dict(
            {
                "model_name": ["TNN_Small"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [8],
                "layers": [[3, 1, 1, 1, 1, 3]],
            },
        ),
    )
    # Large
    config_list.append(
        permutate_dict(
            {
                "model_name": ["TNN_Large"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [64],
                "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
            },
        ),
    )

    # PINN
    # Small
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_I_Small"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [8],
                "layers": [[3, 1, 1, 1, 1, 3]],
            },
        ),
    )
    # Large
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_I_Large"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [64],
                "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
            },
        ),
    )

    # PINN
    # Small
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_II_Small"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [8],
                "layers": [[3, 1, 1, 3]],
            },
        ),
    )
    # Large
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_II_Large"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [64],
                "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
            },
        ),
    )

    # PINN
    # Small
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_III_Small"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [8],
                "layers": [[3, 1, 1, 3]],
            },
        ),
    )
    # Large
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_III_Large"],
                "N_train": [500, 50000],
                "acc_noise": [0.0, 0.1],
                "num_units": [64],
                "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
            },
        ),
    )

    experiments = make_experiments(config_list)
    return experiments


def setup_fast_experiments():
    config_list = []

    config_list.append(
        permutate_dict(
            {
                "model_name": ["PM"],
                "N_train": [500],
                "acc_noise": [0.0],
            },
        ),
    )

    config_list.append(
        permutate_dict(
            {
                "model_name": ["SH"],
                "N_train": [500],
                "acc_noise": [0.0],
                "deg": [15],
            },
        ),
    )

    config_list.append(
        permutate_dict(
            {
                "model_name": ["MASCONS"],
                "N_train": [500],
                "acc_noise": [0.0],
                "elements": [55],
            },
        ),
    )

    from GravNN.CelestialBodies.Asteroids import Eros

    config_list.append(
        permutate_dict(
            {
                "model_name": ["POLYHEDRAL"],
                "N_train": [500],
                "acc_noise": [0.0],
                "shape": [Eros().obj_77],
            },
        ),
    )

    # Numerical
    config_list.append(
        permutate_dict(
            {
                "model_name": ["ELM"],
                "N_train": [500],
                "acc_noise": [0.0],
                "num_units": [40],
            },
        ),
    )

    # NN
    # Small
    config_list.append(
        permutate_dict(
            {
                "model_name": ["TNN_Small"],
                "N_train": [500],
                "acc_noise": [0.0],
                "num_units": [8],
                "layers": [[3, 1, 1, 3]],
            },
        ),
    )

    # PINN
    # Small
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_I_Small"],
                "N_train": [500],
                "acc_noise": [0.0],
                "num_units": [8],
                "layers": [[3, 1, 1, 3]],
            },
        ),
    )

    # PINN
    # Small
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_II_Small"],
                "N_train": [500],
                "acc_noise": [0.0],
                "num_units": [8],
                "layers": [[3, 1, 1, 3]],
            },
        ),
    )

    # PINN
    # Small
    config_list.append(
        permutate_dict(
            {
                "model_name": ["PINN_III_Small"],
                "N_train": [500],
                "acc_noise": [0.0],
                "num_units": [8],
                "layers": [[3, 1, 1, 3]],
            },
        ),
    )

    experiments = make_experiments(config_list)
    return experiments


def setup_debug_experiments():
    config_list = []

    config_list.append(
        permutate_dict(
            {
                "model_name": ["SH"],
                "N_train": [500],
                "acc_noise": [0.0],
                "deg": [175],
            },
        ),
    )

    experiments = make_experiments(config_list)
    return experiments
