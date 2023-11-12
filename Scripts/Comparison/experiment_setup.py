from interfaces import make_experiments

from GravNN.Networks.utils import permutate_dict


def setup_experiments():
    config_list = []

    config_list.append(
        permutate_dict(
            {
                "model_name": ["PM"],
                "N_train": [500, 50000],
                "acc_noise": [0.0],
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
                "acc_noise": [0.0],
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
                "acc_noise": [0.0],
                "shape": [Eros().obj_77, Eros().obj_20k],
            },
        ),
    )

    # Numerical
    config_list.append(
        permutate_dict(
            {
                "model_name": ["ELM"],
                "N_train": [500, 50000],
                "acc_noise": [0.0],
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
                "acc_noise": [0.0],
                "num_units": [8],
                "layers": [[3, 1, 1, 3]],
            },
        ),
    )
    # Large
    config_list.append(
        permutate_dict(
            {
                "model_name": ["TNN_Large"],
                "N_train": [500, 50000],
                "acc_noise": [0.0],
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
                "acc_noise": [0.0],
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
                "acc_noise": [0.0],
                "num_units": [64],
                "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
            },
        ),
    )

    experiments = make_experiments(config_list)
    return experiments
