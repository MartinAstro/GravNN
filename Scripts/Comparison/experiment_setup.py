from interfaces import make_experiments

from GravNN.Networks.utils import permutate_dict


def setup_experiments():
    config_list = []

    # config_list.append(permutate_dict({
    #     "model_name" : ["PM"],
    #     "N_train" : [500, 50000],
    #     "acc_noise" : [0.0, 0.1],
    #     }),
    # )

    config_list.append(
        permutate_dict(
            {
                "model_name": ["SH"],
                "N_train": [500],
                "acc_noise": [0.0, 0.1],
                "deg": [8, 128],
            },
        ),
    )

    # config_list.append(permutate_dict({
    #     "model_name" : ["MASCONS"],
    #     "N_train" : [500, 50000],
    #     "acc_noise" : [0.0, 0.1],
    #     "elements" : [500, 50000],
    # }))

    # from GravNN.CelestialBodies.Asteroids import Eros
    # config_list.append(permutate_dict({
    #     "model_name" : ["POLYHEDRAL"],
    #     "N_train" : [None],
    #     "acc_noise" : [None],
    #     "shape" : [Eros().obj_8k, Eros().obj_200k],
    # }))

    # # Numerical
    # config_list.append(permutate_dict({
    #     "model_name" : ["ELM"],
    #     "N_train" : [500, 50000],
    #     "acc_noise" : [0.0, 0.1],
    #     "num_units" : [1000, 10000],
    # }))

    # config_list.append(permutate_dict({
    #     "model_name" : ["NN"],
    #     "N_train" : [500, 50000],
    #     "acc_noise" : [0.0, 0.1],
    #     "num_units" : [20, 40],
    # }))

    # config_list.append(permutate_dict({
    #     "model_name" : ["PINN"],
    #     "N_train" : [500, 50000],
    #     "acc_noise" : [0.0, 0.1],
    #     "num_units" : [20, 40],
    # }))

    experiments = make_experiments(config_list)
    return experiments
