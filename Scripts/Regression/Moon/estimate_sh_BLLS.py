import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from GravNN.Networks.Configs import *
from GravNN.Networks.Data import get_preprocessed_data
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import format_coefficients, save

np.random.seed(1234)


def regress_sh_model(config, max_deg, noise, idx):
    print("Max Deg: " + str(max_deg))
    # Get randomly shuffled data
    planet = config["planet"][0]
    dataset, val_dataset, transformers = get_preprocessed_data(config)
    x = dataset[0]
    a = dataset[2]
    N_train = config["N_train"][0]

    file_name = "regress_%d_%.1f_%d_%d.csv" % (max_deg, noise, idx, N_train)
    obj_file = os.path.join(
        os.path.abspath("."),
        "GravNN",
        "Files",
        "GravityModels",
        "Regressed",
        "Moon",
        file_name,
    )
    regressor = BLLS(max_deg, planet, remove_deg=2)  #
    results = regressor.update(x, a)
    C_lm, S_lm = format_coefficients(results, max_deg, 2)
    save(obj_file, planet, C_lm, S_lm)
    return file_name, max_deg, noise, idx


def generate_args(config, num_models, noise_list, deg_list):
    args = []
    for idx in range(num_models):
        for noise in noise_list:
            for deg in deg_list:
                config["acc_noise"] = [noise]
                config["seed"] = [idx]
                args.append((config, deg, noise, idx))
    return args


def regression_9500_v1():
    """Multiprocessed version of generate models. Trains multiple networks
    simultaneously to speed up regression.
    """
    config = get_default_moon_config()
    config["N_train"] = [9500]
    config["N_val"] = [500]
    config["scale_by"] = ["none"]
    model_deg = 40
    model_interval = 10
    num_models = 3

    noise_list = [0.0, 0.2]
    deg_list = np.arange(3, model_deg, model_interval, dtype=int)
    model_id_list = np.arange(0, num_models, 1, dtype=int)

    sh_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [noise_list, deg_list, model_id_list],
            names=["noise", "degree", "id"],
        ),
        columns=["model_identifier"],
    )

    pool = mp.Pool(1)
    args = generate_args(config, num_models, noise_list, deg_list)
    results = pool.starmap_async(regress_sh_model, args)
    tuples = results.get()
    for tuple in tuples:
        sh_model_name, deg, noise, idx = tuple
        sh_df.loc[(noise, deg, idx)] = sh_model_name

    sh_df.to_pickle("Data/Dataframes/Regression/Moon_SH_regression_9500_v1.data")


def regression_5000_v1():
    """Multiprocessed version of generate models. Trains multiple networks
    simultaneously to speed up regression.
    """
    config = get_default_moon_config()
    config["N_train"] = [5000]
    config["N_val"] = [500]
    config["scale_by"] = ["none"]
    model_deg = 63
    model_interval = 10
    num_models = 3

    noise_list = [0.0, 0.2]
    deg_list = np.arange(3, model_deg, model_interval, dtype=int)
    model_id_list = np.arange(0, num_models, 1, dtype=int)

    sh_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [noise_list, deg_list, model_id_list],
            names=["noise", "degree", "id"],
        ),
        columns=["model_identifier"],
    )

    pool = mp.Pool(1)
    args = generate_args(config, num_models, noise_list, deg_list)
    results = pool.starmap_async(regress_sh_model, args)
    tuples = results.get()
    for tuple in tuples:
        sh_model_name, deg, noise, idx = tuple
        sh_df.loc[(noise, deg, idx)] = sh_model_name

    sh_df.to_pickle("Data/Dataframes/Regression/Moon_SH_regression_5000_v1.data")


if __name__ == "__main__":
    # regression_9500_v1()
    regression_5000_v1()
