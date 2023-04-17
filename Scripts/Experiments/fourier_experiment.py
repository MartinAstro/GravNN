import multiprocessing as mp
import os
from pprint import pprint

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    threads = 4

    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [50000],
        "N_train": [45000],
        "N_val": [5000],
        "num_units": [40],
        "loss_fcns": [["percent", "rms"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.0001],
        "dropout": [0.0],
        "batch_size": [45000],
        "epochs": [10000],
        "acc_noise": [0.0],
        "preprocessing": [["pines", "r_inv", "fourier"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "tanh_r": [10],  # point mass should have minimal role
        "tanh_k": [1e3],
        "fuse_model": [True],
    }

    fourier_hparams = {
        "fourier_features": [5, 10, 20],
        "fourier_sigma": [1],
        "shared_freq": [False],
        "shared_offset": [False],
        "trainable": [False],
    }

    hparams.update(fourier_hparams)

    df_file = "Data/Dataframes/fourier_experiment_not_trainable_032523.data"
    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)

    hparams.update({"trainable": [True]})
    df_file = "Data/Dataframes/fourier_experiment_trainable_032523.data"
    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)

    hparams.update({"shared_freq": [True]})
    hparams.update({"shared_offset": [True]})
    df_file = "Data/Dataframes/fourier_experiment_trainable_shared_032523.data"
    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def run(config):
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.Saver import ModelSaver
    from GravNN.Networks.utils import configure_tensorflow, populate_config_objects

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    pprint(config)

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)

    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
