import os
from pprint import pprint

import tensorflow as tf

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

df_file = "Data/Dataframes/earth_revisited_032723.data"


def main():
    config = get_default_earth_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        # "N_dist": [5000000],
        # "N_train": [4900000],
        # "N_val": [100000],
        "N_dist": [50000],
        "N_train": [45000],
        "N_val": [5000],
        "num_units": [20],
        "loss_fcns": [["percent", "rms"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.0001],
        "batch_size": [2**12],
        "epochs": [20],
        "preprocessing": [["pines", "r_inv"]],
        "patience": [5000],
        "dropout": [0.0],
        # "batch_size": [4500],
        "min_delta": [0.01],
        "acc_noise": [0.0],
        # "preprocessing": [["pines", "r_inv", "fourier"]],
        "PINN_constraint_fcn": ["pinn_a"],
        # "fourier_features": [5],
        "fourier_sigma": [1],
        # "shared_freq": [False],
        # "shared_offset": [False],
        # "trainable": [True],
        "base_2_init": [True],
        "fourier_features": [10],
        "trainable": [False],
        "shared_freq": [True],
        "shared_offset": [True],
        "tanh_k": [3],
        "tanh_r": [Earth().radius * 10],
        # "augment_data_config": [
        #     {
        #         "N_dist": [105000],
        #         "N_train": [100000],
        #         "N_val": [5000],
        #         "radius_min": [Earth().radius + 420000],
        #         "radius_max": [Earth().radius * 10],
        #     },
        # ],
    }
    args = configure_run_args(config, hparams)
    configs = [run(*args[0])]
    save_training(df_file, configs)

    # run with larger BS for longer
    tf.keras.backend.clear_session()
    hparams.update(
        {
            "init_file": [configs[0]["id"][0]],
            "batch_size": [2**16],
            "learning_rate": [0.01],
            "epochs": [20],
        },
    )
    args = configure_run_args(config, hparams)
    configs = [run(*args[0])]
    save_training(df_file, configs)


def run(config):
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel, load_config_and_model
    from GravNN.Networks.Saver import ModelSaver
    from GravNN.Networks.utils import configure_tensorflow, populate_config_objects

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    pprint(config)

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    if config["init_file"][0] is not None:
        _, model = load_config_and_model(
            config["init_file"][0],
            df_file,
            custom_dtype=config["dtype"][0],
            only_weights=True,
        )
    else:
        model = PINNGravityModel(config)
    history = model.train(data)
    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
