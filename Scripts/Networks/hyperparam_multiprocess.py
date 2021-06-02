import multiprocessing as mp
import pandas as pd
from script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *

def main():
    df_file = "Data/Dataframes/useless_05_28_21.data"
    threads = 6

    #config = get_default_earth_config()
    #config = get_default_earth_pinn_config()
    config = get_default_eros_pinn_config()

    hparams = {
        "N_dist": [100000],
        "N_train": [49000],
        "epochs": [1000],
        "decay_rate_epoch": [25000],
        "decay_epoch_0": [25000],
        "decay_rate": [0.5],
        "learning_rate": [0.005],
        "batch_size": [131072 * 2],
        "activation": ["gelu"],
        "initializer": ["glorot_uniform"],
        "network_type": ["traditional"],
        "PINN_constraint_fcn": [ "pinn_A"],
        "scale_by": ["a"],
        "mixed_precision": [False],
        "num_units": [20],
        "schedule_type" : ['exp_decay'],

        "beta" : [0.9],

    }

    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def run(config_original, hparams):
    import copy
    import numpy as np
    from GravNN.Networks.utils import (
        configure_tensorflow,
        set_mixed_precision,
        check_config_combos,
        format_config_combos,
    )
    from GravNN.Networks.Callbacks import CustomCallback
    from GravNN.Networks.Data import get_preprocessed_data, configure_dataset
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Networks import load_network
    from GravNN.Networks.utils import load_hparams_to_config, configure_optimizer
    from GravNN.Networks.Schedules import get_schedule

    tf = configure_tensorflow()
    mixed_precision = set_mixed_precision() if config_original['mixed_precision'][0] else None
    np.random.seed(1234)
    tf.random.set_seed(0)
    #tf.config.run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    # Standardize Configuration
    config = copy.deepcopy(config_original)
    config = load_hparams_to_config(hparams, config)
    check_config_combos(config)
    config = format_config_combos(config)
    print(config)

    # Get data, network, optimizer, and generate model
    train_data, val_data, transformers = get_preprocessed_data(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    optimizer = configure_optimizer(config, mixed_precision)
    network = load_network(config)
    model = CustomModel(config, network)
    model.compile(optimizer=optimizer, loss="mse")

    # Train network
    callback = CustomCallback()
    schedule = get_schedule(config)

    history = model.fit(
        dataset,
        epochs=config["epochs"][0],
        verbose=0,
        validation_data=val_dataset,
        callbacks=[callback, schedule],
    )
    history.history["time_delta"] = callback.time_delta
    model.history = history

    # Save network and config information
    model.config["time_delta"] = [callback.time_delta]
    model.config["x_transformer"][0] = transformers["x"]
    model.config["u_transformer"][0] = transformers["u"]
    model.config["a_transformer"][0] = transformers["a"]

    # Appends the model config to a perscribed df
    model.save(df_file=None)
    return model.config


if __name__ == "__main__":
    main()