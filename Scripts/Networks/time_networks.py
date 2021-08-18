import multiprocessing as mp
import pandas as pd
from script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *

def main():
    #5,000,000 data --  Plus a little extra epochs for the best performing two 
    df_file = 'Data/Dataframes/hyperparameter_earth_time.data'
    directory = 'logs/hyperparameter_earth_time/'
    hparams = {
        'N_dist' : [5000000],
        'N_train' : [100000, 250000, 500000, 750000, 1000000, 2500000, 4900000],
        'epochs' : [30],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [50000], # the last one is to simulate virtually no decay
        'decay_epoch_0' : [50000],
        'decay_rate' : [2.0],
        'learning_rate' : [2E-2],
        'batch_size': [2**14,2**15,2**16,2**17,2**18,2**19, 2**20],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
        'schedule_type' : ['none']
        #'init_file' : [2459314.280798611]
    }

    threads = 1
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
    )
    from GravNN.Networks.Callbacks import TimingCallback
    from GravNN.Networks.Data import get_preprocessed_data, configure_dataset
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Networks import load_network
    from GravNN.Networks.utils import load_hparams_to_config, configure_optimizer
    from GravNN.Networks.Schedules import get_schedule

    tf = configure_tensorflow()
    mixed_precision = set_mixed_precision() if config_original['mixed_precision'][0] else None
    np.random.seed(1234)
    tf.random.set_seed(0)

    tf.keras.backend.clear_session()

    # Standardize Configuration
    config = copy.deepcopy(config_original)
    config = load_hparams_to_config(hparams, config)
    check_config_combos(config)
    print(config)

    # Get data, network, optimizer, and generate model
    train_data, val_data, transformers = get_preprocessed_data(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    optimizer = configure_optimizer(config, mixed_precision)
    network = load_network(config)
    model = CustomModel(config, network)
    model.compile(optimizer=optimizer, loss="mse")  # , run_eagerly=True)

    # Train network
    callback = TimingCallback()
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