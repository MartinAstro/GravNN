import multiprocessing as mp
from script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():
    df_file = "Data/Dataframes/eros_residual_r.data"

    threads = 2
    config = get_default_eros_config()

    hparams = {
        "grav_file" : [Eros().obj_200k],
        "radius_max" : [Eros().radius * 3],
        "radius_min" : [Eros().radius * 2],
        "N_train" : [2500],
        "acc_noise" : [0.0, 0.2],
        "network_type" : ['sph_pines_traditional'],

        "extra_distribution" : [RandomAsteroidDist],
        "extra_radius_min" : [0],
        "extra_radius_max" : [Eros().radius*2],
        "extra_N_dist" : [1000],
        "extra_N_train" : [250],
        "extra_N_val" : [500],

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
    hparams.update(ReduceLrOnPlateauConfig())

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
    from GravNN.Networks.Callbacks import SimpleCallback
    from GravNN.Networks.Data import get_preprocessed_data, configure_dataset, compute_input_layer_normalization_constants
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Networks import load_network
    from GravNN.Networks.utils import load_hparams_to_config, configure_optimizer
    from GravNN.Networks.Schedules import get_schedule

    tf = configure_tensorflow()
    mixed_precision = set_mixed_precision() if config_original['mixed_precision'][0] else None
    try:
        np.random.seed(hparams['seed'])
        tf.random.set_seed(hparams['seed'])
    except:
        np.random.seed(config_original['seed'][0])
        tf.random.set_seed(config_original['seed'][0])
    #tf.config.run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    # Standardize Configuration
    config = copy.deepcopy(config_original)
    config = load_hparams_to_config(hparams, config)

    check_config_combos(config)
    print(config)

    # Get data, network, optimizer, and generate model
    train_data, val_data, transformers = get_preprocessed_data(config)
    compute_input_layer_normalization_constants(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    optimizer = configure_optimizer(config, mixed_precision)
    network = load_network(config)
    model = CustomModel(config, network)
    model.compile(optimizer=optimizer, loss="mse")
    
    # Train network
    callback = SimpleCallback()
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
    model.config["a_bar_transformer"][0] = transformers["a_bar"]

    model.save(df_file=None)

    # import matplotlib.pyplot as plt
    # plt.plot(history.history['val_loss'][1000:])
    # plt.show()
    # Appends the model config to a perscribed df
    return model.config


if __name__ == "__main__":
    main()
