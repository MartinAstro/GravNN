import multiprocessing as mp
from numba.core.types.abstract import Dummy
import pandas as pd
from script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
from GravNN.Preprocessors.UniformScaler import UniformScaler

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():
    #df_file = "Data/Dataframes/useless_062221_potential.data" # "pinn_a", "pinn_p", "pinn_ap"
    df_file = "Data/Dataframes/useless_062321_potential.data" # spherical hand norm layer
    df_file = "Data/Dataframes/useless_062421_potential_v7.data" # No spherical and norm layer
    df_file = "Data/Dataframes/useless_062521_potential_v3.data" # Simple scaling
    df_file = "Data/Dataframes/useless_062621_potential.data" # Pinn_P acceleration (trad net and sph net)
    df_file = "Data/Dataframes/useless_062621_potential_v1.data" # with extra spherical coordinate transformeration

    df_file = "Data/Dataframes/useless_062721_v3.data" # phi and theta scaled by 180 and 360 independent of training data domain
    df_file = "Data/Dataframes/useless_062721_v4.data" # Scale the accelerations from 0,1; 4800 training data -> 8500 training data 
    df_file = "Data/Dataframes/useless_062721_v5.data" # More data did not help -- if anything made the phi component even worse --> Try not normalizing the cartesian vectors a priori
    # it's possible that the unit scaling followed by [-1,1] is causing issues. 

    df_file = "Data/Dataframes/useless_062821_v2.data" # N_train = 2500 (instead of 4800) and just cartesian (which has proper a_r values)
    df_file = "Data/Dataframes/useless_062821_v3.data" # Remove the concatenation of the [x_cart, x_sph] from the network (cleaner discontinuities at the poles); Change phi to atan instead of atan2 (didn't do anything); Normalize r from 0 to 1 instead of -1 to 1
    df_file = "Data/Dataframes/useless_062821_v4.data" # try pinn_a and see what the potential looks like
    df_file = "Data/Dataframes/useless_062821_v5.data" # change phi calc to acos instead of atan2
    df_file = "Data/Dataframes/useless_062821_v6.data" # Regenerate the data to esnure that potential is correct from [0,420] altitude so a more representative gradient is observed
    df_file = "Data/Dataframes/useless_062821_v7.data" # More data at altitude 
    df_file = "Data/Dataframes/useless_062821_v8.data" # make the spherical inputs (1/r)


    df_file = "Data/Dataframes/useless_062821_v2_4.data" 


    df_file = "Data/Dataframes/useless_063021_v1.data" # Cylindrical
    df_file = "Data/Dataframes/useless_063021_v2.data" # More Data in Spherical
    df_file = "Data/Dataframes/useless_063021_v3.data" # More Data in Cartesian -- doesn't do well
    df_file = "Data/Dataframes/useless_063021_v4.data" # Batch Normalization; 0.1 Dropout
    df_file = "Data/Dataframes/useless_063021_v5.data" # Sphere calc in init_scope()
    df_file = "Data/Dataframes/useless_070121_v1.data" # ResNet
    df_file = "Data/Dataframes/useless_070121_v2.data" # Eros -- accidentally earth
    df_file = "Data/Dataframes/useless_070121_v3.data" # Eros

    df_file = "Data/Dataframes/useless_070221_v4.data" # Outer training distribution
    df_file = "Data/Dataframes/useless_070621_v1.data" # Outer training distribution + 250 points within Brill, erroneous (accidentally made half the distribution from within sphere)
    df_file = "Data/Dataframes/useless_070621_v3.data" # Outer training distribution + 250 points within Brill, erroneous (250 out of 2750 within sphere) -- data wasnt added
    df_file = "Data/Dataframes/useless_070621_v4.data" # Outer training distribution + 250 points within Brill, corrected

    df_file = "Data/Dataframes/useless_070721_v1.data" # [0,10000]
    df_file = "Data/Dataframes/useless_070721_v2.data" # [5000,10000]


    # PLC, ALC, APLC
    df_file = "Data/Dataframes/useless_070721_v3.data" # [0,10000] PLC, ALC, APLC -- erroneous
    # df_file = "Data/Dataframes/useless_070721_v4.data" # [5000,10000]
    # df_file = "Data/Dataframes/useless_070721_v5.data" # [5000,10000] + [0,5000]

    threads = 1
    config = get_prototype_eros_config()
    # config = get_default_earth_config()
    #config = get_default_earth_pinn_config()
    #config = get_default_eros_pinn_config()
    #config = get_default_eros_config()

    hparams = {
        "N_dist": [50000],
        "N_train": [2500],
        "N_val" : [150],
        "epochs": [7500],

        'distribution' : [RandomAsteroidDist],
        'radius_min' : [0], # Make sure it isn't a float
        # 'radius_min' : [Eros().radius + 5000.0],
        'radius_max' : [Eros().radius + 10000.0],

        # 'extra_distribution' : [RandomAsteroidDist],
        # 'extra_radius_min' : [0],
        # 'extra_radius_max' : [Eros().radius + 5000.0],
        # 'extra_N_dist' : [500],
        # 'extra_N_train' : [250],
        # 'extra_N_val' : [150],

        #"radius_max" : [Earth().radius+50000.0], # Use a tighter radius
        "decay_rate_epoch": [2500],
        "decay_epoch_0": [5000],
        "decay_rate": [0.5],
        "learning_rate": [0.001],
        "batch_size": [131072 // 2],
        "activation": ["gelu"],
        "initializer": ["glorot_uniform"],
        'x_transformer' : [UniformScaler(feature_range=(-1,1))],
        'u_transformer' : [UniformScaler(feature_range=(-1,1))],
        'a_transformer' : [UniformScaler(feature_range=(-1,1))],
        "PINN_constraint_fcn": ["pinn_plc"],#lc", "pinn_plc", "pinn_aplc"],#"pinn_ap"],
        
        "scale_by": ["non_dim"],#"u"],
        "mixed_precision": [False],
        'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
        "num_units": [20],
        "schedule_type" : ['exp_decay'],
        "beta" : [0.9],
        "deg_removed" : [-1],
        #'dropout': [0.1],
        #'batch_norm' :[True],
        #'basis': ['spherical'],
        #'augmentation':['periodic'],
        "network_type": ['sph_traditional'],#["sph_traditional"],
        "input_layer": ["cart_and_sph"],
        #"input_layer": ["none"],
        'sph_in_graph': [True],
        "override" : [False]
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
    from GravNN.Networks.Data import get_preprocessed_data, configure_dataset, compute_normalization_layer_constants
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Networks import load_network
    from GravNN.Networks.utils import load_hparams_to_config, configure_optimizer
    from GravNN.Networks.Schedules import get_schedule

    tf = configure_tensorflow()
    mixed_precision = set_mixed_precision() if config_original['mixed_precision'][0] else None
    np.random.seed(1234)
    tf.random.set_seed(0)
    # tf.config.run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    # Standardize Configuration
    config = copy.deepcopy(config_original)
    config = load_hparams_to_config(hparams, config)

    check_config_combos(config)
    config = format_config_combos(config)
    print(config)

    # Get data, network, optimizer, and generate model
    train_data, val_data, transformers = get_preprocessed_data(config)
    compute_normalization_layer_constants(config)
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
    model.config["a_bar_transformer"][0] = transformers["a_bar"]


    model.save(df_file=None)

    import matplotlib.pyplot as plt
    plt.plot(history.history['val_loss'][1000:])
    plt.show()
    # Appends the model config to a perscribed df
    return model.config


if __name__ == "__main__":
    main()
