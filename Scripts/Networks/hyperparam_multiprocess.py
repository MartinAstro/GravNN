from GravNN.Trajectories.EphemerisDist import EphemerisDist
import multiprocessing as mp
from numba.core.types.abstract import Dummy
import pandas as pd
from script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
from GravNN.Preprocessors.UniformScaler import UniformScaler
from GravNN.Trajectories.utils import single_near_trajectory
import time
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():
    df_file = "Data/Dataframes/useless_062821_v8.data" # make the spherical inputs (1/r)

    df_file = "Data/Dataframes/useless_070221_v4.data" # Outer training distribution
    df_file = "Data/Dataframes/useless_070621_v1.data" # Outer training distribution + 250 points within Brill, erroneous (accidentally made half the distribution from within sphere)
    df_file = "Data/Dataframes/useless_070621_v3.data" # Outer training distribution + 250 points within Brill, erroneous (250 out of 2750 within sphere) -- data wasnt added
    df_file = "Data/Dataframes/useless_070621_v4.data" # Outer training distribution + 250 points within Brill, corrected

    df_file = "Data/Dataframes/transformers_wo_constraints.data" # Transformer Performance on r and r_bar

    df_file = "Data/Dataframes/traditional_w_constraints_annealing.data" 

    df_file = "Data/Dataframes/small_data_pinn_constraints_wo_annealing.data"
    df_file = "Data/Dataframes/small_data_pinn_constraints_wo_annealing_lr_plateau.data"
    df_file = "Data/Dataframes/small_data_pinn_constraints_w_annealing_lr_plateau.data"

    df_file = "Data/Dataframes/medium_data_pinn_constraints_wo_annealing_lr_plateau.data"
    df_file = "Data/Dataframes/v_tiny_data_pinn_constraints_wo_annealing_lr_plateau.data"
    df_file = "Data/Dataframes/v_v_tiny_data_pinn_constraints_wo_annealing_lr_plateau.data"

    df_file = "Data/Dataframes/no_pinn.data"

    df_file = "Data/Dataframes/transformer_wo_annealing.data"

    df_file = "Data/Dataframes/bennu_traditional_wo_annealing.data"

    df_file = "Data/Dataframes/bennu_official_w_noise.data"
    df_file = "Data/Dataframes/bennu_official_w_noise_2.data"

    df_file = "Data/Dataframes/eros_official_w_noise.data"
    df_file = "Data/Dataframes/eros_official_w_noise_trad.data"
    df_file = "Data/Dataframes/eros_official_w_noise_transformer.data"
    df_file = "Data/Dataframes/eros_official_w_noise_transformer_dropout.data"

    df_file = "Data/Dataframes/eros_official_w_noise_5_seeds.data"


    df_file = "Data/Dataframes/eros_official_noise_annealing.data"

    threads = 4
    config = get_default_eros_config()
    # config = get_default_bennu_config()
    #config = get_prototype_toutatis_config()

    #config = get_default_earth_config()
    #config = get_default_earth_pinn_config()
    #config = get_default_eros_pinn_config()
    #config = get_default_eros_config()

    hparams = {
        # "grav_file": [Bennu().stl_200k],
        "grav_file" : [Eros().obj_200k],
        # "N_dist": [50000],

        
        "N_train": [2500, 2500//2, 2500//4],
        "PINN_constraint_fcn": ["pinn_ap", 'pinn_aplc', 'pinn_alc'],
        "acc_noise" : [0.0, 0.1, 0.2], # percent
        'seed' : [0],#,1,2,3,4],
        "network_type" :['sph_pines_traditional'],
        # "network_type" : ['sph_pines_transformer'],
        # 'transformer_units' : [20], 
        "lr_anneal" : [True],

        # "PINN_constraint_fcn": [ "pinn_a"],
        # "acc_noise" : [0.0], # percent


        #"ref_radius" : [Eros().radius*3/2],
        "N_val" : [1500],
        "epochs": [7500],
        "dropout" : [0.0],

        "learning_rate": [0.001*2],
        "batch_size": [131072 // 2],

        # "PINN_constraint_fcn": ["pinn_aplc"],


        "num_units": [20],
        "beta" : [0.9],

        # "schedule_type" : ['exp_decay'],
        # "decay_rate_epoch": [2500],
        # "decay_epoch_0": [500],
        # "decay_rate": [0.5],

        'schedule_type' : ['plateau'],
        "patience" : [250],
        "decay_rate" : [0.9],
        "min_delta" : [0.0001],
        "min_lr" : [0.0001],
        #'batch_norm' :[True],

        "remove_point_mass" : [False], # remove point mass from polyhedral model
        "override" : [False]
    }

    # traj_params = {
    #     "N_train": [40000],
    #     "N_val": [5000],
    #     "custom_data_fcn" : [single_near_trajectory]
    #     # "distribution" : [EphemerisDist],
    #     # "source" : ["NEAR"],
    #     # "target" : ["EROS"],
    #     # "frame" : ["EROS_FIXED"],
    #     # "start_time" : ["Feb 24, 2000"],
    #     # "end_time" : ["Feb 06, 2001"],
    #     # "sampling_interval" : [10*60],
    #     # "celestial_body" : [Eros()],
    # }
    # hparams.update(traj_params)

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
    np.random.seed(hparams['seed'])
    tf.random.set_seed(hparams['seed'])
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
