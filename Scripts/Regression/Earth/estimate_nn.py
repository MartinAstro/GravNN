import numpy as np
import pandas as pd
import os
import copy
import multiprocessing as mp

from tensorflow.keras import mixed_precision
from GravNN.Networks.Schedules import get_schedule

from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data

from GravNN.Support.StateObject import StateObject
from GravNN.Regression.Regression import Regression
from GravNN.Regression.utils import save, format_coefficients
from GravNN.Networks.Data import compute_input_layer_normalization_constants, configure_dataset, training_validation_split, get_preprocessed_data
from GravNN.Networks.Configs import *
np.random.seed(1234)

def regress_nn_model(config):
    from GravNN.Networks.utils import configure_tensorflow, configure_optimizer, set_mixed_precision, check_config_combos, load_hparams_to_config
    tf = configure_tensorflow()
    tf.keras.backend.clear_session()
    from GravNN.Networks.Networks import load_network
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Callbacks import SimpleCallback
    load_hparams_to_config({}, config)
    check_config_combos(config)

    train_data, val_data, transformers = get_preprocessed_data(config)
    #compute_input_layer_normalization_constants(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    optimizer = configure_optimizer(config, mixed_precision)
    network = load_network(config)
    model = CustomModel(config, network)
    model.compile(optimizer=optimizer, loss="mse")
    
    # Train network
    callback = SimpleCallback()
    schedule = get_schedule(config)
    early_stop = tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss', min_delta=config['min_delta'][0], patience=1000, verbose=1,
                                        mode='auto', baseline=None, restore_best_weights=True
                                    )

    history = model.fit(
        dataset,
        epochs=config["epochs"][0],
        verbose=0,
        validation_data=val_dataset,
        callbacks=[callback, schedule, early_stop],
    )
    history.history["time_delta"] = callback.time_delta
    model.history = history

    # Save network and config information
    model.config["time_delta"] = [callback.time_delta]
    model.config["x_transformer"][0] = transformers["x"]
    model.config["u_transformer"][0] = transformers["u"]
    model.config["a_transformer"][0] = transformers["a"]
    #model.config["a_bar_transformer"][0] = transformers["a_bar"]

    model.save(df_file=None)


    model.config["PINN_constraint_fcn"] = [
        model.config["PINN_constraint_fcn"][0]
    ]  # Can't have multiple args in each list

    return model.config


def regress_networks_models_mp(args, pool, df_regressed):
    nn_df = pd.DataFrame(index=pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]], names=['noise', 'nodes', 'id']), columns=['model_identifier'])
    pinn_df = pd.DataFrame(index=pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]], names=['noise', 'nodes', 'id']), columns=['model_identifier'])

    results = pool.starmap_async(regress_nn_model, args)
    configs = results.get()
    pinn_counter = 0
    nn_counter = 0
    df_all = pd.DataFrame().from_dict(configs[0]).set_index('timetag')
    for i in range(len(configs)):
        config = configs[i]
        noise = config['acc_noise'][0]
        num_units = config['num_units'][0]
        if config['PINN_constraint_fcn'][0].__name__ == "no_pinn":
            nn_df.loc[(noise, num_units, nn_counter)] = config['id'][0]
            nn_counter += 1
        else:
            pinn_df.loc[(noise, num_units, pinn_counter)] = config['id'][0]
            pinn_counter += 1

        if i == 0 : continue
        df = pd.DataFrame().from_dict(config).set_index('timetag')
        df_all = df_all.append(df)
    df_all.to_pickle(df_regressed)
 
    return nn_df, pinn_df

def generate_mp_args(num_units_list, num_models, noise_list, config):
    args = []
    for num_units in num_units_list:
        for idx in range(num_models):
            for noise in noise_list:
                for pinn_constraint in ['no_pinn', 'pinn_A']:
                    sub_config= copy.deepcopy(config)
                    sub_config['acc_noise'] = [noise]
                    sub_config['seed'] = [idx]
                    sub_config['num_units'] = [num_units]
                    sub_config['PINN_constraint_fcn'] = [pinn_constraint]
                    args.append((sub_config,))
    return args

def main():
    """Multiprocessed version of generate models. Trains multiple networks 
    simultaneously to speed up regression.
    """
    config = get_default_earth_config()
    config['N_train'] = [9500]
    config['N_val'] = [500]
    config['scale_by'] = ["a"]

    config['schedule_type'] = ['plateau']
    config["patience"] = [500]
    config["decay_rate"] = [0.9]
    config["min_delta"] = [1E-6]
    config["min_lr"] = [1E-6]   
    # config['epochs'] = [100]

    num_models = 10
    num_units_list = [10, 20, 30]
    noise_list =  [0.0, 0.2]
    
    df_regressed = "Data/Dataframes/regressed_models_4.data"

    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle("Data/Dataframes/regress_nn_4.data")
    pinn_df.to_pickle("Data/Dataframes/regress_pinn_4.data")

if __name__ == "__main__":
    main()

