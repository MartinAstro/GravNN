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

def regress_nn_model(config):
    from GravNN.Networks.utils import configure_tensorflow, configure_optimizer, set_mixed_precision, check_config_combos, populate_config_objects
    tf = configure_tensorflow()
    tf.keras.backend.clear_session()
    from GravNN.Networks.Networks import load_network
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.Callbacks import SimpleCallback
    import time
    populate_config_objects({}, config)
    check_config_combos(config)
    np.random.seed(config['seed'][0])
    time.sleep(np.random.randint(0, 10))

    if config['PINN_constraint_fcn'][0].__name__ == "pinn_A":
        config.update({
            "x_transformer": [MinMaxScaler(feature_range=(-1, 1))],
            "u_transformer": [UniformScaler(feature_range=(-1, 1))],
            "a_transformer": [UniformScaler(feature_range=(-1, 1))],
        })

    train_data, val_data, transformers = get_preprocessed_data(config)
    if config['network_type'][0].__name__ == "SphericalPinesTraditionalNet":
        compute_input_layer_normalization_constants(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    optimizer = configure_optimizer(config, mixed_precision)
    model = PINNGravityModel(config)
    model.compile(optimizer=optimizer, loss="mse")
    
    # Train network
    callback = SimpleCallback()
    schedule = get_schedule(config)
    if config.get("early_stop", [False])[0]:
        early_stop = tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss', min_delta=config.get("min_delta", [1E-6])[0], patience=1000, verbose=1,
                                            mode='auto', baseline=None, restore_best_weights=True
                                        )
        callback_list = [callback, schedule, early_stop]
    else:
        callback_list = [callback, schedule]

    history = model.fit(
        dataset,
        epochs=config["epochs"][0],
        verbose=0,
        validation_data=val_dataset,
        callbacks=callback_list
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

def regression_9500_v1():
    """As close to original configuration as first draft"""

    config = get_default_earth_config()
    config['N_train'] = [9500]
    config['N_val'] = [5000]
    config['scale_by'] = ["a"]
   
    config['schedule_type'] = ['plateau'] # TODO: Try exponential
    config["patience"] = [500]
    config["decay_rate"] = [0.9]
    config["min_delta"] = [1E-6]
    config["min_lr"] = [1E-6]   

    # config['schedule_type'] = ['exp_decay'] # TODO: Try exponential
    # config["decay_rate"] = [0.5]
    # config["decay_rate_epoch"] = [25000]
    # config["decay_epoch_0"] = [25000]

    config['batch_size'] = [40000]
    config['lr'] = [0.005]
    config['activation'] = ['gelu']

    num_models = 1
    num_units_list = [10, 20, 30]
    noise_list =  [0.0, 0.2]
    
    df_regressed = "Data/Dataframes/Regression/Earth_ML_models_regression_9500_v1.data"
    nn_file_name = "Data/Dataframes/Regression/Earth_NN_regression_9500_v1.data"
    pinn_file_name = "Data/Dataframes/Regression/Earth_PINN_regression_9500_v1.data"
    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle(nn_file_name)
    pinn_df.to_pickle(pinn_file_name)

def regression_9500_v2():
    """Changing to Pines architecture to see if performance can improve in best case. (No noise, 1 network)
    
    Got close with PINN, but still worse than SH. Possibly would improve without early stopping
    """

    config = get_default_earth_config()
    
    config['N_train'] = [9500]
    config['N_val'] = [500]
    config['scale_by'] = ["a"]
    config['early_stop'] = [True]
   
    config['schedule_type'] = ['exp_decay'] # TODO: Try exponential
    config["decay_rate"] = [0.5]
    config["decay_rate_epoch"] = [25000]
    config["decay_epoch_0"] = [25000]

    config['batch_size'] = [40000]
    config['lr'] = [0.005]
    config['activation'] = ['gelu']
    config["network_type"] = ["sph_pines_traditional"]
    config["custom_input_layer"] = [None]
    config["ref_radius"] = [Earth().radius]
    config["skip_normalization"] = [False]

    config.update({
                "x_transformer": [UniformScaler(feature_range=(-1, 1))],
                "u_transformer": [UniformScaler(feature_range=(-1, 1))],
                "a_transformer": [UniformScaler(feature_range=(-1, 1))],
                "scale_by": ["non_dim"],
                "dummy_transformer": [DummyScaler()],
                })
    num_models = 1#10
    num_units_list = [20]
    noise_list =  [0.0]
    
    df_regressed = "Data/Dataframes/Regression/Earth_ML_models_regression_9500_v2.data"
    nn_file_name = "Data/Dataframes/Regression/Earth_NN_regression_9500_v2.data"
    pinn_file_name = "Data/Dataframes/Regression/Earth_PINN_regression_9500_v2.data"
    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle(nn_file_name)
    pinn_df.to_pickle(pinn_file_name)



def regression_18000_v1():
    """Use less data"""

    config = get_default_earth_config()
    config['N_train'] = [18000]
    config['N_val'] = [500]
    config['scale_by'] = ["a"]
   
    config['schedule_type'] = ['exp_decay'] # TODO: Try exponential
    config["decay_rate"] = [0.5]
    config["decay_rate_epoch"] = [25000]
    config["decay_epoch_0"] = [25000]

    config['batch_size'] = [40000]
    config['lr'] = [0.005]
    config['activation'] = ['gelu']

    num_models = 1
    num_units_list = [20]
    noise_list =  [0.0]
    
    df_regressed = "Data/Dataframes/Regression/Earth_ML_models_regression_18000_v1.data"
    nn_file_name = "Data/Dataframes/Regression/Earth_NN_regression_18000_v1.data"
    pinn_file_name = "Data/Dataframes/Regression/Earth_PINN_regression_18000_v1.data"
    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle(nn_file_name)
    pinn_df.to_pickle(pinn_file_name)

def regression_18000_v2():
    """Use More Data and change the decay epoch to be much later (75000 instead of 25000)"""

    config = get_default_earth_config()
    config['N_train'] = [18000]
    config['N_val'] = [500]
    config['scale_by'] = ["a"]
   
    config['schedule_type'] = ['exp_decay'] # TODO: Try exponential
    config["decay_rate"] = [0.5]
    config["decay_rate_epoch"] = [25000]
    config["decay_epoch_0"] = [75000]

    config['batch_size'] = [40000]
    config['learning_rate'] = [0.001]
    config['activation'] = ['gelu']

    num_models = 1
    num_units_list = [20]
    noise_list =  [0.0]
    
    df_regressed = "Data/Dataframes/Regression/Earth_ML_models_regression_18000_v2.data"
    nn_file_name = "Data/Dataframes/Regression/Earth_NN_regression_18000_v2.data"
    pinn_file_name = "Data/Dataframes/Regression/Earth_PINN_regression_18000_v2.data"
    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle(nn_file_name)
    pinn_df.to_pickle(pinn_file_name)


def regression_9500_v3():
    """Use less data"""

    config = get_default_earth_config()
    config['N_train'] = [9500]
    config['N_val'] = [500]
    config['scale_by'] = ["a"]
   
    config['schedule_type'] = ['exp_decay'] # TODO: Try exponential
    config["decay_rate"] = [0.5]
    config["decay_rate_epoch"] = [25000]
    config["decay_epoch_0"] = [25000]

    config['batch_size'] = [40000]
    config['learning_rate'] = [0.001]
    config['activation'] = ['tanh']

    num_models = 1
    num_units_list = [20]
    noise_list =  [0.0]
    
    df_regressed = "Data/Dataframes/Regression/Earth_ML_models_regression_9500_v3.data"
    nn_file_name = "Data/Dataframes/Regression/Earth_NN_regression_9500_v3.data"
    pinn_file_name = "Data/Dataframes/Regression/Earth_PINN_regression_9500_v3.data"
    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle(nn_file_name)
    pinn_df.to_pickle(pinn_file_name)

def regression_9500_v4():
    """Use less data"""

    config = get_default_earth_config()
    config['N_train'] = [9500]
    config['N_val'] = [500]
    config['scale_by'] = ["a"]
   
    config['schedule_type'] = ['exp_decay'] # TODO: Try exponential
    config["decay_rate"] = [0.5]
    config["decay_rate_epoch"] = [25000]
    config["decay_epoch_0"] = [25000]

    config['batch_size'] = [40000]
    config['learning_rate'] = [0.001]
    config['activation'] = ['gelu']

    num_models = 1
    num_units_list = [20]
    noise_list =  [0.0]
    
    df_regressed = "Data/Dataframes/Regression/Earth_ML_models_regression_9500_v4.data"
    nn_file_name = "Data/Dataframes/Regression/Earth_NN_regression_9500_v4.data"
    pinn_file_name = "Data/Dataframes/Regression/Earth_PINN_regression_9500_v4.data"
    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle(nn_file_name)
    pinn_df.to_pickle(pinn_file_name)

def regression_5000_v1():
    """Use less data"""

    config = get_default_earth_config()
    config['N_train'] = [5000]
    config['N_val'] = [500]
    config['scale_by'] = ["a"]
   
    config['schedule_type'] = ['exp_decay'] # TODO: Try exponential
    config["decay_rate"] = [0.5]
    config["decay_rate_epoch"] = [25000]
    config["decay_epoch_0"] = [25000]

    config['batch_size'] = [40000]
    config['lr'] = [0.005]
    config['activation'] = ['tanh']

    num_models = 1
    num_units_list = [20]
    noise_list =  [0.0]
    
    df_regressed = "Data/Dataframes/Regression/Earth_ML_models_regression_5000_v1.data"
    nn_file_name = "Data/Dataframes/Regression/Earth_NN_regression_5000_v1.data"
    pinn_file_name = "Data/Dataframes/Regression/Earth_PINN_regression_5000_v1.data"
    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle(nn_file_name)
    pinn_df.to_pickle(pinn_file_name)


def regression_5000_v2():
    """Use less data"""

    config = get_default_earth_config()
    config['N_train'] = [5000]
    config['N_val'] = [500]
    config['scale_by'] = ["a"]
   
    config['schedule_type'] = ['exp_decay'] # TODO: Try exponential
    config["decay_rate"] = [0.5]
    config["decay_rate_epoch"] = [25000]
    config["decay_epoch_0"] = [25000]

    config['batch_size'] = [40000]
    config['lr'] = [0.005]
    config['activation'] = ['tanh']

    num_models = 3
    num_units_list = [10, 20]
    noise_list =  [0.0, 0.2]
    
    df_regressed = "Data/Dataframes/Regression/Earth_ML_models_regression_5000_v2.data"
    nn_file_name = "Data/Dataframes/Regression/Earth_NN_regression_5000_v2.data"
    pinn_file_name = "Data/Dataframes/Regression/Earth_PINN_regression_5000_v2.data"
    pool = mp.Pool(6)
    args = generate_mp_args(num_units_list, num_models, noise_list, config)
    nn_df, pinn_df = regress_networks_models_mp(args, pool, df_regressed)    

    nn_df.to_pickle(nn_file_name)
    pinn_df.to_pickle(pinn_file_name)


if __name__ == "__main__":
    # regression_9500_v1()
    # regression_9500_v2()
    
    # regression_5000_v1() #P Never run
    # regression_18000_v1()
    # regression_18000_v2()

    # regression_9500_v3() # TanH -- Best yet
    # regression_9500_v4() # Gelu
    # regression_5000_v1()
    regression_5000_v2()



