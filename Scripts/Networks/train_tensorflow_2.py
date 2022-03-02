import multiprocessing as mp
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
from Hyperparam_inits import hyperparams_eros
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():

    threads = 1
    df_file = "Data/Dataframes/eros_pinn_III.data" 


    config = get_default_eros_config()

    from GravNN.GravityModels.PointMass import get_pm_data
    from GravNN.GravityModels.Polyhedral import get_poly_data
    from GravNN.CelestialBodies.Asteroids import Eros

    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        # "gravity_data_fcn" : [get_pm_data],
        "gravity_data_fcn" : [get_poly_data],
        "grav_file": [Eros().obj_8k],
        "N_dist": [25000],
        "N_train": [20000],
        "N_val": [5000],
        "num_units": [40],
        "radius_max" : [Eros().radius*10],
        "network_type" : ["sph_pines_traditional_v2"],
        "scale_by" : ['non_dim_radius'],
        "PINN_constraint_fcn" : ['pinn_a'],
        "patience" : [500],
        "init_file" : [2459641.262199074] #PM
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
    )
    from GravNN.Networks.Callbacks import SimpleCallback
    from GravNN.Networks.Data import get_preprocessed_data, configure_dataset
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
    tf.config.run_functions_eagerly(False)
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
    model.compile(optimizer=optimizer, loss="mse")
    
    # Train network
    callback = SimpleCallback()
    schedule = get_schedule(config)
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=model.directory+"/network",
    #     monitor='val_loss',
    #     # save_freq='epochs',#int(np.ceil(config['N_train'][0]/config['batch_size'][0]))*500,
    #     mode='auto',
    #     period=500,
    #     initial_value_threshold=0.00000005,
    #     save_best_only=True)

    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=500)

    checkpoint_callback = None
    history = model.fit(
        dataset,
        epochs=config["epochs"][0],
        verbose=0,
        validation_data=val_dataset,
        callbacks=[callback, schedule, tensorboard_callback],
    )
    history.history["time_delta"] = callback.time_delta
    model.history = history

    # Save network and config information
    model.config["time_delta"] = [callback.time_delta]
    model.config["x_transformer"][0] = transformers["x"]
    model.config["u_transformer"][0] = transformers["u"]
    model.config["a_transformer"][0] = transformers["a"]
    model.config["a_bar_transformer"][0] = transformers["a_bar"]

    model.save(df_file=None, checkpoint_callback=checkpoint_callback)

    # import matplotlib.pyplot as plt
    # plt.plot(history.history['val_loss'][1000:])
    # plt.show()
    # Appends the model config to a perscribed df
    return model.config


if __name__ == "__main__":
    main()
