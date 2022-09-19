import multiprocessing as mp
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
from Hyperparam_inits import hyperparams_eros
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():

    threads = 1

    # df_file = "Data/Dataframes/eros_pinn_III_040822.data" 

    df_file = "Data/Dataframes/eros_pinn_II_III.data" 
    df_file = "Data/Dataframes/eros_pinn_II_III_warm_start.data" 
    df_file = "Data/Dataframes/eros_comp_planes.data" 
    df_file = "Data/Dataframes/eros_BVP_PINN_III.data" 
    df_file = "Data/Dataframes/eros_point_mass.data" 
    df_file = "Data/Dataframes/eros_point_mass_v2.data" 
    df_file = "Data/Dataframes/eros_point_mass_v3.data" 
    # df_file = "Data/Dataframes/eros_point_mass_alc.data" 
    config = get_default_eros_config()
    # config = get_default_earth_config()

    from GravNN.GravityModels.PointMass import get_pm_data
    from GravNN.GravityModels.Polyhedral import get_poly_data
    from GravNN.CelestialBodies.Asteroids import Eros

    config.update(PINN_III())
    # config.update(PINN_II())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        # "grav_file": [Eros().obj_200k],
        "grav_file": [Eros().obj_8k],
        "N_dist": [100000],
        "N_train": [9850],
        "N_val": [1500],
        "radius_max" : [Eros().radius*10],
    
        "learning_rate": [0.002],
        "num_units": [40],
        # "PINN_constraint_fcn" : ['pinn_alc'],
        "PINN_constraint_fcn" : ['pinn_a'],
        "patience" : [50],
        'override': [False],
        'ref_radius': [Eros().radius],
        "batch_size" : [2**13],
        "epochs" : [1000],
        # "init_file" : [2459774.886435185] #  Best one yet, batch size of 2**14 for 15000 epochs, max error 3% average at 0.5%. Goal now is to reduce average error 
        "remove_point_mass" : [False],
        "gravity_data_fcn" : [get_pm_data],
        # "init_file" : [2459811.101574074] #PM PINN-A
        # "init_file" : [2459816.249097222], #PM PINN-A
        "jit_compile" : [False],
        "dtype" : ['float64']
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
    from GravNN.Networks.utils import populate_config_objects, configure_optimizer
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
    config = populate_config_objects(config)

    check_config_combos(config)
    print(config)

    # Get data, network, optimizer, and generate model
    train_data, val_data, transformers = get_preprocessed_data(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    optimizer = configure_optimizer(config, mixed_precision)
    model = CustomModel(config)
    model.compile(optimizer=optimizer, loss="mse")
    
    # Train network
    callback = SimpleCallback(config['batch_size'][0], print_interval=1)
    schedule = get_schedule(config)

    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=500)

    history = model.fit(
        dataset,
        epochs=config["epochs"][0],
        verbose=0,
        validation_data=val_dataset,
        callbacks=[callback, schedule],#, tensorboard_callback],
        use_multiprocessing=True
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
    return model.config


if __name__ == "__main__":
    main()
