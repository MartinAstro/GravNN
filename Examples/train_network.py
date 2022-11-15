import multiprocessing as mp
from GravNN.Networks.Callbacks import LossComponentsCallback
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():

    # dataframe where the network configuration info will be saved
    df_file = "Data/Dataframes/example_training.data" 

    # a default set of hyperparameters / configuration details for PINN
    config = get_default_eros_config()

    # hyperparameters which overwrite defaults
    hparams = PINN_IV()
    hparams.update(ReduceLrOnPlateauConfig())
    hparams.update({
        "grav_file" : [Eros().obj_8k],
        "N_dist": [5000],
        "N_train": [4500],
        "N_val": [500],
        "epochs" : [5000],
        "batch_size" : [4096],
        "network_type" : ["sph_pines_traditional"],
        "PINN_constraint_fcn": ["pinn_alc"],
        "dtype" : ['float64']
    })


    threads = 1
    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def run(config):
    # Tensorflow dependent functions must be defined inside of 
    # run function for thread-safe behavior.
    import numpy as np
    import pandas as pd
    np.random.seed(config['seed'][0])
    from GravNN.Networks.utils import configure_tensorflow
    from GravNN.Networks.Callbacks import SimpleCallback
    from GravNN.Networks.Data import get_preprocessed_data, configure_dataset, compute_input_layer_normalization_constants
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.utils import populate_config_objects, configure_optimizer
    from GravNN.Networks.Schedules import get_schedule

    tf, mixed_precision = configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    print(config)

    # Get data, network, optimizer, and generate model
    train_data, val_data, transformers = get_preprocessed_data(config)
    compute_input_layer_normalization_constants(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    optimizer = configure_optimizer(config, mixed_precision)
    model = PINNGravityModel(config)
    model.compile(optimizer=optimizer, loss="mse")
    
    # Train network
    callback = SimpleCallback(config['batch_size'][0])
    loss_callback = LossComponentsCallback(config['batch_size'][0])
    schedule = get_schedule(config)

    history = model.fit(
        dataset,
        epochs=config["epochs"][0],
        verbose=0,
        validation_data=val_dataset,
        callbacks=[callback, loss_callback, schedule],
    )
    history.history["time_delta"] = callback.time_delta
    model.history = history

    # Saving history object to a dataframe for use in plots
    hist_df = pd.DataFrame(history.history) 

    hist_csv_file = 'history_percent.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    model.save(df_file=None, history=history, transformers=transformers)

    # Appends the model config to a perscribed df
    return model.config


if __name__ == "__main__":
    main()
