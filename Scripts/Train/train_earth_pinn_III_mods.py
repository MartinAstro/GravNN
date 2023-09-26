import multiprocessing as mp
import os
from pprint import pprint

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    threads = 2
    config = get_default_earth_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    df_file = "Data/Dataframes/earth_loss_fcn_experiment.data"
    hparams = {
        "N_dist": [10000],
        "N_train": [5000],
        "N_val": [1000],
        "num_units": [20],
        "radius_max": [Earth().radius * 15],
        "loss_fcns": [["rms"], ["percent"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.001],
        "dropout": [0.0],
        "batch_size": [2**18],
        "epochs": [5000],
        "acc_noise": [0.0],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        # "final_layer_initializer": ["glorot_uniform"],
        "scale_nn_potential": [False],
        "trainable": [False],
        "fuse_models": [False],
        "deg_removed": [2],
        "enforce_bc": [False],
    }
    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)

    df_file = "Data/Dataframes/earth_scaled_potential_experiment.data"
    hparams.update(
        {
            "scale_nn_potential": [False, True],
            "loss_fcns": [["percent"]],
        },
    )
    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)

    df_file = "Data/Dataframes/earth_boundary_conditions_experiment.data"
    hparams.update(
        {
            "scale_nn_potential": [True],
            "loss_fcns": [["percent"]],
            "enforce_bc": [True, False],
        },
    )
    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def run(config):
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.Saver import ModelSaver
    from GravNN.Networks.utils import configure_tensorflow, populate_config_objects

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    pprint(config)

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)

    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
