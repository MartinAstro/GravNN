import multiprocessing as mp
import os
from pprint import pprint

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def multiprocess_run(df_file, hparams, config):
    args = configure_run_args(config, hparams)
    with mp.Pool(4) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def main():
    PINN_I_config = get_default_eros_config()
    PINN_I_config.update(ReduceLrOnPlateauConfig())
    PINN_I_config.update(PINN_I())

    PINN_II_config = get_default_eros_config()
    PINN_II_config.update(ReduceLrOnPlateauConfig())
    PINN_II_config.update(PINN_II())

    PINN_III_config = get_default_eros_config()
    PINN_III_config.update(ReduceLrOnPlateauConfig())
    PINN_III_config.update(PINN_III())

    hparams = {
        "num_units": [10, 20, 40, 80],
        "epochs": [2**12, 2**13, 2**14, 2**15],
    }
    PINN_III_config.update(
        {
            "preprocessing": [["pines", "r_inv", "fourier"]],
            "PINN_constraint_fcn": ["pinn_a"],
            # "fourier_features": [5],
            "fourier_sigma": [1],
            # "shared_freq": [False],
            # "shared_offset": [False],
            # "trainable": [True],
            "base_2_init": [True],
            "fourier_features": [10],
            "trainable": [False],
            "shared_freq": [True],
            "shared_offset": [True],
            "tanh_k": [3],
            "tanh_r": [Earth().radius * 10],
        },
    )

    df_file = "Data/Dataframes/eros_PINN_I_032823.data"
    multiprocess_run(df_file, hparams, PINN_I_config)

    df_file = "Data/Dataframes/eros_PINN_II_032823.data"
    multiprocess_run(df_file, hparams, PINN_II_config)

    df_file = "Data/Dataframes/eros_PINN_III_032823.data"
    multiprocess_run(df_file, hparams, PINN_III_config)


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

    return model.config


if __name__ == "__main__":
    main()
