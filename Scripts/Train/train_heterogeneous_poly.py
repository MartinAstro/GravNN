import os
import time
from pprint import pprint

import StatOD

from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_symmetric_data
from GravNN.Networks.Configs import *

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    df_file = statOD_dir + "Data/Dataframes/eros_hetero_poly_072023.data"
    df_file = statOD_dir + "Data/Dataframes/eros_hetero_poly_072423.data"
    df_file = statOD_dir + "Data/Dataframes/eros_hetero_poly_072523.data"
    df_file = statOD_dir + "Data/Dataframes/eros_hetero_poly_072523_10.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [6000],
        "N_train": [5000],
        "N_val": [500],
        "num_units": [10],
        "loss_fcns": [["percent", "rms"]],
        "decay_rate": [0.5],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.001],
        "batch_size": [2**11],
        "epochs": [5000],
        "radius_max": [Eros().radius * 10],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "gravity_data_fcn": [get_hetero_poly_symmetric_data],
        "fuse_models": [True],
        # "eager": [True],
        # "grav_file": [Eros().obj_200k],
    }
    config.update(hparams)
    run(config, df_file)

    # config, model = load_config_and_model(df_file, config['id'][0])
    # plot_planes(model, config)


def run(config, df_file=None):
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
    start_time = time.time()
    history = model.train(data)
    model.config["dt"] = [time.time() - start_time]
    saver = ModelSaver(model, history=history)
    saver.save(df_file=df_file)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
