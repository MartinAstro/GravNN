import multiprocessing as mp
import os
from pprint import pprint

from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Trajectories.SurfaceDist import SurfaceDist

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    threads = 1

    df_file = "Data/Dataframes/heterogenous_eros_041823.data"

    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [1000000],
        "N_train": [990000],
        "N_val": [10000],
        "radius_max": [Eros().radius * 10],
        "num_units": [20],
        "loss_fcns": [["percent", "rms"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.0001],
        "dropout": [0.0],
        "batch_size": [2**18],
        "epochs": [5000],
        "acc_noise": [0.0],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "trainable": [False],
        "fuse_models": [True],
        "tanh_r": [10],
        "tanh_k": [1e3],
        "grav_file": [Eros().obj_200k],
        "gravity_data_fcn": [get_hetero_poly_data],
        "augment_data_config": [
            {
                "distribution": [SurfaceDist],
                "N_dist": [200700],
                "N_train": [199700],
                "N_val": [1000],
            },
        ],
    }
    args = configure_run_args(config, hparams)

    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)

    # from Scripts.Figures.Earth.nn_brillouin_map import main as main_maps
    # main_maps()
    # from GravNN.Visualization.HistoryVisualizer import main as main_history
    # main_history()


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

    # config["PINN_constraint_fcn"] = ["pinn_al"]
    # # tf.keras.backend.clear_session()
    # data = DataSet(config)
    # model_AL = PINNGravityModel(config)
    # model_AL.network = model.network
    # model_AL.compile(model.optimizer)
    # model_AL.train(data)

    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
