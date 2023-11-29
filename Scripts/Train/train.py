import os
from pprint import pprint

from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    df_file = "Data/Dataframes/eros_072223.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [50000],
        "N_train": [2**15],
        "N_val": [4096],
        "num_units": [16],
        "layers": [[3, 1, 1, 1, 1, 1, 1, 3]],
        "loss_fcns": [["percent", "mse"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.0001],
        "dropout": [0.0],
        # "batch_size": [2**11],
        "batch_size": [2**16],
        # "batch_size": [4500],
        "epochs": [200],
        "acc_noise": [0.0],
        "gravity_data_fcn": [get_hetero_poly_data],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "tanh_k": [0.1],
        "early_stop": [True],
    }
    args = configure_run_args(config, hparams)

    configs = [run(*args[0])]
    # with mp.Pool(threads) as pool:
    #     results = pool.starmap_async(run, args)
    #     configs = results.get()
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
    model.config["val_loss"] = history.history["val_percent_mean"][-1]

    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
