import os
from pprint import pprint

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def get_hparams():
    hparams = {
        "N_dist": [8192],
        "radius_max": [Eros().radius * 50],
        "N_train": [1000],
        "N_val": [4096],
        "num_units": [16],
        "layers": [[3, 1, 1, 1, 1, 1, 1, 3]],
        "learning_rate": [2**-8],
        "batch_size": [2**11],
        "epochs": [2**13],
    }
    return hparams


def PINN_III_mod_I_Train():
    df_file = "Data/Dataframes/pinn_loss_mod_I.data"
    config = get_default_eros_config()
    config.update(PINN_II())
    config.update(ReduceLrOnPlateauConfig())
    hparams = get_hparams()
    hparams.update(
        {
            "preprocessing": [["pines", "r_inv"]],
            "final_layer_initializer": ["zeros"],
            "scale_by": ["non_dim_v3"],
        },
    )
    args = configure_run_args(config, hparams)
    configs = [run(*args[0])]
    save_training(df_file, configs)


def PINN_III_mod_II_Train():
    df_file = "Data/Dataframes/pinn_loss_mod_II.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    hparams = get_hparams()
    hparams.update(
        {
            "preprocessing": [["pines", "r_inv"]],
            "final_layer_initializer": ["zeros"],
            "scale_by": ["non_dim_v3"],
        },
    )
    hparams.update({"loss_fcns": [["percent", "rms"]]})
    args = configure_run_args(config, hparams)
    configs = [run(*args[0])]
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
    model.config["val_loss"] = history.history["val_percent_mean"][-1]

    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    PINN_III_mod_I_Train()
    PINN_III_mod_II_Train()
