import copy
import os
from pprint import pprint

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    default_hparams = {
        "N_dist": [50000],
        "N_train": [45000],
        "N_val": [5000],
        "num_units": [20],
        "loss_fcns": [["percent", "rms"]],
        "jit_compile": [False],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.0001],
        "dropout": [0.0],
        "batch_size": [4500],
        "epochs": [20000],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "trainable_tanh": [True],
        "acc_noise": [0.0],
        "tanh_r": [10],
        "tanh_k": [1e3],
    }

    PINN_A_hparams = default_hparams.copy()

    PINN_AL_hparams = default_hparams.copy()
    PINN_AL_hparams.update({"PINN_constraint_fcn": ["pinn_al"]})

    PINN_A_AL_hparams = default_hparams.copy()
    PINN_A_AL_hparams.update({"warm_start": [True]})
    PINN_A_AL_hparams.update({"epochs": [20000 // 2]})

    PINN_A_AL_anneal_hparams = default_hparams.copy()
    PINN_A_AL_anneal_hparams.update({"warm_start": [True]})
    PINN_A_AL_anneal_hparams.update({"epochs": [20000 // 2]})
    PINN_A_AL_anneal_hparams.update({"warm_lr_anneal": [True]})

    hparams = [
        PINN_A_hparams,
        PINN_AL_hparams,
        PINN_A_AL_hparams,
        PINN_A_AL_anneal_hparams,
    ]
    hparams_noise = copy.deepcopy(hparams)
    df_file = "Data/Dataframes/LR_Anneal_No_Noise_032423.data"
    configs = []
    for hparam in hparams:
        args = configure_run_args(config, hparam)
        config_output = run(*args[0])
        configs.append(config_output)
    save_training(df_file, configs)

    df_file = "Data/Dataframes/LR_Anneal_With_Noise_032623.data"
    configs = []
    for hparam in hparams_noise:
        hparam["acc_noise"] = [0.1]
        args = configure_run_args(config, hparam)
        config_output = run(*args[0])
        configs.append(config_output)
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

    if config.get("warm_start", [False])[0]:
        # change to AL
        config["PINN_constraint_fcn"] = ["pinn_al"]

        # optionally use LR annealing
        if config.get("warm_lr_anneal", [False])[0]:
            config["lr_anneal"] = [True]

        # tf.keras.backend.clear_session()
        data = DataSet(config)
        model_AL = PINNGravityModel(config)
        model_AL.network = model.network
        model_AL.compile(model.optimizer)
        model_AL.train(data)

    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
