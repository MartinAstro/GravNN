from Trainer import PoolTrainer

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training


def default_config():
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    config.update(
        {
            # "N_dist": [150000],
            "N_dist": [50000],
            "N_train": [45000],
            "N_val": [5000],
            "radius_max": [Eros().radius * 3],
            "batch_size": [2**13],
            "learning_rate": [2**-6],
            "epochs": [2**13],  # 8192
            # "eager": [True],
            # "jit_compile": [False],
        },
    )
    return config


def small_network():
    hparams = {
        "num_units": [8],
        "layers": [[3, 1, 1, 3]],
    }
    return hparams


def large_network():
    hparams = {
        "num_units": [32],
        "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
    }
    return hparams


def run(hparams, df_file):
    config = default_config()
    pool_trainer = PoolTrainer(config, hparams, df_file)
    run_data = pool_trainer.run(threads=1)
    save_training(df_file, run_data)


def width_depth(df_file):
    hparams = {
        "num_units": [8, 16, 32, 64],
        "layers": [
            [3, 1, 1, 3],
            [3, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        ],
    }
    run(hparams, df_file)


def batch_learning(df_file):
    hparams = {
        "num_units": [16],
        "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
        "batch_size": [2**i for i in range(7, 15 + 1, 2)],
        "learning_rate": [2**i for i in range(-16, -8 + 1, 2)],
    }
    run(hparams, df_file)


def data_epochs_small(df_file):
    hparams = {
        "N_train": [2**i for i in range(9, 17)],
        "epochs": [2**i for i in range(9, 17)],
    }
    hparams.update(small_network())
    run(hparams, df_file)


def data_epochs_large(df_file):
    hparams = {
        "N_train": [2**i for i in range(9, 17)],
        "epochs": [2**i for i in range(9, 17)],
    }
    hparams.update(large_network())
    run(hparams, df_file)


def noise_loss_small(df_file):
    hparams = {
        "acc_noise": [0.1],
        "N_train": [2**i for i in range(9, 17)],
        "PINN_constraint_fcn": ["pinn_a", "pinn_al"],
    }
    hparams.update(small_network())
    run(hparams, df_file)


def noise_loss_large(df_file):
    hparams = {
        "acc_noise": [0.1],
        "N_train": [2**i for i in range(9, 17)],
        "PINN_constraint_fcn": ["pinn_a", "pinn_al"],
    }
    hparams.update(large_network())
    run(hparams, df_file)


if __name__ == "__main__":
    width_depth("Data/Dataframes/ablation_width_depth.data")
    batch_learning("Data/Dataframes/ablation_batch_learning.data")

    # data_epochs_small("Data/Dataframes/ablation_data_epochs.data")
    # data_epochs_large("Data/Dataframes/ablation_data_epochs.data")

    # noise_loss_small("Data/Dataframes/ablation_noise_loss.data")
    # noise_loss_large("Data/Dataframes/ablation_noise_loss.data")
