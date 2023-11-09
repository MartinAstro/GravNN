import sys

from Trainer import HPCTrainer

from GravNN.Networks.Configs import *


def default_config():
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    config.update(
        {
            # "N_dist": [150000],
            "N_dist": [150000],
            "N_train": [2**15],  # 32,768
            "N_val": [2**12],  # 4096
            "radius_max": [Eros().radius * 3],
            "batch_size": [2**13],
            "learning_rate": [2**-13],
            "epochs": [2**13],  # 8192
            # "eager": [True],
            # "jit_compile": [False],
        },
    )
    return config


def optimized_config():
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    config.update(
        {
            # "N_dist": [150000],
            "N_dist": [150000],
            "N_train": [2**15],  # 32,768
            "N_val": [2**12],  # 4096
            "radius_max": [Eros().radius * 3],
            "batch_size": [2**11],
            "learning_rate": [2**-10],
            "epochs": [2**13],  # 8192
            "layers": [[3, 1, 1, 1, 1, 1, 1, 1]],
            "num_units": [32],
            # "eager": [True],
            # "jit_compile": [False],
        },
    )
    return config


def small_network():
    # This has 480 parameters (8x6)
    hparams = {
        "num_units": [8],
        "layers": [[3, 1, 1, 3]],
    }
    return hparams


def large_network():
    # this has 8640 parameters (32x8)
    hparams = {
        "num_units": [64],
        "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
    }
    return hparams


def run(hparams, df_file):
    config = optimized_config()
    hpc_trainer = HPCTrainer(config, hparams, df_file)
    idx = int(sys.argv[1])
    hpc_trainer.run(idx)


def width_depth(df_file):
    hparams = {
        "num_units": [8, 16, 32, 64],
        "layers": [
            [3, 1, 1, 3],
            [3, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        ],
    }
    run(hparams, df_file)


def batch_learning(df_file):
    hparams = {
        "batch_size": [2**i for i in range(7, 15 + 1, 2)],
        "learning_rate": [2**i for i in range(-16, -8 + 1, 2)],
    }
    run(hparams, df_file)


def data_epochs_small(df_file):
    hparams = {
        "N_train": [2**i for i in range(9, 16, 2)],
        "epochs": [2**i for i in range(9, 16, 2)],
    }
    hparams.update(small_network())
    run(hparams, df_file)


def data_epochs_large(df_file):
    hparams = {
        "N_train": [2**i for i in range(9, 16, 2)],
        "epochs": [2**i for i in range(9, 16, 2)],
    }
    hparams.update(large_network())
    run(hparams, df_file)


def noise_loss_small(df_file):
    hparams = {
        "acc_noise": [0.1],
        "N_train": [2**i for i in range(9, 18)],
        "PINN_constraint_fcn": ["pinn_a", "pinn_al"],
        "early_stop": [True],
    }
    hparams.update(small_network())
    run(hparams, df_file)


def noise_loss_large(df_file):
    hparams = {
        "acc_noise": [0.1],
        "N_train": [2**i for i in range(9, 18)],
        "PINN_constraint_fcn": ["pinn_a", "pinn_al"],
        "early_stop": [True],
    }
    hparams.update(large_network())
    run(hparams, df_file)


if __name__ == "__main__":
    # width_depth("Data/Dataframes/ablation_width_depth.data")
    batch_learning("Data/Dataframes/ablation_batch_learning.data")

    data_epochs_small("Data/Dataframes/ablation_data_epochs_small.data")
    data_epochs_large("Data/Dataframes/ablation_data_epochs_large.data")

    noise_loss_small("Data/Dataframes/ablation_noise_loss_small.data")
    noise_loss_large("Data/Dataframes/ablation_noise_loss_large.data")
