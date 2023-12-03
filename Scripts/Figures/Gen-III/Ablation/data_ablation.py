import sys

from hparam_ablation import default_config
from Trainer import HPCTrainer

from GravNN.Networks.Configs import *
from GravNN.Networks.utils import permutate_dict


def optimized_config():
    config = default_config()
    config.update(
        {
            # Use a medium size network
            "layers": [[3, 1, 1, 1, 1, 1, 1, 1]],
            "num_units": [32],
            # These are the best hparams
            "batch_size": [2**11],
            "learning_rate": [2**-10],
        },
    )
    return config


def noise_loss_hparams():
    hparams = {
        "batch_size": [2**15],
        "acc_noise": [0.1],
        "N_train": [2**i for i in range(9, 18)],
        "PINN_constraint_fcn": ["pinn_a", "pinn_al"],
        "early_stop": [True],
    }
    return hparams


def data_epoch_hparams():
    hparams = {
        "N_train": [2**i for i in range(9, 16, 2)],
        "epochs": [2**i for i in range(9, 16, 2)],
    }
    return hparams


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


def data_epochs_small(df_file):
    config = optimized_config()
    hparams = data_epoch_hparams()
    hparams.update(small_network())
    run(config, hparams, df_file)


def data_epochs_large(df_file):
    config = optimized_config()
    hparams = data_epoch_hparams()
    hparams.update(large_network())
    run(config, hparams, df_file)


def noise_loss_small(df_file):
    config = optimized_config()
    hparams = noise_loss_hparams()
    hparams.update(small_network())
    run(config, hparams, df_file)


def noise_loss_large(df_file):
    config = optimized_config()
    hparams = noise_loss_hparams()
    hparams.update(large_network())
    run(config, hparams, df_file)


def run(config, hparams, df_file):
    hparams_permutations = permutate_dict(hparams)
    hpc_trainer = HPCTrainer(config, hparams_permutations, df_file)
    idx = int(sys.argv[1])
    hpc_trainer.run(idx)


if __name__ == "__main__":
    # To be run after hparam_ablation.py and optimized hparams are selected
    data_epochs_small("Data/Dataframes/ablation_data_epochs_small.data")
    data_epochs_large("Data/Dataframes/ablation_data_epochs_large.data")

    noise_loss_small("Data/Dataframes/ablation_noise_loss_small.data")
    noise_loss_large("Data/Dataframes/ablation_noise_loss_large.data")
