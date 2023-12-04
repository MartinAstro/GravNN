import sys

from data_ablation import large_network, optimized_config, small_network
from Trainer import HPCTrainer

from GravNN.Networks.Configs import *
from GravNN.Networks.utils import permutate_dict


def noise_loss_hparams():
    hparams = {
        "batch_size": [2**15],
        "acc_noise": [0.1],
        "N_train": [2**i for i in range(9, 18)],
        "PINN_constraint_fcn": ["pinn_a", "pinn_al"],
        "early_stop": [True],
    }
    return hparams


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
    noise_loss_small("Data/Dataframes/ablation_noise_loss_small_120323.data")
    noise_loss_large("Data/Dataframes/ablation_noise_loss_large_120323.data")
