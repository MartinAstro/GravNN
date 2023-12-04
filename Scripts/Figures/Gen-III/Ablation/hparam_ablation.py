from Trainer import HPCTrainer

from GravNN.Networks.Configs import *
from GravNN.Networks.utils import permutate_dict


def default_config():
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    config.update(
        {
            "N_dist": [100000],
            "N_train": [2**15],  # 32,768
            "N_val": [2**12],  # 4096
            "radius_max": [Eros().radius * 3],
            "batch_size": [2**11],
            "learning_rate": [2**-8],
            "epochs": [2**13],  # 8192
            "num_units": [32],
            "layers": [[3, 1, 1, 1, 1, 1, 1, 1]],
            "obj_file": [Eros().obj_200k],
        },
    )
    return config


def width_depth(df_file):
    config = default_config()
    hparams = {
        "num_units": [8, 16, 32, 64],
        "layers": [
            [3, 1, 1, 3],
            [3, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        ],
    }
    run(config, hparams, df_file)


def batch_learning(df_file):
    config = default_config()
    hparams = {
        "batch_size": [2**i for i in range(7, 15 + 1, 2)],
        "learning_rate": [2**i for i in range(-14, -8 + 1, 2)],
    }
    run(config, hparams, df_file)


def run(config, hparams, df_file):
    hparams_permutations = permutate_dict(hparams)
    hpc_trainer = HPCTrainer(config, hparams_permutations, df_file)
    idx = int(sys.argv[1])
    hpc_trainer.run(idx)


if __name__ == "__main__":
    # Run this to determine optimal hparams
    # Run data_ablation afterwords.
    width_depth("Data/Dataframes/ablation_width_depth_120323.data")
    batch_learning("Data/Dataframes/ablation_batch_learning_120323.data")
