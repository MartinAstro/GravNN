from Trainer import PoolTrainer

from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training


def width_depth_ablation():
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    hparams = {
        "N_dist": [1000000],
        "N_train": [50000],
        "N_val": [5000],
        "epochs": [8192],
        "radius_max": [Eros().radius * 3],
        "num_units": [8, 16, 32, 64],
        "layers": [
            [3, 1, 1, 3],
            [3, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        ],
        "batch_size": [2**11, 2**12, 2**13, 2**14],
        "learning_rate": [2**-7, 2**-6, 2**-5, 2**-4],
    }

    # Multiprocess
    df_file = "Data/Dataframes/pinn_III_ablation_width_depth.data"
    pool_trainer = PoolTrainer(config, hparams, df_file).run(threads=1)
    run_data = pool_trainer.run()
    save_training(df_file, run_data)


def data_epochs_ablation():
    df_file = "Data/Dataframes/pinn_III_ablation_data_epochs.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    hparams = {
        "N_dist": [1000000],
        "N_train": [
            2**9,
            2**10,
            2**11,
            2**12,
            2**13,
            2**14,
            2**15,
            2**16,
        ],
        "epochs": [
            2**9,
            2**10,
            2**11,
            2**12,
            2**13,
            2**14,
            2**15,
            2**16,
        ],
        "N_val": [5000],
        "radius_max": [Eros().radius * 3],
        "num_units": [8],
        "layers": [
            [3, 1, 1, 3],
        ],
        "batch_size": [2**11],
        "learning_rate": [2**-6],
    }
    hparams_copy = hparams.copy()

    # Multiprocess
    pool_trainer = PoolTrainer(config, hparams, df_file).run(threads=1)
    run_data = pool_trainer.run()
    save_training(df_file, run_data)

    # Big Network
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    hparams_copy.update(
        {
            "num_units": [32],
            "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
        },
    )

    pool_trainer = PoolTrainer(config, hparams, df_file).run(threads=1)
    run_data = pool_trainer.run()
    save_training(df_file, run_data)


def noise_loss_ablation():
    df_file = "Data/Dataframes/pinn_III_ablation_noise_loss.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    hparams = {
        "N_dist": [1000000],
        "N_train": [
            2**9,
            2**10,
            2**11,
            2**12,
            2**13,
            2**14,
            2**15,
            2**16,
        ],
        "epochs": [2**13],
        "N_val": [5000],
        "radius_max": [Eros().radius * 3],
        "num_units": [8],
        "layers": [
            [3, 1, 1, 3],
        ],
        "PINN_constraint_fcn": ["pinn_a", "pinn_al"],
        "batch_size": [2**11],
        "learning_rate": [2**-6],
    }
    hparams_copy = hparams.copy()

    # Multiprocess
    pool_trainer = PoolTrainer(config, hparams, df_file).run(threads=1)
    run_data = pool_trainer.run()
    save_training(df_file, run_data)

    # Big Network
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())
    hparams_copy.update(
        {
            "num_units": [32],
            "layers": [[3, 1, 1, 1, 1, 1, 1, 1, 1, 3]],
        },
    )

    pool_trainer = PoolTrainer(config, hparams, df_file).run(threads=1)
    run_data = pool_trainer.run()
    save_training(df_file, run_data)


if __name__ == "__main__":
    width_depth_ablation()
    data_epochs_ablation()
    noise_loss_ablation()
