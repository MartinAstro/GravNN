import os
import sys

import matplotlib.pyplot as plt

import GravNN
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_symmetric_data
from GravNN.Networks.Configs import *
from GravNN.Networks.Data import DataSet
from GravNN.Networks.Layers import *
from GravNN.Networks.Model import PINNGravityModel
from GravNN.Networks.Saver import ModelSaver
from GravNN.Networks.utils import configure_tensorflow, populate_config_objects
from GravNN.Regression.utils import append_data, preprocess_data
from GravNN.Support.ProgressBar import ProgressBar
from GravNN.Trajectories.utils import (
    generate_near_hopper_trajectories,
    generate_near_orbit_trajectories,
)

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def regress_nn(
    max_traj,
    model,
    trajectories,
    hopper_trajectories,
    include_hoppers,
    df_file,
    acc_noise,
    pos_noise,
):
    planet = model.config["planet"][0]

    x_train = []
    y_train = []

    pbar = ProgressBar(len(trajectories), enable=True)

    plt.figure()
    total_samples = 0
    hopper_samples = 0

    # For each orbit, train the network
    # for k in range(len(trajectories) - 4, len(trajectories)):
    for k in range(max_traj):
        trajectory = trajectories[k]
        x, a, u = get_hetero_poly_symmetric_data(
            trajectory,
            planet.obj_8k,
            point_mass_removed=[False],
        )
        x_errored, a_errored = preprocess_data(x, a, acc_noise, pos_noise)
        x_train, y_train = append_data(x_train, y_train, x_errored, a_errored)

        # Don't include the hoppers in the sample count because those samples are used
        # to compute the times in the plotting routines.
        if include_hoppers:
            hop_trajectory = hopper_trajectories[k]
            x_hop, a_hop, u_hop = get_hetero_poly_symmetric_data(
                hop_trajectory,
                planet.obj_8k,
                remove_point_mass=[False],
            )
            hopper_samples += len(x_hop)
            x_errored, a_errored = preprocess_data(x_hop, a_hop, acc_noise, pos_noise)
            x_train, y_train = append_data(x_train, y_train, x_errored, a_errored)
        pbar.update(k)

    total_samples = len(x_train) - hopper_samples

    data = DataSet()
    data.config = {"dtype": ["float32"]}

    x_preprocessor = model.config["x_transformer"][0]
    a_preprocessor = model.config["a_transformer"][0]

    x_train_processed = x_preprocessor.transform(x_train)
    y_train_processed = a_preprocessor.transform(y_train)

    data.from_raw_data(x_train_processed, y_train_processed)
    history = model.train(data)

    plt.plot(history.history["loss"], label=str(k))

    regression_dict = {
        "planet": [planet.__class__.__name__],
        "trajectory": [trajectory.__class__.__name__],
        "hoppers": [include_hoppers],
        "samples": [total_samples],
        "acc_noise": [acc_noise],
        "pos_noise": [pos_noise],
        "seed": [0],
    }
    model.config.update(regression_dict)
    saver = ModelSaver(model, history)
    saver.save(df_file=df_file)

    plt.legend()
    plt.show()


def get_args(idx):
    args = [
        [False, 0.0, 0, "pinn_a"],
        [False, 0.1, 1, "pinn_a"],
        [True, 0.0, 0, "pinn_a"],
        [True, 0.1, 1, "pinn_a"],
        [False, 0.0, 0, "pinn_al"],
        [False, 0.1, 1, "pinn_al"],
        [True, 0.0, 0, "pinn_al"],
        [True, 0.1, 1, "pinn_al"],
    ]
    return args[idx]


def main():
    # Access the values of the arguments
    sampling_interval = 10 * 60
    trajectories = generate_near_orbit_trajectories(sampling_inteval=sampling_interval)
    hopper_trajectories = generate_near_hopper_trajectories(
        sampling_inteval=sampling_interval,
    )

    # override args with HPC idx
    idx = int(sys.argv[1])
    # idx = 23
    args = get_args(idx // 24)
    # 0  - 24 will be 0
    # 24 - 48 will be 1

    hoppers = args[0]
    acc_noise = args[1]
    pos_noise = args[2]
    loss = args[3]

    max_traj = int(idx % 24)
    print("VALUES")
    print(max_traj, idx // 24)

    gravnn_dir = os.path.abspath(os.path.dirname(GravNN.__file__))
    model_specifier = f"{acc_noise}_{pos_noise}_{hoppers}_{loss}.data"
    df_file = f"{gravnn_dir}/../Data/Dataframes/eros_regression_{model_specifier}"

    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [100],
        "N_train": [90],
        "N_val": [10],
        "num_units": [20],
        "radius_min": [0.0],
        "radius_max": [Eros().radius * 10],
        "loss_fcns": [["percent", "rms"]],
        "lr_anneal": [False],
        "learning_rate": [0.0005],
        "dropout": [0.0],
        # "batch_size": [2**18],
        "batch_size": [2**18],
        "epochs": [15000],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": [loss],
        # "ref_radius_analytic": [10],
        "tanh_r": [10],
    }
    config.update(hparams)
    config = populate_config_objects(config)
    configure_tensorflow(config)

    # Need to initialize data to configure the transformer scales
    DataSet(config)
    model = PINNGravityModel(config)

    regress_nn(
        max_traj,
        model,
        trajectories,
        hopper_trajectories,
        hoppers,
        df_file,
        acc_noise,
        pos_noise,
    )


if __name__ == "__main__":
    main()
