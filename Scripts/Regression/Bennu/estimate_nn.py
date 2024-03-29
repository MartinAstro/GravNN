import os

import matplotlib.pyplot as plt
import numpy as np

from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Networks.Configs import *
from GravNN.Networks.Data import DataSet
from GravNN.Networks.Model import PINNGravityModel
from GravNN.Networks.Saver import ModelSaver
from GravNN.Support.ProgressBar import ProgressBar
from GravNN.Trajectories.utils import generate_orex_orbit_trajectories

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def append_data(x_train, y_train, x, y):
    try:
        for i in range(len(x)):
            x_train.append(x[i])
            y_train.append(y[i])
    except Exception:
        x_train = np.concatenate((x_train, x))
        y_train = np.concatenate((y_train, y))
    return x_train, y_train


def regress_nn(model, trajectories, include_hoppers=False):
    planet = model.config["planet"][0]

    x_train = []
    y_train = []

    pbar = ProgressBar(len(trajectories), enable=True)

    plt.figure()
    total_samples = 0
    hopper_samples = 0

    # For each orbit, train the network
    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        x, a, u = get_poly_data(trajectory, planet.obj_200k, remove_point_mass=[False])
        x_train, y_train = append_data(x_train, y_train, x, a)

        # Don't include the hoppers in the sample count because those samples are used
        # to compute the times in the plotting routines.
        if include_hoppers:
            hop_trajectory = hopper_trajectories[k]
            x_hop, a_hop, u_hop = get_poly_data(
                hop_trajectory,
                planet.obj_200k,
                remove_point_mass=[False],
            )
            hopper_samples += len(x_hop)
            x_train, y_train = append_data(x_train, y_train, x_hop, a_hop)

        total_samples = len(x_train) - hopper_samples

        data = DataSet(x_train, y_train, shuffle=True)
        history = model.train(data)
        saver = ModelSaver(model, history)
        saver.save(df_file=None)

        plt.plot(history.history["loss"], label=str(k))

        planet_name = planet.__class__.__name__
        trajectory_name = trajectory.__class__.__name__
        file_name = "%s/%s/%s/%d.data" % (
            planet_name,
            trajectory_name,
            str(include_hoppers),
            total_samples,
        )
        directory = os.path.curdir + "/GravNN/Files/GravityModels/Regressed/"
        os.makedirs(os.path.dirname(directory + file_name), exist_ok=True)
        save_dir = directory + file_name
        saver = ModelSaver(model, history, save_dir)
        saver.save(df_file=None)

        pbar.update(k)

    plt.legend()
    # plt.show()


def main():
    include_hoppers = True
    sampling_interval = 10 * 60
    trajectories = generate_orex_orbit_trajectories(sampling_inteval=sampling_interval)

    config = get_default_bennu_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [50000],
        "N_train": [45000],
        "N_val": [5000],
        "num_units": [20],
        "loss_fcns": [["percent", "rms"]],
        "jit_compile": [False],
        "lr_anneal": [True],
        "eager": [False],
        "learning_rate": [0.0001],
        "dropout": [0.0],
        "batch_size": [4500],
        "epochs": [50000],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_al"],
        # "tanh_r" : [3],
        # "tanh_k": [5],
    }
    config.update(hparams)
    model = PINNGravityModel(config)

    regress_nn(model, trajectories, include_hoppers=include_hoppers)


if __name__ == "__main__":
    main()
