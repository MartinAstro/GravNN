import argparse
import copy
import os

import matplotlib.pyplot as plt

import GravNN
from GravNN.GravityModels.Polyhedral import get_poly_data
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
    model,
    trajectories,
    hopper_trajectories,
    include_hoppers,
    df_file,
    acc_noise,
    pos_noise,
    new_model=False,
    config=None,
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
    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        x, a, u = get_poly_data(trajectory, planet.obj_8k, remove_point_mass=[False])
        x_errored, a_errored = preprocess_data(x, a, acc_noise, pos_noise)
        x_train, y_train = append_data(x_train, y_train, x_errored, a_errored)

        # Don't include the hoppers in the sample count because those samples are used
        # to compute the times in the plotting routines.
        if include_hoppers:
            hop_trajectory = hopper_trajectories[k]
            x_hop, a_hop, u_hop = get_poly_data(
                hop_trajectory,
                planet.obj_8k,
                remove_point_mass=[False],
            )
            hopper_samples += len(x_hop)
            x_errored, a_errored = preprocess_data(x_hop, a_hop, acc_noise, pos_noise)
            x_train, y_train = append_data(x_train, y_train, x_errored, a_errored)

        total_samples = len(x_train) - hopper_samples

        data = DataSet()
        data.config = {"dtype": ["float32"]}

        if new_model:
            new_config = None
            if new_config is None:
                new_config = copy.deepcopy(config)
            model = PINNGravityModel(new_config)

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

        pbar.update(k)

    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Regress SH from NEAR data.")

    # Add arguments with default values
    parser.add_argument(
        "--hoppers",
        type=bool,
        default=False,
        help="Include hoppers in the regression",
    )
    parser.add_argument(
        "--acc_noise",
        type=float,
        default=0.0,
        help="Acceleration error ratio [-].",
    )
    parser.add_argument(
        "--pos_noise",
        type=float,
        default=0.0,
        help="position error [m].",
    )
    parser.add_argument(
        "--fuse_models",
        type=bool,
        default=True,
        help="Fuse analytic model into PINN",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    sampling_interval = 10 * 60
    trajectories = generate_near_orbit_trajectories(sampling_inteval=sampling_interval)
    hopper_trajectories = generate_near_hopper_trajectories(
        sampling_inteval=sampling_interval,
    )
    hoppers = args.hoppers
    acc_noise = args.acc_noise
    pos_noise = args.pos_noise
    fuse_models = args.fuse_models

    gravnn_dir = os.path.abspath(os.path.dirname(GravNN.__file__))
    model_specifier = f"{hoppers}_{acc_noise}_{pos_noise}_{fuse_models}.data"
    df_file = f"{gravnn_dir}/../Data/Dataframes/eros_regression_{model_specifier}"

    # remove the df_file if it exists
    if os.path.exists(df_file):
        os.remove(df_file)

    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [100],
        "N_train": [90],
        "N_val": [10],
        "num_units": [20],
        "radius_min": [0.0],
        "radius_max": [Eros().radius * 3],
        "loss_fcns": [["percent", "rms"]],
        "lr_anneal": [False],
        "learning_rate": [0.0001 * 10],
        "dropout": [0.0],
        "batch_size": [2**18],
        "epochs": [2500],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        # "ref_radius_analytic": [10],
        "fuse_models": [fuse_models],
    }
    config.update(hparams)
    config = populate_config_objects(config)
    configure_tensorflow(config)

    # Need to initialize data to configure the transformer scales
    DataSet(config)
    model = PINNGravityModel(config)

    regress_nn(
        model,
        trajectories,
        hopper_trajectories,
        hoppers,
        df_file,
        acc_noise,
        pos_noise,
        new_model=True,
        config=config,
    )


if __name__ == "__main__":
    main()
