import argparse
import os
import sys

import pandas as pd

import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import (
    append_data,
    format_coefficients,
    preprocess_data,
    save,
)
from GravNN.Support.ProgressBar import ProgressBar
from GravNN.Trajectories.utils import (
    generate_near_hopper_trajectories,
    generate_near_orbit_trajectories,
)


def BLLS_SH(
    regress_deg,
    remove_deg,
    trajectories,
    hopper_trajectories,
    include_hoppers=False,
    acc_noise=0.0,
    pos_noise=0.0,
):
    # This has the true SH coef of Eros
    # -- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.998.2986&rep=rep1&type=pdf
    planet = Eros()

    N = regress_deg
    M = remove_deg

    # Initialize the regressor
    regressor = BLLS(N, planet, M)

    # Record time history of regressor
    x_hat_hist = []

    x_train = []
    y_train = []

    total_samples = 0
    hopper_samples = 0
    pbar = ProgressBar(len(trajectories), enable=True)

    df = pd.DataFrame()
    # for k in range(len(trajectories) - 4, len(trajectories)):
    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        x, a, u = get_poly_data(trajectory, planet.obj_8k)
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

        ###########
        # REGRESS #
        ###########
        x_hat = regressor.update(x_train, y_train)
        x_hat_hist.append(x_hat)

        ###############
        # Save to CSV #
        ###############
        C_lm, S_lm = format_coefficients(x_hat, N, M)
        model_name = (
            f"model_{regress_deg}_{k}_{acc_noise}_{pos_noise}_{include_hoppers}.csv"
        )
        directory = f"{GravNN.__path__[0]}/Files/GravityModels/Regressed/Eros/SH/BLLS"
        file_name = f"{directory}/{model_name}"
        save(file_name, planet, C_lm, S_lm)

        ###################
        # Save to dataframe
        ###################
        regress_dict = {
            "samples": total_samples,
            "C_lm": C_lm,
            "S_lm": S_lm,
            "N": N,
            "M": M,
            "hoppers": include_hoppers,
            "file_name": file_name,
            "acc_noise": acc_noise,
            "pos_noise": pos_noise,
        }
        df_k = pd.DataFrame.from_dict([regress_dict])
        df = df.append(df_k, ignore_index=True)

        pbar.update(k)

    gravnn_dir = os.path.dirname(GravNN.__file__) + "/../"
    model_name = f"{regress_deg}_{acc_noise}_{pos_noise}_{include_hoppers}.data"
    df.to_pickle(
        f"{gravnn_dir}/Data/Dataframes/eros_sh_regression_{model_name}",
    )

    return C_lm, S_lm


def get_args(idx):
    args = [
        [4, False, 0.0, 0],
        [4, False, 0.1, 0],
        [4, False, 0.0, 1],
        [4, False, 0.1, 1],
        [4, True, 0.0, 0],
        [4, True, 0.1, 0],
        [4, True, 0.0, 1],
        [4, True, 0.1, 1],
        [16, False, 0.0, 0],
        [16, False, 0.1, 0],
        [16, False, 0.0, 1],
        [16, False, 0.1, 1],
        [16, True, 0.0, 0],
        [16, True, 0.1, 0],
        [16, True, 0.0, 1],
        [16, True, 0.1, 1],
    ]
    return args[idx]


def main():
    parser = argparse.ArgumentParser(description="Regress SH from NEAR data.")

    # Add arguments with default values
    parser.add_argument("--deg", type=int, default=4, help="Max degree to regress")
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

    # Parse the command-line arguments
    args = parser.parse_args()
    max_deg = args.deg
    hoppers = args.hoppers
    acc_noise = args.acc_noise
    pos_noise = args.pos_noise

    # Access the values of the arguments
    sampling_interval = 10 * 60
    trajectories = generate_near_orbit_trajectories(sampling_inteval=sampling_interval)
    hopper_trajectories = generate_near_hopper_trajectories(
        sampling_inteval=sampling_interval,
    )

    # override command line args with HPC idx
    idx = int(sys.argv[1])
    args = get_args(idx)

    max_deg = args[0]
    hoppers = args[1]
    acc_noise = args[2]
    pos_noise = args[3]

    remove_deg = -1

    BLLS_SH(
        max_deg,
        remove_deg,
        trajectories,
        hopper_trajectories,
        hoppers,
        acc_noise,
        pos_noise,
    )


if __name__ == "__main__":
    main()
