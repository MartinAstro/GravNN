import numpy as np
import pandas as pd
import tensorflow as tf
from utils import compute_stats

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Networks.Model import count_nonzero_params, load_config_and_model
from GravNN.Support.StateObject import StateObject
from GravNN.Trajectories import FibonacciDist

np.random.seed(1234)
tf.random.set_seed(0)


def compute_nn_regression_statistics(
    nn_df,
    nn_df_stats_file,
    model_df,
    trajectory,
    grid_true,
):
    x = trajectory.positions
    df_all = pd.DataFrame()
    model_df = pd.read_pickle(model_df)
    for i in range(len(nn_df)):
        row = nn_df.iloc[i]
        model_id = row["model_identifier"]  # TODO: Change this name
        config, model = load_config_and_model(model_df, model_id)
        a_est = model.compute_acceleration(x)
        grid_pred = StateObject(trajectory=trajectory, accelerations=a_est)

        entries = compute_stats(grid_true, grid_pred)
        entries.update({"params": [count_nonzero_params(model)]})

        df = pd.DataFrame().from_dict(entries)  # .set_index(nn_df.index[i])
        df_all = df_all.append(df)

    df_all.index = nn_df.index
    nn_df = nn_df.join(df_all)
    nn_df.to_pickle(nn_df_stats_file)


def main():
    """Given the regressed spherical harmonic and neural network models
    (generate_models_mp.py), compute the associated error of these
    regressed representations and store in new regress_stats dataframe.
    """

    planet = Earth()
    sh_file = planet.sh_file
    max_deg = 1000

    trajectory = FibonacciDist(planet, planet.radius, 250000)
    model_df = "Data/Dataframes/Earth_ML_models_regression_9500_v1.data"
    nn_file = "Data/Dataframes/Regression/Earth_NN_regression_9500_v1"
    pinn_file = "Data/Dataframes/Regression/Earth_PINN_regression_9500_v1"

    # Pines
    model_df = "Data/Dataframes/Regression/Earth_ML_models_regression_9500_v2.data"
    nn_file = "Data/Dataframes/Regression/Earth_NN_regression_9500_v2"
    pinn_file = "Data/Dataframes/Regression/Earth_PINN_regression_9500_v2"

    # More Data
    model_df = "Data/Dataframes/Regression/Earth_ML_models_regression_18000_v2.data"
    nn_file = "Data/Dataframes/Regression/Earth_NN_regression_18000_v2"
    pinn_file = "Data/Dataframes/Regression/Earth_PINN_regression_18000_v2"

    model_df = "Data/Dataframes/Regression/Earth_ML_models_regression_9500_v3.data"
    nn_file = "Data/Dataframes/Regression/Earth_NN_regression_9500_v3"
    pinn_file = "Data/Dataframes/Regression/Earth_PINN_regression_9500_v3"

    model_df = "Data/Dataframes/Regression/Earth_ML_models_regression_9500_v4.data"
    nn_file = "Data/Dataframes/Regression/Earth_NN_regression_9500_v4"
    pinn_file = "Data/Dataframes/Regression/Earth_PINN_regression_9500_v4"

    model_df = "Data/Dataframes/Regression/Earth_ML_models_regression_5000_v1.data"
    nn_file = "Data/Dataframes/Regression/Earth_NN_regression_5000_v1"
    pinn_file = "Data/Dataframes/Regression/Earth_PINN_regression_5000_v1"

    model_df = "Data/Dataframes/Regression/Earth_ML_models_regression_5000_v2.data"
    nn_file = "Data/Dataframes/Regression/Earth_NN_regression_5000_v2"
    pinn_file = "Data/Dataframes/Regression/Earth_PINN_regression_5000_v2"

    nn_df = pd.read_pickle(nn_file + ".data")
    pinn_df = pd.read_pickle(pinn_file + ".data")
    nn_df_stats_file = nn_file + "_stats.data"
    pinn_df_stats_file = pinn_file + "_stats.data"

    x, a, u = get_sh_data(
        trajectory,
        sh_file,
        max_deg=max_deg,
        deg_removed=2,
        override=False,
    )
    grid_true = StateObject(trajectory=trajectory, accelerations=a)

    compute_nn_regression_statistics(
        nn_df,
        nn_df_stats_file,
        model_df,
        trajectory,
        grid_true,
    )
    compute_nn_regression_statistics(
        pinn_df,
        pinn_df_stats_file,
        model_df,
        trajectory,
        grid_true,
    )


if __name__ == "__main__":
    main()
