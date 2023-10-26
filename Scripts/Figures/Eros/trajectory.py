import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.Analysis.TrajectoryExperiment import TestModel, TrajectoryExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    generate_heterogeneous_model,
)
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.TrajectoryVisualizer import TrajectoryVisualizer


def main():
    planet = Eros()
    obj_file = planet.obj_200k
    traj_exp_file = "Data/Experiments/trajectory_hetero_200k_v2.data"
    init_state = np.array(
        [
            -10800.002,
            15273.506,
            10800.00,
            -2.383735,
            -3.371111,
            2.3837354,
        ],
    )

    true_model = generate_heterogeneous_model(planet, obj_file)
    test_poly_model = Polyhedral(planet, obj_file)

    df = pd.read_pickle("Data/Dataframes/heterogeneous_symmetric_080523.data")
    # df = pd.read_pickle("Data/Dataframes/heterogeneous_asymmetric_080523.data")
    model_id = df.id.values[-1]
    config, test_pinn_model = load_config_and_model(df, model_id)

    if os.path.exists(traj_exp_file):
        with open(traj_exp_file, "rb") as f:
            experiment = pickle.load(f)
    else:
        poly_test = TestModel(test_poly_model, "Poly", "r")
        pinn_test = TestModel(test_pinn_model, "PINN", "g")
        experiment = TrajectoryExperiment(
            true_model,
            [poly_test, pinn_test],
            initial_state=init_state,
            period=1 * 24 * 3600,  # 24 * 3600,
            pbar=True,
            tol=1e-8,
        )
        experiment.run()

        experiment.test_models[-1].pop("model")
        with open(traj_exp_file, "wb") as f:
            pickle.dump(experiment, f)

    vis = TrajectoryVisualizer(experiment, obj_file=obj_file)
    vis.plot()
    plt.figure(1)
    vis.save(plt.gcf(), "Eros/speed_and_error.pdf")
    fig = plt.figure(2)
    vis.save(fig, "Eros/trajectory.pdf")

    plt.show()


if __name__ == "__main__":
    main()
