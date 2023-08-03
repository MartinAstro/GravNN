import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.Analysis.TrajectoryExperiment import TrajectoryExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import HeterogeneousPoly
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.TrajectoryVisualizer import TrajectoryVisualizer


def main():
    planet = Eros()
    obj_file = planet.obj_200k
    traj_exp_file = "Data/Experiments/trajectory_hetero_200k.data"
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

    true_model = HeterogeneousPoly(planet, obj_file)

    mass_1 = Eros()
    mass_1.mu = planet.mu / 10
    r_offset_1 = [planet.radius / 3, 0, 0]

    mass_2 = Eros()
    mass_2.mu = -planet.mu / 10
    r_offset_2 = [-planet.radius / 3, 0, 0]

    point_mass_1 = PointMass(mass_1)
    point_mass_2 = PointMass(mass_2)

    true_model.add_point_mass(point_mass_1, r_offset_1)
    true_model.add_point_mass(point_mass_2, r_offset_2)

    test_poly_model = Polyhedral(planet, obj_file)

    df = pd.read_pickle("Data/Dataframes/heterogenous_eros_041823.data")
    model_id = df.id.values[-1]
    config, test_pinn_model = load_config_and_model(df, model_id)

    if os.path.exists(traj_exp_file):
        with open(traj_exp_file, "rb") as f:
            experiment = pickle.load(f)
    else:
        experiment = TrajectoryExperiment(
            true_model,
            initial_state=init_state,
            period=1 * 24 * 3600,  # 24 * 3600,
        )
        experiment.add_test_model(test_poly_model, "Poly", "r")
        experiment.add_test_model(test_pinn_model, "PINN", "g")
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
