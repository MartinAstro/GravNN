import time

import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Trajectories.RandomDist import RandomDist


def main():
    # Load a trained gravity PINN
    df = pd.read_pickle("Data/Dataframes/example_training.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(df, model_id)

    # Generate sample testing data randomly distributed around the asteroid
    planet = Eros()
    trajectory = RandomDist(
        planet,
        radius_bounds=[0.0, planet.radius * 3],
        points=500,
        obj_file=planet.obj_8k,
    )

    # Define the analytic gravity model used to define ground truth
    gravity_model = Polyhedral(planet, planet.obj_8k)

    # Time how long it takes to compute the acceleration using the analytic model
    analytic_start_time = time.time()
    true_accelerations = gravity_model.compute_acceleration(trajectory.positions)
    analytic_delta = time.time() - analytic_start_time
    print("Analytic Time: %.2f" % (analytic_delta))

    # Time PINN gravity model
    PINN_start_time = time.time()
    pred_accelerations = model.compute_acceleration(trajectory.positions)
    PINN_delta = time.time() - PINN_start_time
    print("PINN Time: %.2f" % (PINN_delta))

    # Compute the error of the PINN model
    diff = true_accelerations - pred_accelerations
    percent_error = (
        np.linalg.norm(diff, axis=1) / np.linalg.norm(true_accelerations, axis=1) * 100
    )
    avg_percent_error = np.average(percent_error)
    std_percent_error = np.std(percent_error)
    print("Average Percent Error: %.2f ± %.2f" % (avg_percent_error, std_percent_error))
    print("Speedup: %.2fx" % (analytic_delta / PINN_delta))


if __name__ == "__main__":
    main()
