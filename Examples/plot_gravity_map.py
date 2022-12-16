import matplotlib.pyplot as plt
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import DHGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Networks.Model import load_config_and_model


def main():
    # Load a PINN Model
    df = pd.read_pickle("Data/Dataframes/example_training.data")
    model_id = df["id"].values[0]
    config, model = load_config_and_model(model_id, df)

    # Build a grid of position data at a fix altitude that can be used to produce maps
    planet = Eros()
    trajectory = DHGridDist(planet, planet.radius, degree=20) # degree corresponds to density of map

    # define the gravity model to produce true accelerations
    # populate it with the trajectory to evaluate accelerations for
    # call load() to see if data has already been produced and cached
    grav_model = Polyhedral(planet, planet.obj_8k, trajectory=trajectory).load()

    # Gather true and predicted accelerations
    a_true = grav_model.accelerations
    a_pred = model.compute_acceleration(trajectory.positions)

    # Store the results in a grid object which allows for quick operations
    # and easy plotting
    grid_true = Grid(trajectory=trajectory, accelerations=a_true)
    grid_pred = Grid(trajectory=trajectory, accelerations=a_pred)

    # compute difference
    error_grid = (grid_pred - grid_true)

    # Instantiate a plotting class which produces maps from grid objects
    map_vis = MapVisualization()
    map_vis.tick_interval = [45,45] # interval for Lat/Long
    map_vis.newFig(fig_size=(6, 6))
    plt.subplot(3,1,1)
    map_vis.plot_grid(grid_true.total, vlim=None, label="True Accel.", new_fig=False)
    plt.subplot(3,1,2)
    map_vis.plot_grid(grid_pred.total, vlim=None, label="Predicted Accel.", new_fig=False)
    plt.subplot(3,1,3)
    map_vis.plot_grid(error_grid.total, vlim=None, label="Difference", new_fig=False)
    plt.show()

if __name__ == "__main__":
    main()