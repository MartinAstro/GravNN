import os

from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapBase import MapBase
from GravNN.Visualization.VisualizationBase import VisualizationBase


class AsteroidSurfaceVisualizer(VisualizationBase):
    def __init__(self, PINN_model, trajectory, config, **kwargs):
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.PINN_model = PINN_model
        self.trajectory = trajectory
        self.config = config
        self.compute_accelerations()
        self.compute_grids()

    def compute_accelerations(self):
        self.pred_acc = self.PINN_model.compute_acceleration(self.trajectory.positions)

        planet = self.config["planet"][0]
        shape_file = self.trajectory.shape_file
        gravity_model = Polyhedral(planet, shape_file, self.trajectory).load()
        self.true_acc = gravity_model.accelerations

    def compute_grids(self):
        self.grid_true = Grid(self.trajectory, self.true_acc)
        self.grid_pred = Grid(self.trajectory, self.pred_acc)
        self.grid_error = (self.grid_true - self.grid_pred) / self.grid_pred * 100

    def plot(self, **kwargs):
        vis = MapBase()
        vis.plot_grid(self.grid_true.total, "True", **kwargs)
        vis.plot_grid(self.grid_pred.total, "Pred", **kwargs)
        vis.plot_grid(self.grid_error.total, "Percent Error", **kwargs)


def main():
    import matplotlib.pyplot as plt
    import pandas as pd

    from GravNN.Networks.Model import load_config_and_model
    from GravNN.Trajectories.SurfaceDHGridDist import SurfaceDHGridDist

    df, idx = pd.read_pickle("Data/Dataframes/example.data"), -2  # i = 105 is best
    model_id = df["id"].values[idx]

    config, model = load_config_and_model(df, model_id)
    planet = config["planet"][0]
    trajectory = SurfaceDHGridDist(planet, planet.radius, 30, planet.obj_8k)

    vis = AsteroidSurfaceVisualizer(model, trajectory, config)
    acc_mean = vis.grid_true.total.mean()
    acc_std = vis.grid_true.total.std()
    vlim = [acc_mean - 0.5 * acc_std, acc_mean + 0.5 * acc_std]
    vis.plot(vlim=vlim)
    plt.show()


if __name__ == "__main__":
    main()
