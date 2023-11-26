import matplotlib.pyplot as plt
import numpy as np

from GravNN.Analysis.ExperimentBase import ExperimentBase
from GravNN.Trajectories.SurfaceDist import SurfaceDist


class SurfaceExperiment(ExperimentBase):
    def __init__(self, model, true_model, **kwargs):
        super().__init__(model, true_model, **kwargs)
        self.model = model
        self.true_model = true_model

        self.planet = true_model.planet
        self.obj_file = true_model.obj_file

    def get_true_data(self):
        traj = SurfaceDist(self.planet, self.obj_file)
        self.x_true = traj.positions

        self.true_model.trajectory = traj
        self.true_model.homogeneous_poly.trajectory = traj
        self.true_model.load()

        self.a_true = self.true_model.accelerations
        self.u_true = self.true_model.potentials

    def get_model_data(self):
        if not hasattr(self, "a_pred"):
            self.a_pred = self.model.compute_acceleration(self.x_true)
        try:
            self.a_pred = self.a_pred.numpy()
        except Exception:
            pass

    def compute_percent_error(self):
        def percent_error(x_hat, x_true):
            diff_mag = np.linalg.norm(x_true - x_hat, axis=1)
            true_mag = np.linalg.norm(x_true, axis=1)
            percent_error = diff_mag / true_mag * 100
            return percent_error

        self.percent_error_acc = percent_error(self.a_pred, self.a_true)

    def generate_data(self):
        self.get_true_data()
        self.get_model_data()
        self.compute_percent_error()
        data = {
            "a_pred": self.a_pred,
        }
        return data


if __name__ == "__main__":
    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.GravityModels.HeterogeneousPoly import (
        generate_heterogeneous_model,
    )
    from GravNN.GravityModels.Polyhedral import Polyhedral
    from GravNN.Trajectories.SurfaceDist import SurfaceDist
    from GravNN.Visualization.SurfaceVisualizer import SurfaceVisualizer

    planet = Eros()
    obj_file = planet.obj_8k
    model = Polyhedral(planet, obj_file)
    true_model = generate_heterogeneous_model(planet, obj_file)
    exp = SurfaceExperiment(model, true_model)
    exp.run()

    vis = SurfaceVisualizer(exp)
    vis.plot()
    plt.show()
