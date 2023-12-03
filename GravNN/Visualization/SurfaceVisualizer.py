import matplotlib.pyplot as plt

from GravNN.Visualization.PolyVisualization import PolyVisualization


class SurfaceVisualizer(PolyVisualization):
    def __init__(self, experiment, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment

    def plot_test_model(self, **kwargs):
        self.plot_polyhedron(
            self.experiment.obj_file,
            self.experiment.a_pred,
            label="Predicted Acceleration ($m/s^2$)",
            cmap="bwr",
            **kwargs,
        )

    def plot_true_model(self, **kwargs):
        self.plot_polyhedron(
            self.experiment.obj_file,
            self.experiment.a_true,
            label="True Acceleration ($m/s^2$)",
            cmap="bwr",
            **kwargs,
        )

    def plot_percent_error(self, **kwargs):
        self.plot_polyhedron(
            self.experiment.obj_file,
            self.experiment.percent_error_acc,
            label="Acceleration Error (\%)",
            percent=True,
            **kwargs,
        )
        pass

    def plot(self):
        self.plot_true_model()
        self.plot_test_model()
        self.plot_percent_error()


if __name__ == "__main__":
    from GravNN.Analysis.SurfaceExperiment import SurfaceExperiment
    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
    from GravNN.GravityModels.Polyhedral import Polyhedral

    planet = Eros()
    test_model = Polyhedral(planet, planet.obj_8k)
    true_model = generate_heterogeneous_model(planet, planet.obj_8k)
    exp = SurfaceExperiment(test_model, true_model)
    exp.run()

    vis = SurfaceVisualizer(exp)
    vis.plot()
    plt.show()
