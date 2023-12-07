import matplotlib.pyplot as plt

from GravNN.Analysis.SurfaceExperiment import SurfaceExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Configs.Eros_Configs import get_default_eros_config
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.SurfaceVisualizer import SurfaceVisualizer


class SurfaceVisualizerMod(SurfaceVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fig_size = (self.w_tri, self.w_tri)

    def plot_percent_error(self, **kwargs):
        self.plot_polyhedron(
            self.experiment.obj_file,
            self.experiment.percent_error_acc,
            label="Acceleration Error (\%)",
            cmap="jet",
            max_percent=0.1,
            cmap_reverse=False,
            percent=True,
            cbar=False,
        )

        plt.gca().set_axis_off()
        plt.gca().view_init(elev=90, azim=-90)

    def plot(self):
        self.plot_percent_error()


def plot_surface(true_model, test_model, label):
    exp = SurfaceExperiment(test_model, true_model)
    exp.run(override=True)

    vis = SurfaceVisualizerMod(exp)
    vis.plot()

    save_name = f"primary_surface_{label}".replace(" ", "_")
    vis.save(plt.gcf(), save_name)


if __name__ == "__main__":
    planet = Eros()
    obj_file = planet.obj_200k

    config = get_default_eros_config()
    true_model = generate_heterogeneous_model(planet, obj_file)
    poly_model = Polyhedral(planet, obj_file)
    pinn_II_config, pinn_II_model = load_config_and_model(
        "Data/Dataframes/pinn_primary_figure_II.data",
        idx=-1,
    )
    pinn_III_config, pinn_III_model = load_config_and_model(
        "Data/Dataframes/pinn_primary_figure_III.data",
        idx=-1,
    )

    # plot_surface(true_model, poly_model, "Polyhedral")
    plot_surface(true_model, pinn_II_model, "PINN II")
    plot_surface(true_model, pinn_III_model, "PINN III")

    plt.show()
