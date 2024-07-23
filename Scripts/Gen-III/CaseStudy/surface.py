import matplotlib.pyplot as plt
import numpy as np

from GravNN.Analysis.SurfaceExperiment import SurfaceExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
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
            # max_percent=0.1,
            max_percent=1,
            min_percent=0.0001,
            log=True,
            cmap_reverse=False,
            percent=True,
            cbar=False,
        )

        plt.gca().set_axis_off()
        plt.gca().view_init(elev=90, azim=-90)

        avg = np.nanmean(self.experiment.percent_error_acc)
        model_avg = f"Mean Error: {avg:.2E}"

        model_name = f"{kwargs.get('model_name', '')}"
        plt.annotate(
            model_name,
            xy=(0.5, 0.85),
            xycoords="axes fraction",
            ha="center",
            bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.7),
            fontsize=6,
        )
        plt.annotate(
            model_avg,
            xy=(0.5, 0.15),
            xycoords="axes fraction",
            ha="center",
            bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.7),
            fontsize=6,
        )

    def plot(self, label):
        self.plot_percent_error(model_name=label)


def plot_surface(true_model, test_model, label):
    exp = SurfaceExperiment(test_model, true_model)
    exp.run(override=False)

    vis = SurfaceVisualizerMod(exp)
    vis.plot(label)

    save_name = f"primary_surface_{label}".replace(" ", "_")
    vis.save(plt.gcf(), save_name)


if __name__ == "__main__":
    planet = Eros()
    obj_file = planet.obj_200k

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

    plot_surface(true_model, poly_model, "Polyhedral")
    plot_surface(true_model, pinn_II_model, "PINN II")
    plot_surface(true_model, pinn_III_model, "PINN III")

    plt.show()
