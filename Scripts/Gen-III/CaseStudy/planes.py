import matplotlib.pyplot as plt
import numpy as np

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


class PlanesVisualizerMod(PlanesVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fig_size = (self.w_tri, self.w_tri)

    def default_plot(self, x, y, colorbar_label, **kwargs):
        fig, ax = self.newFig()

        # If cBar is explicitly set to False, only make one row
        use_cbar = kwargs.get("cbar", False)

        defaults = {
            "ticks": False,
            "labels": False,
        }
        defaults.update(kwargs)
        defaults.update({"cbar": use_cbar})

        self.plot_plane(
            x,
            y,
            plane="xy",
            colorbar_label=colorbar_label,
            **defaults,
        )

        # make an annotated box at the top center of the figure
        # stating the model, and a second annotated box at the bottom
        # center of the figure stating the average error
        avg = np.nanmean(y)
        model_name = f"{kwargs.get('model_name', '')}"
        model_avg = f"Mean Error: {avg:.2E}"

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


def plot_planes(config, model, label):
    planet = config["planet"][0]
    # bounds = [-2 * planet.radius, 2 * planet.radius]
    bounds = [-5 * planet.radius, 5 * planet.radius]
    bounds = [-3 * planet.radius, 3 * planet.radius]
    planes_exp = PlanesExperiment(model, config, bounds, 200)

    if isinstance(model, Polyhedral):
        planes_exp.load_model_data(model)
    planes_exp.run()

    vis = PlanesVisualizerMod(planes_exp)
    vis.plot(z_max=100, z_min=0.01, log=True, model_name=label)
    # vis.plot(z_max=10, z_min=0.01, log=True)

    save_name = f"primary_planes_{label}".replace(" ", "_")
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

    plot_planes(pinn_III_config, poly_model, "Polyhedral")
    plot_planes(pinn_III_config, pinn_II_model, "PINN II")
    plot_planes(pinn_III_config, pinn_III_model, "PINN III")

    plt.show()
