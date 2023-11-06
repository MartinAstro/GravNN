import matplotlib.pyplot as plt

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Configs.Eros_Configs import get_default_eros_config
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


def plot_planes(config, model, label):
    planet = config["planet"][0]
    bounds = [-2 * planet.radius, 2 * planet.radius]
    planes_exp = PlanesExperiment(model, config, bounds, 100)
    planes_exp.run()

    vis = PlanesVisualizerMod(planes_exp)
    vis.plot(z_max=10)

    save_name = f"primary_planes_{label}"
    vis.save(plt.gcf(), save_name)


if __name__ == "__main__":
    planet = Eros()
    obj_file = planet.obj_8k

    config = get_default_eros_config()
    true_model = generate_heterogeneous_model(planet, obj_file)
    poly_model = Polyhedral(planet, obj_file)
    # pinn_II_config, pinn_II_model = load_config_and_model()
    # pinn_III_config, pinn_III_model = load_config_and_model()

    plot_planes(config, poly_model, "Polyhedral")
    # plot_planes(config, pinn_II_model, "PINN II", new_fig=False)
    # plot_planes(config, pinn_III_model, "PINN III", new_fig=False)

    plt.show()
