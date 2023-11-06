import matplotlib.pyplot as plt

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Configs.Eros_Configs import get_default_eros_config
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Visualization.VisualizationBase import VisualizationBase


def plot_extrapolation(config, model, label, new_fig=True):
    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(
        model,
        config,
        points=1000,
        max_radius_scale=10,
    )
    extrapolation_exp.run(override=False)

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(
        extrapolation_exp,
        x_axis="dist_2_COM",
        plot_fcn=plt.semilogy,
        annotate=False,
    )
    vis.fig_size = (vis.w_full, vis.w_half)
    if not new_fig:
        plt.figure(1)

    vis.plot_extrapolation_percent_error(
        plot_std=False,
        plot_max=False,
        new_fig=new_fig,
        annotate=False,
        linewidth=1,
        label=label,
    )
    plt.ylim([1e-4, 1e2])
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    planet = Eros()
    obj_file = planet.obj_8k

    config = get_default_eros_config()
    true_model = generate_heterogeneous_model(planet, obj_file)
    poly_model = Polyhedral(planet, obj_file)
    # pinn_II_config, pinn_II_model = load_config_and_model()
    # pinn_III_config, pinn_III_model = load_config_and_model()

    plot_extrapolation(config, poly_model, "Polyhedral")
    # plot_extrapolation(config, pinn_II_model, "PINN II", new_fig=False)
    # plot_extrapolation(config, pinn_III_model, "PINN III", new_fig=False)

    vis = VisualizationBase()
    save_name = "primary_extrapolation"
    vis.save(plt.gcf(), save_name)

    plt.show()
