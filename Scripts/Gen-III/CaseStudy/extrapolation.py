import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Visualization.VisualizationBase import VisualizationBase


def shade_background():
    y_max = 1e2
    y_min = 1e-4
    alpha = 0.2
    eps = 1e-3
    plt.fill_between(
        [0, 1 - eps],
        y_min,
        y_max,
        color="yellow",
        alpha=alpha,
    )
    plt.fill_between(
        [1, 10 - eps],
        y_min,
        y_max,
        color="green",
        alpha=alpha,
    )
    plt.fill_between(
        [10, 20],
        y_min,
        y_max,
        color="red",
        alpha=alpha,
    )
    # add a legend for the colors in upper left
    legend_2 = plt.legend(
        handles=[
            Rectangle((0, 0), 1, 1, alpha=alpha, color="yellow", label="Interior"),
            Rectangle((0, 0), 1, 1, alpha=alpha, color="green", label="Exterior"),
            Rectangle((0, 0), 1, 1, alpha=alpha, color="red", label="Extrapolation"),
        ],
        loc="upper right",
    )
    plt.gca().add_artist(legend_2)


def plot_extrapolation(config, model, label, color, new_fig=True):
    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(
        model,
        config,
        points=5000,
        extrapolation_bound=100,
    )
    extrapolation_exp.test_dist_2_surf_idx = None  # hack to avoid sorting
    extrapolation_exp.test_r_surf = None  # hack to avoid sorting
    extrapolation_exp.run(override=False)

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(
        extrapolation_exp,
        x_axis="dist_2_COM",
        plot_fcn=plt.semilogy,
        annotate=False,
    )
    vis.fig_size = (vis.w_full, vis.h_tri * 0.9)
    if not new_fig:
        plt.figure(1)

    vis.plot_extrapolation_percent_error(
        plot_std=False,
        plot_max=False,
        new_fig=new_fig,
        annotate=False,
        linewidth=1,
        label=label,
        color=color,
    )
    plt.ylim([1e-4, 1e2])
    plt.xlim([0, 20])
    plt.tight_layout()


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

    plot_extrapolation(pinn_III_config, poly_model, "Polyhedral", "r")
    plot_extrapolation(pinn_III_config, pinn_II_model, "PINN II", "b", new_fig=False)
    plot_extrapolation(pinn_III_config, pinn_III_model, "PINN III", "g", new_fig=False)
    legend_1 = plt.legend(loc="lower right")
    plt.gca().add_artist(legend_1)

    shade_background()

    vis = VisualizationBase()
    vis.fig_size = (vis.w_full, vis.w_half)
    save_name = "primary_extrapolation"
    vis.save(plt.gcf(), save_name, bbox_inches=None)

    plt.show()
