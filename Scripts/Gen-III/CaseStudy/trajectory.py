import matplotlib.pyplot as plt
import numpy as np

from GravNN.Analysis.TrajectoryExperiment import TestModel, TrajectoryExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.TrajectoryVisualizer import TrajectoryVisualizer
from GravNN.Visualization.VisualizationBase import VisualizationBase


class TrajectoryVisualizerMod(TrajectoryVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fig_size = (self.w_tri, self.w_tri)

    def plot(self):
        fig, ax = self.newFig(fig_size=(self.w_tri * 2, self.w_tri))
        self.plot_position_error()
        plt.gca().yaxis.tick_left()
        plt.gca().yaxis.set_label_position("left")
        # Make a legend for the models used
        plt.gca().legend(
            fontsize=4,
            loc="lower right",
        )

        ax2 = plt.twinx()
        self.plot_execution_time()
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        ax2.set_yscale("linear")

        # Make all lines on the twin axis dotted
        for line in ax2.get_lines():
            line.set_linestyle("--")
        # drop grid
        ax2.grid(False)

        # Make a custom legend which shows the solid lines as $\Delta R$ and dotted lines as $t_{exec}$
        plt.legend(
            [
                plt.Line2D((0, 1), (0, 0), color="k", linestyle="-", linewidth=1),
                plt.Line2D((0, 1), (0, 0), color="k", linestyle="--", linewidth=1),
            ],
            [r"$\Delta R$ (km)", r"$t_{exec}$ (s)"],
            loc="upper left",
            fontsize=4,
            # frameon=False,
            # handlelength=1,
        )

        self.plot_3d_trajectory()
        plt.gca().view_init(elev=0, azim=-90)
        # plt.gca().view_init(elev=5, azim=-85)

        # move the legend away from the edge slightly
        plt.gca().legend(loc="upper left", fontsize=4, bbox_to_anchor=(0.10, 0.90))

        # Make the 3D axes take up more of the figure size
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)


def plot_trajectory(true_model, models):
    # planet.radius*2, 0.1, np.pi / 2, 0, 0, 0 -- polar orbit
    init_state = np.array(
        [
            2.88000000e04,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            -1.81246412e-07,
            4.14643442e00,
        ],
    )

    rot_rate = 2 * np.pi / (3600 * 24)
    trajectory_exp = TrajectoryExperiment(
        true_model,
        models,
        initial_state=init_state,
        pbar=True,
        t_mesh_density=1000,
        period=24 * 3600,  # 24 * 3600,
        omega_vec=np.array([0, 0, rot_rate * 10]),
    )
    trajectory_exp.run(override=False, override_truth=False)

    vis = TrajectoryVisualizerMod(trajectory_exp, obj_file=planet.obj_8k, frame="B")
    vis.plot()


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

    poly_test = TestModel(poly_model, "Poly", "r")
    pinn_II_test = TestModel(pinn_II_model, "PINN II", "b")
    pinn_III_test = TestModel(pinn_III_model, "PINN III", "g")
    models = [poly_test, pinn_II_test, pinn_III_test]
    # models = [pinn_II_test, pinn_III_test]

    plot_trajectory(true_model, models)

    vis = VisualizationBase()
    save_name = "primary_orbit"
    vis.save(plt.figure(2), save_name)

    save_name = "primary_trajectory"
    vis.save(plt.figure(1), save_name)
    plt.show()
