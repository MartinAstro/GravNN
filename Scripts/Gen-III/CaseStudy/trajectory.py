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

    def plot_position_error(self):
        for model in self.experiment.test_models:
            label = model.label
            color = model.color
            time = model.orbit.solution.t
            dr = model.metrics["pos_diff_inst"]
            plt.semilogy(time / 3600, dr, label=label, color=color)

            dt = time[1] - time[0]

            print("Model: ", label, " dR: ", model.metrics["pos_diff"][-1])
            print(
                "Model: ",
                label,
                " dR-Avg: ",
                model.metrics["pos_diff"][-1] * dt / time[-1],
            )
            print("Model: ", label, " dR_i: ", model.metrics["pos_diff_inst"][-1])

        plt.ylabel(r"Inst. $\Delta x$ [m]")
        # plt.ylabel("$\sum |\Delta x|$ Error [km]")
        plt.xlabel("Simulated Time [hr]")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.ylim([1e-1, 5e4])
        # plt.ylim([1e-3, 1e4])

    def plot(self):
        fig, ax = self.newFig(fig_size=(self.w_tri * 2, self.w_tri))
        self.plot_position_error()
        plt.gca().yaxis.tick_left()
        plt.gca().yaxis.set_label_position("left")
        # Make a legend for the models used
        plt.gca().legend(fontsize=4, loc="lower right")

        self.fig_size = (self.w_tri, self.w_tri)
        self.plot_trajectory_projection(linewidth=0.25, frame="B")
        plt.gca().legend(loc="lower right", fontsize=4)
        # make lines in legend wider
        for line in plt.gca().get_legend().get_lines():
            line.set_linewidth(0.5)

        plt.gca().yaxis.set_ticks_position("right")
        # rotate tick labels
        plt.setp(plt.gca().get_yticklabels(), rotation=90)
        plt.gca().yaxis.set_label_position("right")
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
    # plt.show()
