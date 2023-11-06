import matplotlib.pyplot as plt
import numpy as np

from GravNN.Analysis.TrajectoryExperiment import TestModel, TrajectoryExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Configs.Eros_Configs import get_default_eros_config
from GravNN.Visualization.TrajectoryVisualizer import TrajectoryVisualizer
from GravNN.Visualization.VisualizationBase import VisualizationBase


class TrajectoryVisualizerMod(TrajectoryVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fig_size = (self.w_tri, self.w_tri)

    def plot(self):
        self.plot_reference_trajectory()
        self.plot_shape_model()

        # remove legend
        plt.gca().get_legend().remove()

        # # remove ticks
        # plt.gca().set_xticks([])
        # plt.gca().set_yticks([])
        # plt.gca().set_zticks([])

        # # set elevation and azimuth
        plt.gca().view_init(elev=0, azim=-90)

        # # decrease the padding / whitespace
        # plt.gca().xaxis.set_tick_params(pad=-1)
        # plt.gca().yaxis.set_tick_params(pad=-1)
        # plt.gca().zaxis.set_tick_params(pad=-1)

        # # decrease the whitespace outside the axes
        # plt.gca().set_xmargin(0)
        # plt.gca().set_ymargin(0)
        # plt.gca().set_zmargin(0)

        # remove the axes
        plt.gca().set_axis_off()

        fig, ax = self.newFig(fig_size=(self.w_tri * 2, self.w_tri))
        self.plot_position_error()
        plt.gca().yaxis.tick_left()
        plt.gca().yaxis.set_label_position("left")
        plt.twinx()
        self.plot_execution_time()
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")


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
        t_mesh_density=10000,
        period=24 * 3600 * 10,  # 24 * 3600,
        omega_vec=np.array([0, 0, rot_rate]),
    )
    trajectory_exp.run()

    vis = TrajectoryVisualizerMod(trajectory_exp, obj_file=planet.obj_8k, frame="B")
    vis.plot()


if __name__ == "__main__":
    planet = Eros()
    obj_file = planet.obj_8k

    config = get_default_eros_config()
    true_model = generate_heterogeneous_model(planet, obj_file)
    poly_model = Polyhedral(planet, obj_file)
    # pinn_II_config, pinn_II_model = load_config_and_model()
    # pinn_III_config, pinn_III_model = load_config_and_model()

    poly_test = TestModel(poly_model, "Poly", "g")
    models = [poly_test]

    plot_trajectory(true_model, models)

    vis = VisualizationBase()
    save_name = "primary_orbit"
    vis.save(plt.figure(1), save_name)

    save_name = "primary_trajectory"
    vis.save(plt.figure(2), save_name)
    plt.show()
