import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from GravNN.Analysis.TrajectoryExperiment import TestModel
from GravNN.Visualization.VisualizationBase import VisualizationBase


class TrajectoryVisualizer(VisualizationBase):
    def __init__(self, experiment, obj_file=None, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.mesh = None
        # plt.rc("lines", linewidth=1)

        if obj_file is not None:
            filename, file_extension = os.path.splitext(obj_file)
            self.mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])

    def plot_position_error(self):
        # self.newFig(fig_size=(self.w_full, self.h_full / 5))
        for model_dict in self.experiment.test_models:
            label = model_dict.label
            color = model_dict.color
            time = model_dict.orbit.solution.t
            dr = model_dict.metrics["pos_diff"]
            plt.semilogy(time, dr / 1000.0, label=label, color=color)

        plt.ylabel("$|\Delta r|$ Error [km]")
        plt.xlabel("Time [s]")
        # plt.gca().set_xticklabels("")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        # plt.legend()

    def plot_execution_time(self):
        # self.newFig(fig_size=(self.w_full, self.h_full / 5))
        for model_dict in self.experiment.test_models:
            time_real = model_dict.orbit.elapsed_time[1:]
            time_sim = model_dict.orbit.solution.t[1:]
            label = model_dict.label
            color = model_dict.color
            plt.semilogy(time_sim, time_real, label=label, color=color)

        plt.ylabel("Real Time [s]")
        plt.xlabel("Simulated Time [s]")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        # plt.legend()

    def plot_3d_trajectory(self, new_fig=True):
        if new_fig:
            self.new3DFig()

        true_sol = self.experiment.true_sol
        X, Y, Z = true_sol.y[0:3]
        plt.plot(X, Y, Z, label="True", color="black")
        plt.gca().scatter(X[0], Y[0], Z[0], c="g", s=2)

        for model_dict in self.experiment.test_models:
            sol = model_dict.orbit.solution
            X, Y, Z = sol.y[0:3]
            label = model_dict.label
            color = model_dict.color
            linestyle = model_dict.linestyle
            plt.plot(X, Y, Z, label=label, color=color, linestyle=linestyle)

        plt.legend()
        # plt.gca().set_xlabel("X [m]")
        # plt.gca().set_ylabel("Y [m]")
        # plt.gca().set_zlabel("Z [m]")

        plt.gca().set_xticklabels("")
        plt.gca().set_yticklabels("")
        plt.gca().set_zticklabels("")

        min_lim = np.min(sol.y[0:3])
        max_lim = np.max(sol.y[0:3])
        plt.gca().axes.set_xlim3d(left=min_lim, right=max_lim)
        plt.gca().axes.set_ylim3d(bottom=min_lim, top=max_lim)
        plt.gca().axes.set_zlim3d(bottom=min_lim, top=max_lim)

        plt.gca().set_box_aspect((1, 1, 1))
        plt.gca().view_init(elev=35, azim=235, roll=0)

        if self.mesh is not None:
            tri = Poly3DCollection(
                self.mesh.triangles * 1000,
                cmap=plt.get_cmap("Greys"),
                alpha=0.4,
                # shade=True,
            )

            plt.gca().add_collection3d(tri)

    def plot_reference_trajectory(self, new_fig=True, show_mesh=False, az=235, el=35):
        if new_fig:
            self.new3DFig()
        true_sol = self.experiment.true_sol
        X, Y, Z = true_sol.y[0:3]
        plt.plot(X, Y, Z, label="True", color="black")
        plt.gca().scatter(X[0], Y[0], Z[0], c="g", s=2)

        if show_mesh:
            if self.mesh is not None:
                tri = Poly3DCollection(
                    self.mesh.triangles * 1000,
                    cmap=plt.get_cmap("Greys"),
                    alpha=0.4,
                    # shade=True,
                )

                plt.gca().add_collection3d(tri)
        plt.gca().view_init(elev=el, azim=az, roll=0)

    def plot_3d_trajectory_individually(self, idx, az=235, el=35):
        for i, model_dict in enumerate(self.experiment.test_models):
            if i != idx:
                continue
            self.new3DFig()
            self.plot_reference_trajectory(new_fig=False)

            sol = model_dict.orbit.solution
            X, Y, Z = sol.y[0:3]
            label = model_dict.label
            color = model_dict.color
            linestyle = model_dict.linestyle
            plt.plot(X, Y, Z, label=label, color=color, linestyle=linestyle)

            plt.legend()
            # plt.gca().set_xlabel("X [m]")
            # plt.gca().set_ylabel("Y [m]")
            # plt.gca().set_zlabel("Z [m]")

            plt.gca().set_xticklabels("")
            plt.gca().set_yticklabels("")
            plt.gca().set_zticklabels("")

            min_lim = np.min(sol.y[0:3])
            max_lim = np.max(sol.y[0:3])
            plt.gca().axes.set_xlim3d(left=min_lim, right=max_lim)
            plt.gca().axes.set_ylim3d(bottom=min_lim, top=max_lim)
            plt.gca().axes.set_zlim3d(bottom=min_lim, top=max_lim)

            plt.gca().set_box_aspect((1, 1, 1))
            plt.gca().view_init(elev=el, azim=az, roll=0)

            if self.mesh is not None:
                tri = Poly3DCollection(
                    self.mesh.triangles * 1000,
                    cmap=plt.get_cmap("Greys"),
                    alpha=0.4,
                    # shade=True,
                )

                plt.gca().add_collection3d(tri)

    def plot(self, new_fig=True):
        if new_fig:
            self.newFig((self.w_full / 2, self.w_full / 2))
        plt.subplot2grid((2, 1), (0, 0))
        self.plot_position_error()
        plt.subplot2grid((2, 1), (1, 0))
        self.plot_execution_time()

        self.plot_3d_trajectory()


if __name__ == "__main__":
    import pandas as pd

    from GravNN.Analysis.TrajectoryExperiment import TrajectoryExperiment
    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
    from GravNN.GravityModels.Polyhedral import Polyhedral
    from GravNN.Networks.Model import load_config_and_model

    planet = Eros()

    init_state = np.array(
        [
            -10800.002,
            15273.506,
            10800.00,
            -2.383735,
            -3.371111,
            2.3837354,
        ],
    )

    true_model = generate_heterogeneous_model(planet, planet.obj_8k)
    test_poly_model = Polyhedral(planet, planet.obj_8k)

    df = pd.read_pickle("Data/Dataframes/eros_poly_071123.data")
    model_id = df.id.values[-1]
    config, test_pinn_model = load_config_and_model(df, model_id)

    poly_test = TestModel(test_poly_model, "Poly", "r")
    pinn_test = TestModel(test_pinn_model, "PINN", "g")
    experiment = TrajectoryExperiment(
        true_model,
        [poly_test, pinn_test],
        initial_state=init_state,
        pbar=True,
        period=1 * 3600,  # 24 * 3600,
    )
    experiment.run(override=True)

    vis = TrajectoryVisualizer(experiment, obj_file=planet.obj_8k)
    vis.plot()

    plt.show()
