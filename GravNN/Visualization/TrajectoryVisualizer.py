import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
            dr = model_dict["pos_diff"]
            time = model_dict["solution"].t
            label = model_dict["label"]
            color = model_dict["color"]
            plt.plot(time, dr / 1000.0, label=label, color=color)

        plt.ylabel("$|\Delta r|$ Error [km]")
        # plt.xlabel("Time [s]")
        plt.gca().set_xticklabels("")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        # plt.legend()

    def plot_execution_time(self):
        # self.newFig(fig_size=(self.w_full, self.h_full / 5))
        for model_dict in self.experiment.test_models:
            time_real = model_dict["elapsed_time"][1:]
            time_sim = model_dict["solution"].t[1:]
            label = model_dict["label"]
            color = model_dict["color"]
            plt.semilogy(time_sim, time_real, label=label, color=color)

        plt.ylabel("Real Time [s]")
        plt.xlabel("Simulated Time [s]")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        # plt.legend()

    def plot_3d_trajectory(self):
        self.new3DFig(fig_size=(self.w_full / 2, self.w_full / 2))

        true_sol = self.experiment.true_sol
        X, Y, Z = true_sol.y[0:3]
        plt.plot(X, Y, Z, label="True", color="black")
        plt.gca().scatter(X[0], Y[0], Z[0], c="g", s=2)

        for model_dict in self.experiment.test_models:
            sol = model_dict["solution"]
            X, Y, Z = sol.y[0:3]
            label = model_dict["label"]
            color = model_dict["color"]
            plt.plot(X, Y, Z, label=label, color=color)

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

    def plot(self):
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
    from GravNN.GravityModels.HeterogeneousPoly import HeterogeneousPoly
    from GravNN.GravityModels.PointMass import PointMass
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

    true_model = HeterogeneousPoly(planet, planet.obj_8k)

    mass_1 = Eros()
    mass_1.mu = planet.mu / 10
    r_offset_1 = [planet.radius / 3, 0, 0]

    mass_2 = Eros()
    mass_2.mu = -planet.mu / 10
    r_offset_2 = [-planet.radius / 3, 0, 0]

    point_mass_1 = PointMass(mass_1)
    point_mass_2 = PointMass(mass_2)

    true_model.add_point_mass(point_mass_1, r_offset_1)
    true_model.add_point_mass(point_mass_2, r_offset_2)

    test_poly_model = Polyhedral(planet, planet.obj_8k)

    df = pd.read_pickle("Data/Dataframes/heterogenous_eros_041823.data")
    model_id = df.id.values[-1]
    config, test_pinn_model = load_config_and_model(model_id, df)

    experiment = TrajectoryExperiment(
        true_model,
        initial_state=init_state,
        period=1 * 24 * 3600,  # 24 * 3600,
    )
    experiment.add_test_model(test_poly_model, "Poly", "r")
    experiment.add_test_model(test_pinn_model, "PINN", "g")
    experiment.run()

    vis = TrajectoryVisualizer(experiment, obj_file=planet.obj_8k)
    vis.plot()

    plt.show()
