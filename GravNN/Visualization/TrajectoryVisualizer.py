import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from GravNN.Analysis.TrajectoryExperiment import TestModel, compute_BN
from GravNN.Visualization.VisualizationBase import VisualizationBase


class TrajectoryVisualizer(VisualizationBase):
    def __init__(self, experiment, obj_file=None, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.mesh = None
        self.frame = kwargs.get("frame", "N")

        self.true_model = TestModel(self.experiment.true_model, "True", "black")
        self.true_model.orbit = self.experiment.true_orbit

        if obj_file is not None:
            filename, file_extension = os.path.splitext(obj_file)
            self.mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])

    def plot_position_error(self):
        for model in self.experiment.test_models:
            label = model.label
            color = model.color
            time = model.orbit.solution.t
            dr = model.metrics["pos_diff"]
            plt.semilogy(time, dr / 1000.0, label=label, color=color)

            print("Model: ", label, " dR: ", dr[-1])
            print("Model: ", label, " dR_i: ", model.metrics["pos_diff_inst"][-1])

        plt.ylabel("$\sum |\Delta r|$ Error [km]")
        plt.xlabel("Simulated Time [s]")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")

    def plot_execution_time(self):
        # self.newFig(fig_size=(self.w_full, self.h_full / 5))
        for model in self.experiment.test_models:
            time_real = model.orbit.elapsed_time[1:]
            time_sim = model.orbit.solution.t[1:]
            label = model.label
            color = model.color
            plt.semilogy(time_sim, time_real, label=label, color=color)

            print("Model: ", label, " Time: ", time_real[-1])

        plt.ylabel("Real Time [s]")
        plt.xlabel("Simulated Time [s]")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        # plt.legend()

    def plot_orbit(self, model, az=235, el=35, **kwargs):
        sol = model.orbit.solution
        omega_vec = model.orbit.omega_vec

        if self.frame == "B":
            BN = compute_BN(sol.t, omega_vec)
            X = sol.y[0:3].T
            X = X.reshape((-1, 3, 1))
            X = BN @ X
            X = X.squeeze()
            X, Y, Z = X[:, 0], X[:, 1], X[:, 2]
        else:
            X, Y, Z = sol.y[0:3]

        label = model.label
        color = model.color
        linestyle = model.linestyle
        linewidth = kwargs.get("linewidth", 1)

        plt.plot(
            X,
            Y,
            Z,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

        plt.legend()

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

    def plot_shape_model(self):
        if self.mesh is not None:
            tri = Poly3DCollection(
                self.mesh.triangles * 1000,
                cmap=plt.get_cmap("Greys"),
                alpha=0.4,
            )
            plt.gca().add_collection3d(tri)

    def plot_3d_trajectory(self, new_fig=True, **kwargs):
        if new_fig:
            self.new3DFig()
        self.plot_orbit(self.true_model, **kwargs)
        # plt.gca().scatter(X[0], Y[0], Z[0], c="g", s=2)
        for model in self.experiment.test_models:
            self.plot_orbit(model, **kwargs)
        self.plot_shape_model()

    def plot_reference_trajectory(self, new_fig=True, **kwargs):
        if new_fig:
            self.new3DFig()
        self.plot_orbit(self.true_model, **kwargs)

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

    true_model = generate_heterogeneous_model(planet, planet.obj_200k)
    test_poly_model = Polyhedral(planet, planet.obj_200k)

    df = pd.read_pickle("Data/Dataframes/eros_poly_071123.data")
    model_id = df.id.values[-1]
    config, test_pinn_model = load_config_and_model(df, model_id)

    poly_test = TestModel(test_poly_model, "Poly", "r")
    pinn_test = TestModel(test_pinn_model, "PINN", "g")

    rot_rate = 2 * np.pi / (3600 * 24)
    experiment = TrajectoryExperiment(
        true_model,
        [poly_test, pinn_test],
        initial_state=init_state,
        pbar=True,
        t_mesh_density=1000,
        period=24 * 3600,  # 24 * 3600,
        omega_vec=np.array([0, 0, rot_rate * 10]),
    )
    experiment.run(override=False)

    vis = TrajectoryVisualizer(experiment, obj_file=planet.obj_8k)
    vis.plot()
    vis = TrajectoryVisualizer(experiment, obj_file=planet.obj_8k, frame="B")
    vis.plot()

    plt.show()
