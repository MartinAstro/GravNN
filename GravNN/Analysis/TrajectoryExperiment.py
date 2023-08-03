import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import HeterogeneousPoly
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.ProgressBar import ProgressBar


class TrajectoryExperiment:
    def __init__(
        self,
        true_grav_model,
        initial_state,
        period,
        t_mesh_density=100,
        pbar=False,
        random_seed=1234,
        tol=1e-10,
    ):
        self.true_model = true_grav_model
        self.t_mesh_density = t_mesh_density
        self.period = period
        self.x0 = initial_state
        self.test_models = []
        self.pbar = pbar
        self.tol = tol
        np.random.seed(random_seed)

    def add_test_model(self, model, label, color, linestyle="-"):
        self.test_models.append(
            {
                "model": model,
                "label": label,
                "color": color,
                "linestyle": linestyle,
            },
        )

    def generate_trajectory(self, model, X0, t_eval):
        def fun(t, y, IC=None):
            "Return the first-order system"
            R = np.array([y[0:3]])
            V = np.array([y[3:6]])
            a = model.compute_acceleration(R)
            dxdt = np.hstack((V, a)).squeeze()

            if t > t_eval[fun.t_eval_idx]:
                fun.elapsed_time.append(time.time() - fun.start_time)
                fun.pbar.update(t_eval[fun.t_eval_idx])
                fun.t_eval_idx += 1
            return dxdt

        fun.t_eval_idx = 0
        fun.elapsed_time = []
        fun.pbar = ProgressBar(t_eval[-1], self.pbar)
        fun.start_time = time.time()

        sol = solve_ivp(
            fun,
            [0, t_eval[-1]],
            X0.reshape((-1,)),
            t_eval=t_eval,
            atol=self.tol,
            rtol=self.tol,
        )
        fun.elapsed_time.append(time.time() - fun.start_time)
        fun.pbar.update(t_eval[-1])
        fun.pbar.close()
        return sol, fun.elapsed_time

    def compute_differences(self):
        for i, model_dict in enumerate(self.test_models):
            test_sol = model_dict["solution"]

            dy = test_sol.y - self.true_sol.y
            dX = np.linalg.norm(dy, axis=0)
            dX = np.linalg.norm(dy[0:3], axis=0)

            self.test_models[i].update({"pos_diff": dX})

    def run(self):
        self.t_mesh = np.linspace(0, self.period, self.t_mesh_density, endpoint=True)

        # Generate trajectories using the true and test grav models
        self.true_sol, self.elapsed_time = self.generate_trajectory(
            self.true_model,
            self.x0,
            self.t_mesh,
        )

        # generate trajectories for all test models
        for i, model_dict in enumerate(self.test_models):
            model = model_dict["model"]

            sol, elapsed_time = self.generate_trajectory(
                model,
                self.x0,
                self.t_mesh,
            )
            self.test_models[i].update({"solution": sol, "elapsed_time": elapsed_time})

        self.compute_differences()


def main():
    planet = Eros()

    init_state = np.array(
        [
            4.61513747e04,
            8.12741755e04,
            -1.00860719e04,
            8.49819800e-01,
            -1.49764060e00,
            2.47435298e00,
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
    config, test_pinn_model = load_config_and_model(df, model_id)

    experiment = TrajectoryExperiment(
        true_model,
        initial_state=init_state,
        period=24 * 3600,  # 24 * 3600,
    )
    experiment.add_test_model(test_poly_model, "Poly", "r")
    experiment.add_test_model(test_pinn_model, "PINN", "g")
    experiment.run()

    plt.show()


if __name__ == "__main__":
    main()
