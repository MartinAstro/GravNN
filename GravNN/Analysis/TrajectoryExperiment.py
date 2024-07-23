import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from GravNN.Analysis.ExperimentBase import ExperimentBase
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.ProgressBar import ProgressBar
from GravNN.Support.RigidBodyKinematics import euler1232C


class TestModel:
    def __init__(self, model, label, color, linestyle="-"):
        self.model = model
        self.label = label
        self.color = color
        self.linestyle = linestyle
        self.orbit = None


def compute_BN(t, omega_vec):
    if isinstance(t, float):
        euler = omega_vec * t
        DCM = euler1232C(euler)
    elif isinstance(t, np.ndarray):
        eulers = omega_vec.reshape((3, 1)) * t
        eulers = eulers.T
        DCM = np.array([euler1232C(euler) for euler in eulers])
    else:
        raise Exception("t must be float or np.ndarray")
    return DCM


class TrajectoryPropagator(ExperimentBase):
    def __init__(
        self,
        model,
        initial_state,
        period,
        t_mesh_density=100,
        random_seed=1234,
        tol=1e-10,
        omega_vec=np.array([0.0, 0.0, 0.0]).reshape((3, 1)),
        pbar=False,
    ):
        super().__init__(
            model,
            initial_state=initial_state,
            period=period,
            t_mesh_density=t_mesh_density,
            random_seed=random_seed,
            tol=tol,
            omega_vec=omega_vec,
            pbar=pbar,
        )
        self.true_model = model
        self.t_mesh_density = t_mesh_density
        self.period = period
        self.x0 = initial_state
        self.test_models = []
        self.pbar = pbar
        self.tol = tol
        self.omega_vec = omega_vec
        np.random.seed(random_seed)

    def generate_trajectory(self, model, X0, t_eval):
        def fun(t, y, IC=None):
            "Return the first-order system"
            R = y[0:3]
            V = y[3:6]

            BN = compute_BN(t, self.omega_vec).squeeze()
            x_pos_B = BN @ R
            x_pos_B = x_pos_B.reshape((1, -1))
            a_B = model.compute_acceleration(x_pos_B)
            a_B = np.array(a_B).squeeze()
            a_N = BN.T @ a_B

            dxdt = np.hstack((V, a_N)).squeeze()

            is_nan = np.isnan(y).any()
            diverging = np.abs(a_N).max() > 1e6
            diverged = np.isinf(np.abs(y))
            if is_nan or diverging or diverged.any():
                raise ValueError("Non-feasible values encountered in integration")

            if t > t_eval[fun.t_eval_idx]:
                fun.elapsed_time.append(time.time() - fun.start_time)
                fun.pbar.update(t_eval[fun.t_eval_idx])
                fun.t_eval_idx += 1

            return dxdt

        fun.t_eval_idx = 0
        fun.elapsed_time = []
        fun.pbar = ProgressBar(t_eval[-1], self.pbar)

        # avoid the first call to fun() to avoid a duplicate call to compute_acceleration
        model.compute_acceleration(np.array([[100.0, 100.0, 100.0]]))
        fun.start_time = time.time()

        try:
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

            dt = fun.elapsed_time
        except ValueError as e:
            print("Integration Stopped:", e)
            sol = None
            dt = np.nan

        return sol, dt

    def generate_data(self):
        if not hasattr(self, "solution"):
            self.t_mesh = np.linspace(
                0,
                self.period,
                self.t_mesh_density,
                endpoint=True,
            )

            self.solution, self.elapsed_time = self.generate_trajectory(
                self.true_model,
                self.x0,
                self.t_mesh,
            )
        data = {
            "solution": self.solution,
            "elapsed_time": self.elapsed_time,
            "t_mesh": self.t_mesh,
        }
        return data


class TrajectoryExperiment:
    def __init__(
        self,
        true_model,
        test_models,
        initial_state,
        period,
        t_mesh_density=100,
        random_seed=1234,
        tol=1e-10,
        omega_vec=np.array([0.0, 0.0, 0.0]).reshape((3, 1)),
        pbar=False,
    ):
        self.true_model = true_model
        self.test_models = test_models
        self.initial_state = initial_state
        self.period = period
        self.t_mesh_density = t_mesh_density
        self.pbar = pbar
        self.random_seed = random_seed
        self.tol = tol
        self.omega_vec = omega_vec

    def run(self, override=False, override_truth=False):
        self.true_orbit = TrajectoryPropagator(
            self.true_model,
            initial_state=self.initial_state,
            period=self.period,
            t_mesh_density=self.t_mesh_density,
            random_seed=self.random_seed,
            tol=self.tol,
            omega_vec=self.omega_vec,
            pbar=self.pbar,
        )

        self.true_orbit.run(override=override_truth)

        for i, model in enumerate(self.test_models):
            orbit = TrajectoryPropagator(
                model.model,
                initial_state=self.initial_state,
                period=self.period,
                t_mesh_density=self.t_mesh_density,
                random_seed=self.random_seed,
                tol=self.tol,
                omega_vec=self.omega_vec,
                pbar=self.pbar,
            )
            orbit.run(override=override)
            self.test_models[i].orbit = orbit

        self.true_sol = self.true_orbit.solution
        for i, test_model in enumerate(self.test_models):
            test_sol = test_model.orbit.solution

            if test_model.orbit.solution is not None:
                # if the orbit completed
                dy = test_sol.y - self.true_sol.y
                pos_diff = np.cumsum(np.linalg.norm(dy[0:3, :], axis=0))
                state_diff = np.cumsum(np.linalg.norm(dy, axis=0))
                pos_diff_inst = np.linalg.norm(dy[0:3, :], axis=0)
            else:
                # if the propagation terminated early
                pos_diff = np.nan
                state_diff = np.nan
                pos_diff_inst = np.nan

            metrics = {
                "pos_diff": pos_diff,
                "state_diff": state_diff,
                "pos_diff_inst": pos_diff_inst,
            }
            self.test_models[i].metrics = metrics


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

    true_model = generate_heterogeneous_model(planet, planet.obj_8k)
    test_poly_model = Polyhedral(planet, planet.obj_8k)

    df = pd.read_pickle("Data/Dataframes/eros_poly_071123.data")
    model_id = df.id.values[-1]
    config, test_pinn_model = load_config_and_model(df, model_id)

    poly_test = TestModel(test_poly_model, "Poly", "r")
    pinn_test = TestModel(test_pinn_model, "PINN", "g")
    test_models = [poly_test, pinn_test]
    test_models = [pinn_test]

    experiment = TrajectoryExperiment(
        true_model,
        test_models,
        initial_state=init_state,
        # period=24 * 3600,  # 24 * 3600,
        period=24 * 3600,  # 24 * 3600,
    )
    experiment.run()

    plt.show()


if __name__ == "__main__":
    main()
