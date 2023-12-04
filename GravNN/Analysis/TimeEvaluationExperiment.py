import time

import numpy as np
import tensorflow as tf

from GravNN.Analysis.ExperimentBase import ExperimentBase
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral


class TimeEvaluationExperiment(ExperimentBase):
    def __init__(self, model, points, r_min):
        super().__init__(model, points, r_min)
        self.model = model
        self.points = points
        self.r_min = r_min
        self.x_test = self.get_test_data()

    def initialize_fcn(self, fcn):
        if fcn == "acceleration":
            fcn = self.model.compute_acceleration
        elif fcn == "potential":
            fcn = self.model.compute_potential
        elif fcn == "predict":
            fcn = self.model.predict
        return fcn

    def get_test_data(self):
        random_unit = np.random.uniform(-1, 1, (self.points, 3))
        random_unit /= np.linalg.norm(random_unit)
        random_radius = np.random.uniform(
            self.r_min,
            self.r_min * 100,
            size=(self.points, 1),
        )
        x_test = random_unit * random_radius
        return x_test

    def enforce_type(self, model, x_input):
        try:
            x = x_input.astype(model.dtype)
        except Exception:
            x = x_input.astype(np.float32)

        if self.input_type == "tensor":
            x = tf.constant(x)

        return x

    def batch_test(self, fcn, x):
        # warm start
        _ = fcn(x[0:1, :])

        # evaluate
        start_time = time.time()
        _ = fcn(x)
        dt = time.time() - start_time
        return dt

    def single_test(self, fcn, x):
        # warm start
        _ = fcn(x[0:1, :])

        # evaluate
        start_time = time.time()
        for i in range(len(x)):
            _ = fcn(x[i : i + 1, :])
        dt = time.time() - start_time
        return dt

    def time(self):
        a_fcn = self.model.compute_acceleration

        self.dt_a_batch = self.batch_test(a_fcn, self.x_test)
        self.dt_a_single = self.single_test(a_fcn, self.x_test)

        # self.dt_u_batch = self.batch_test(u_fcn, self.x_test)
        # self.dt_u_single = self.single_test(u_fcn, self.x_test)

        metrics = {
            "a_batch_time": self.dt_a_batch,
            "a_single_time": self.dt_a_single,
            # "u_batch_time": self.dt_u_batch,
            # "u_single_time": self.dt_u_single,
        }
        return metrics

    def generate_data(self):
        data = self.time()
        return data


def main():
    import pandas as pd

    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle("Data/Dataframes/eros_cost_fcn_pinn_III_fuse.data")
    model_id = df["id"].values[-1]
    config, pinn_model = load_config_and_model(df, model_id)
    print(f"Number of Units: {config['num_units'][0]}")
    print(f"Number of Params: {config['params'][0]}")

    # Batch vs Single
    # Numba vs No Numba
    # CPU vs GPU
    extrapolation_exp = TimeEvaluationExperiment(
        pinn_model,
        1000,
        Eros().radius,
    )
    data = extrapolation_exp.run()
    print(data)

    sh_file = Eros().sh_file
    sh_model = SphericalHarmonics(sh_file, degree=16, parallel=False)
    extrapolation_exp = TimeEvaluationExperiment(
        sh_model,
        1000,
        Eros().radius,
    )
    data = extrapolation_exp.run()
    print(data)

    poly_model = Polyhedral(Eros(), Eros().obj_200k)
    extrapolation_exp = TimeEvaluationExperiment(
        poly_model,
        1000,
        Eros().radius,
    )
    data = extrapolation_exp.run()
    print(data)


if __name__ == "__main__":
    main()
