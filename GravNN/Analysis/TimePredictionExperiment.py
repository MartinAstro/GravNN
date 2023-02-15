import time
import numpy as np
class TimePredictionExperiment:
    def __init__(self, model, r_max, r_min, points):
        self.model = model
        self.points = points
        self.r_min = r_min
        self.r_max = r_max

    def get_test_data(self):
        random_unit = np.random.uniform(-1, 1, (self.points, 3))
        random_unit /= np.linalg.norm(random_unit)
        random_radius = np.random.uniform(self.r_min, self.r_max, size=(self.points, 1))
        x_test = random_unit * random_radius
        return x_test

    def time_predictions(self, x, batch):
        # warm start
        try:
            x = x.astype(self.model.dtype)
        except:
            x = x.astype(np.float32)
        import tensorflow as tf
        x_tensor = x
        # x_tensor = tf.constant(x)
        # x_tensor = tf.data.Dataset.from_tensor_slices(x)
        self.model.compute_acceleration(np.array([[1,2,3.]], dtype=np.float32))
        self.model.compute_potential(np.array([[1,2,3.]], dtype=np.float32)) 

        # print(self.model.compute_acceleration.experimental_get_compiler_ir(x_tensor[0:1])(stage='hlo'))
        # print("\n\n\n")
        # print(self.model.compute_acceleration.experimental_get_compiler_ir(x_tensor[0:1])(stage='optimized_hlo'))
        # print("\n\n\n")
        # print(self.model.compute_acceleration.experimental_get_compiler_ir(x_tensor[0:1])(stage='optimized_hlo_dot'))

        if batch:
            def eval(x):
                self.model.compute_acceleration(x_tensor)
                # self.model.compute_potential(x_tensor)
        else:
            def eval(x):
                for i in range(len(x)):
                    self.model.compute_acceleration(x_tensor[i:i+1])
                    # self.model.compute_potential(x_tensor[i:i+1]) 

        # time calls
        start = time.time()
        eval(x)
        dt = time.time() - start

        print(f"Time per sample {dt / len(x)}")
        # try:
        #     print(self.model.compute_acceleration.pretty_printed_concrete_signatures())
        #     print(self.model.compute_potential.pretty_printed_concrete_signatures())
        # except:
        #     pass
        return dt / len(x)

    def run(self, batch):
        x = self.get_test_data()
        self.times = self.time_predictions(x, batch)
        return self.times


def main():
    import pandas as pd

    from GravNN.Analysis.TimePredictionExperiment import \
        TimePredictionExperiment
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle("Data/Dataframes/example.data")
    # df = pd.read_pickle("Data/Dataframes/network_size_test.data")
    model_id = df["id"].values[-1]
    config, pinn_model = load_config_and_model(model_id, df)
    print(f"Number of Units: {config['num_units'][0]}")
    print(f"Number of Params: {config['params'][0]}")
    r_max = config["radius_max"][0]
    r_min = config["radius_min"][0]

    # Batch vs Single
    # Numba vs No Numba
    # CPU vs GPU

    extrapolation_exp = TimePredictionExperiment(pinn_model, r_max, r_min, 10000)
    extrapolation_exp.run(batch=False)

    sh_model = SphericalHarmonics(config["grav_file"][0], degree=100, parallel=False)
    extrapolation_exp = TimePredictionExperiment(sh_model, r_max, r_min, 10000)
    extrapolation_exp.run(batch=False)


if __name__ == "__main__":
    main()
