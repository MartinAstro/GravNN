import time
import numpy as np
import tensorflow as tf

def print_jit_hlo_signatures(model_fcn, x_sample):
    print("HLO")
    print(model_fcn.experimental_get_compiler_ir(x_sample)(stage='hlo'))
    print("\n\n\n")
    print("HLO Optimized")
    print(model_fcn.experimental_get_compiler_ir(x_sample)(stage='optimized_hlo'))
    print("\n\n\n")
    print("HLO Optimized Dot")
    print(model_fcn.experimental_get_compiler_ir(x_sample)(stage='optimized_hlo_dot'))

def print_tf_function_signature(model_fcn):
    print(model_fcn.pretty_printed_concrete_signatures())


class TimePredictionExperiment:
    def __init__(self, model, r_max, r_min, points, fcn, input_type):
        self.model = model
        self.points = points
        self.r_min = r_min
        self.r_max = r_max
        self.fcn = self.initialize_fcn(fcn)
        self.input_type = input_type

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
        random_radius = np.random.uniform(self.r_min, self.r_max, size=(self.points, 1))
        x_test = random_unit * random_radius
        return x_test

    def enforce_type(self, model, x_input):
        try:
            x = x_input.astype(model.dtype)
        except:
            x = x_input.astype(np.float32)

        if self.input_type == "tensor":
            x = tf.constant(x)

        return x


    def time_predictions(self, x_input, batch):
        x = self.enforce_type(self.model, x_input)
        self.fcn(x[0:1,:])

        if batch:
            def eval(x):
                self.fcn(x)
        else:
            def eval(x):
                for i in range(len(x)):
                    self.fcn(x[i:i+1])

        # print_jit_hlo_signatures(self.fcn, x)
        # print_tf_function_signature(self.model)

        # time calls
        start = time.time()
        eval(x)
        dt = time.time() - start
        print(f"Time per sample {dt / len(x)}")
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
    compute_fcn = "acceleration"

    extrapolation_exp = TimePredictionExperiment(pinn_model, r_max, r_min, 10000, compute_fcn, input_type="numpy")
    extrapolation_exp.run(batch=False)

    sh_model = SphericalHarmonics(config["grav_file"][0], degree=100, parallel=False)
    extrapolation_exp = TimePredictionExperiment(sh_model, r_max, r_min, 10000, compute_fcn, input_type="numpy")
    extrapolation_exp.run(batch=False)


if __name__ == "__main__":
    main()
