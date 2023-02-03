import timeit
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

    def time_predictions(self, x):
        # warm start
        self.model.compute_acceleration(x[0:1]).astype(float)

        # time calls
        timer = timeit.Timer(
            stmt="self.model.compute_acceleration(x)", globals={"self": self, "x": x}
        )
        # timer = timeit.Timer(stmt="for i in range(len(x)): self.model.compute_acceleration(x[i:i+1])", globals={"self" : self, "x" : x})
        number = 100
        times = np.array(timer.repeat(repeat=5, number=number)) / number
        print(times)
        return times

    def run(self):
        x = self.get_test_data()
        self.time_predictions(x)


def main():
    import pandas as pd

    from GravNN.Analysis.TimePredictionExperiment import \
        TimePredictionExperiment
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle("Data/Dataframes/network_size_test.data")
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(model_id, df)
    print(f"Number of Units: {config['num_units'][0]}")
    print(f"Number of Params: {config['params'][0]}")
    r_max = config["radius_max"][0]
    r_min = config["radius_min"][0]

    extrapolation_exp = TimePredictionExperiment(model, r_max, r_min, 50)
    extrapolation_exp.run()

    model = SphericalHarmonics(config["grav_file"][0], degree=100, parallel=True)
    extrapolation_exp = TimePredictionExperiment(model, r_max, r_min, 50)
    extrapolation_exp.run()


if __name__ == "__main__":
    main()
