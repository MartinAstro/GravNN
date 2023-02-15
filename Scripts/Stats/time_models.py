import pandas as pd
import numpy as np
from GravNN.Analysis.TimePredictionExperiment import \
    TimePredictionExperiment
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Planets import Earth


class TrivialNet:
    def __init__(self):
        self.W0 = np.random.normal(0,1, size=(10,3))
        self.W = np.random.normal(0,1, size=(8,10,10))
        self.B = np.random.normal(0,1, size=(8,10,))
        self.act = np.sin
        pass

    def compute_acceleration(self, x):
        h_i_m1 = self.W0@x.T
        for i in range(len(self.W)):
            h_i_m1 = self.act(self.W[i]@h_i_m1 + self.B[i])
        return h_i_m1

def time_networks(batch, points):
    df = pd.read_pickle("Data/Dataframes/network_size_test.data")
    r_min = Earth().radius
    r_max = Earth().radius*3
    pinn_df = pd.DataFrame(columns=['params', 'time [s]'])
    for i, model_id in enumerate(df['id'].values):
        config, pinn_model = load_config_and_model(model_id, df)
        extrapolation_exp = TimePredictionExperiment(pinn_model, r_max, r_min, points)
        times = extrapolation_exp.run(batch=batch)
        params = config['params'][0]
        pinn_df_i = pd.DataFrame({
            "params": params,
            "time [s]" : np.average(times)
            }, index=[i])
        pinn_df = pd.concat((pinn_df, pinn_df_i))
    return pinn_df

def time_spherical_harmonics(batch, points, parallel=False, jit=True):
    planet = Earth()
    r_min = Earth().radius
    r_max = Earth().radius*3
    degree_list = [2**i for i in range(0, 12)]
    sh_df = pd.DataFrame(columns=['params', 'time [s]'])
    for i,deg in enumerate(degree_list):
        sh_model = SphericalHarmonics(planet.EGM2008, degree=deg, parallel=parallel, jit=jit)
        extrapolation_exp = TimePredictionExperiment(sh_model, r_max, r_min, points)
        times = extrapolation_exp.run(batch=batch)
        params = deg*(deg+1)
        sh_df_i = pd.DataFrame({
            "params": params,
            "time [s]" : np.average(times)
            }, index=[i])
        sh_df = pd.concat((sh_df, sh_df_i))
    return sh_df

def main():
    batch, points, jit = True, 10000
    args = (
        (True, 10000, True),
        (False, 10000, True)
        (True, 10000, False),
        (False, 10000, False)
        )
    for batch, points, jit in zip(args):
        pinn_df = time_networks(batch=batch, points=points)
        pd.to_pickle(pinn_df,f"Data/Dataframes/pinn_model_times_{batch}_{points}_{jit}.data")
        
        sh_df = time_spherical_harmonics(batch=batch, points=points, jit=jit)
        pd.to_pickle(sh_df,f"Data/Dataframes/sh_model_times_{batch}_{points}_{jit}.data")


if __name__ == "__main__":
    main()
