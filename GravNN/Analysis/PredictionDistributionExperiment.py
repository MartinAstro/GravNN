from GravNN.Networks.utils import _get_loss_fcn
from GravNN.Networks.Data import DataSet
from GravNN.Trajectories.RandomDist import RandomDist
import numpy as np
import pandas as pd
from GravNN.Networks.Losses import *
from pprint import pprint
import matplotlib.pyplot as plt

class PredictionDistributionExperiment:
    def __init__(self, model, config, points, radius_bounds=None, random_seed=1234, remove_J2=False):
        self.config = config.copy()
        self.model = model
        self.points = points
        self.radius_bounds = radius_bounds
        self.remove_J2 = remove_J2
        self.distribution = config['distribution'][0]

        if self.radius_bounds is None:
            min_radius = self.config['radius_min'][0]
            max_radius = self.config['radius_max'][0]

            # If training data was augmented data take the largest radius
            augment_data = self.config.get('augment_data_config', [{}])[0]
            extra_max_radius = augment_data.get('radius_max', [0])[0]
            max_radius = np.max([max_radius, extra_max_radius])

            self.radius_bounds = [min_radius, max_radius]

        # attributes to be populated in run()
        self.positions = None
        self.accelerations = None
        self.potentials = None

        self.predicted_accelerations = None
        self.predicted_potentials = None

        np.random.seed(random_seed)

    def get_data(self):
        planet = self.config["planet"][0]
        grav_file = self.config["grav_file"][0]

        trajectory = self.distribution(planet, self.radius_bounds, self.points, **self.config)
        get_analytic_data_fcn = self.config['gravity_data_fcn'][0]
        
        x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(
            trajectory, grav_file, **self.config
        )

        self.positions = x_unscaled
        self.accelerations = a_unscaled
        self.potentials = u_unscaled

    def get_PINN_data(self):
        positions = self.positions
        self.predicted_accelerations =  self.model.compute_acceleration(positions).numpy().astype(float)
        self.predicted_potentials =  self.model.compute_potential(positions).numpy().astype(float)

    def get_J2_data(self):
        planet = self.config["planet"][0]
        grav_file = self.config["grav_file"][0]

        trajectory = RandomDist(planet, self.radius_bounds, self.points, **self.config)
        get_analytic_data_fcn = self.config['gravity_data_fcn'][0]
        config_mod = self.config.copy()
        config_mod['max_deg'] = [2]
        
        x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(
            trajectory, grav_file, **config_mod
        )

        if self.remove_J2:
            # Low Fidelity
            self.LF_accelerations = a_unscaled
            self.LF_potentials = u_unscaled
        else:
            self.LF_accelerations = np.zeros_like(x_unscaled)
            self.LF_potentials = np.zeros_like(x_unscaled[:,0:1])
        

    def run(self):
        self.get_data()
        self.get_PINN_data()
        self.get_J2_data()
        self.plot()

    def plot(self):
        plt.figure()
        da = self.accelerations - self.predicted_accelerations
        plt.subplot(3,1,1)
        plt.hist(da[:,0], 250)
        plt.subplot(3,1,2)
        plt.hist(da[:,1], 250)
        plt.subplot(3,1,3)
        plt.hist(da[:,2], 250)
        plt.xlabel("Acceleration")

        from GravNN.Support.transformations import project_acceleration
        da_hill = project_acceleration(self.positions, da)
        plt.figure()
        plt.subplot(3,1,1)
        plt.hist(da_hill[:,0], 250)
        plt.subplot(3,1,2)
        plt.hist(da_hill[:,1], 250)
        plt.subplot(3,1,3)
        plt.hist(da_hill[:,2], 250)
        plt.xlabel("Sphere Acceleration")

        plt.figure()
        plt.hist(self.potentials - self.predicted_potentials, 1000)
        plt.xlabel("Potentials")
        plt.show()



def main():
    from GravNN.Networks.Model import load_config_and_model
    df_file = "Data/Dataframes/example.data"
    df = pd.read_pickle(df_file)
    model_id = df.iloc[-1]["id"]
    config, model = load_config_and_model(model_id, df)
    exp = PredictionDistributionExperiment(model, config, 50000, remove_J2=True)
    exp.run()

    # plt.show()
if __name__ == "__main__":
    main()