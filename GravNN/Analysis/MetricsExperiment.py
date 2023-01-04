from GravNN.Networks.utils import _get_loss_fcn
from GravNN.Networks.Data import DataSet
from GravNN.Trajectories.RandomDist import RandomDist
import numpy as np
import pandas as pd
from GravNN.Networks.Losses import *
from pprint import pprint

class MetricsExperiment:
    def __init__(self, model, config, points, radius_bounds=None, random_seed=1234):
        self.config = config.copy()
        self.model = model
        self.points = points
        self.radius_bounds = radius_bounds

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

        trajectory = RandomDist(planet, self.radius_bounds, self.points, **self.config)
        get_analytic_data_fcn = self.config['gravity_data_fcn'][0]
        
        x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(
            trajectory, grav_file, **self.config
        )

        self.positions = x_unscaled
        self.accelerations = a_unscaled
        self.potentials = u_unscaled

    def get_PINN_data(self):
        positions = self.positions
        self.predicted_accelerations =  self.model.compute_acceleration(positions).astype(float)
        self.predicted_potentials =  self.model.compute_potential(positions).numpy().astype(float)

    def compute_losses(self, loss_fcn_list):
        losses = {}
        for loss_key in loss_fcn_list:
            loss_fcn = get_loss_fcn(loss_key)
            
            # Compute loss on acceleration and potential
            losses.update({
                f"{loss_fcn.__name__}" : loss_fcn(
                    self.predicted_accelerations, 
                    self.accelerations
                    )
                })
        self.losses = losses
    
    def compute_metrics(self):
        metrics = {}
        for key, value in self.losses.items():
            metrics.update({
                f"{key}_mean" : np.mean(value),
                f"{key}_std" : np.std(value),
                f"{key}_max" : np.max(value),
            })
        self.metrics = metrics

    def run(self, loss_fcn_list):
        self.get_data()
        self.get_PINN_data()
        self.compute_losses(loss_fcn_list)
        self.compute_metrics()

    def save_metrics_in_df(self, df, i):
        # populate dataframe keys s.t. they can be updated
        for metrics_key in self.metrics.keys():
            if metrics_key not in df.columns:
                df[metrics_key] = None

        idx = df.index[i]
        new_col = pd.DataFrame(self.metrics, index=[idx])
        df.update(new_col)
        return df


def main():
    from GravNN.Networks.Model import load_config_and_model
    df_file = "Data/Dataframes/earth_PINN_III_max.data"
    df_file_new = df_file.split(".data")[0] + "_metrics.data"
    df = pd.read_pickle(df_file)

    for i in range(len(df)):
        model_id = df.iloc[i]["id"]
        config, model = load_config_and_model(model_id, df)
        config['override'] = [False]
        metrics_exp = MetricsExperiment(model, config, 50000)
        loss_fcn_list = ['rms', 'percent', 'angle', 'magnitude']
        metrics_exp.run(loss_fcn_list)
        pprint(metrics_exp.metrics)
        df = metrics_exp.save_metrics_in_df(df, i)

    df.to_pickle(df_file_new)

    # plt.show()
if __name__ == "__main__":
    main()