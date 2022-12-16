from GravNN.Networks.utils import _get_loss_fcn
from GravNN.Networks.Data import DataSet
from GravNN.Trajectories.RandomDist import RandomDist
import numpy as np
import pandas as pd

def percent_error(x_hat, x_true):
    diff_mag = np.linalg.norm(x_true - x_hat, axis=1)
    true_mag = np.linalg.norm(x_true, axis=1)
    percent_error = diff_mag/true_mag*100
    return percent_error

def RMS(x_hat, x_true):
    return np.sqrt(np.sum(np.square(x_true - x_hat), axis=1))

def compute_errors(y, y_hat):
    rms_error = np.square(y_hat - y)
    percent_error = np.linalg.norm(y - y_hat, axis=1) / np.linalg.norm(y, axis=1)*100
    return rms_error.astype(np.float32), percent_error.astype(np.float32)

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

        self.percent_error_acc = None
        self.percent_error_pot = None

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
        self.predicted_accelerations =  self.model.compute_acceleration(positions)
        self.predicted_potentials =  self.model.compute_potential(positions)

    def compute_percent_error(self):
        self.percent_error_acc = percent_error(self.predicted_accelerations, self.accelerations)
        self.percent_error_pot = percent_error(self.predicted_potentials, self.potentials)

    def compute_RMS(self):
        self.RMS_acc = RMS(self.predicted_accelerations, self.accelerations)
        self.RMS_pot = RMS(self.predicted_potentials, self.potentials)

    def compute_weighted_percent_error(self):
        rms_accelerations, percent_accelerations = compute_errors(self.accelerations, self.predicted_accelerations)
        rms_potentials, percent_potentials = compute_errors(self.potentials, self.predicted_potentials)
        w_i_acc = np.linalg.norm(rms_accelerations, axis=1) 
        w_i_pot = np.linalg.norm(rms_potentials, axis=1) 
        self.w_percent_error_acc = np.sum(percent_accelerations*w_i_acc)/np.sum(w_i_acc)
        self.w_percent_error_pot = np.sum(percent_potentials*w_i_pot)/np.sum(w_i_pot)

    def compute_loss(self):
        loss_fcn = _get_loss_fcn(self.config.get('loss_fcn',["percent_rms_summed"])[0])

        rms_accelerations, percent_accelerations = compute_errors(self.accelerations, self.predicted_accelerations) 
        self.loss_acc = np.array([
            loss_fcn(
                np.array([rms_accelerations[i]]), 
                np.array([percent_accelerations[i]])
                ) 
            for i in range(len(rms_accelerations)) 
            ])

        rms_potentials, percent_potentials = compute_errors(self.potentials, self.predicted_potentials) 
        self.loss_pot = np.array([
            loss_fcn(
                np.array([rms_potentials[i]]), 
                np.array([percent_potentials[i]])
                ) 
            for i in range(len(rms_potentials)) 
            ])
    
    def gather_metrics(self):
        data = {
            "RMS" : np.mean(self.RMS_acc),
            "percent_error" : np.mean(self.percent_error_acc),
            "w_percent_error" : float(self.w_percent_error_acc),
            "loss" : np.mean(self.loss_acc),
        }
        self.metrics = data

    def run(self):
        self.get_data()
        self.get_PINN_data()
        self.compute_percent_error()
        self.compute_RMS()
        self.compute_loss()
        self.compute_weighted_percent_error()
        self.gather_metrics()

    def save_metrics_in_df(self, df, i):
        idx = df.index[i]
        new_col = pd.DataFrame(self.metrics, index=[idx])
        df.update(new_col)
        return df


def main():
    from GravNN.Networks.Model import load_config_and_model
    df_file = "Data/Dataframes/earth_all.data"
    df_file_new = "Data/Dataframes/earth_all2.data"
    df = pd.read_pickle(df_file)
    df['RMS'] = None
    df['percent_error'] = None
    df['w_percent_error'] = None
    df['loss'] = None
    for i in range(len(df)):
        model_id = df.iloc[i]["id"]
        config, model = load_config_and_model(model_id, df)
        metrics_exp = MetricsExperiment(model, config, 500)
        metrics_exp.run()
        df = metrics_exp.save_metrics_in_df(df, i)

    df.to_pickle(df_file_new)

    # plt.show()
if __name__ == "__main__":
    main()