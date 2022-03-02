
from GravNN.GravityModels.Polyhedral import get_poly_data, Polyhedral
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Trajectories import SurfaceDist, RandomAsteroidDist

import numpy as np
import pandas as pd


class ExtrapolationExperiment:
    def __init__(self, model, config, points, loss_type='rms', random_seed=1234):
        self.config = config
        self.model = model
        self.points = points
        self.loss_type = loss_type

        self.brillouin_radius = config['planet'][0].radius
        self.training_bounds = [config['radius_min'][0], config['radius_max'][0]]

        # attributes to be populated in run()
        self.positions = None
        self.truth_accelerations = None
        self.truth_potentials = None

        self.predicted_accelerations = None
        self.predicted_potentials = None

        self.percent_error_acc = None
        self.percent_error_pot = None

        self.acc_avg_line = None
        self.acc_std_line = None

        self.pot_avg_line = None
        self.pot_std_line = None

        np.random.seed(random_seed)


    def get_truth_data(self):
        planet = self.config['planet'][0]
        max_radius = self.config['radius_max'][0]
        obj_file = self.config['grav_file'][0]
        gravity_data_fcn = self.config['gravity_data_fcn'][0]

        interpolation_dist = RandomAsteroidDist(planet, 
                            radius_bounds=[0, max_radius],
                            points=self.points,
                            model_file=obj_file)
        extrapolation_dist = RandomAsteroidDist(planet, 
                            radius_bounds=[max_radius, max_radius*10],
                            points=self.points,
                            model_file=obj_file)

        full_dist = interpolation_dist
        full_dist.positions = np.append(full_dist.positions, extrapolation_dist.positions, axis=0)

        x, a, u = gravity_data_fcn(full_dist, obj_file, **self.config)

        x_sph = cart2sph(x)
        sorted_idx = np.argsort(x_sph[:,0])

        self.positions = x[sorted_idx]
        self.truth_accelerations = a[sorted_idx]
        self.truth_potentials = u[sorted_idx]

    def get_PINN_data(self):
        positions = self.positions.astype(np.float32)
        self.predicted_accelerations =  self.model.generate_acceleration(positions)
        self.predicted_potentials =  self.model.generate_potential(positions)

    def compute_percent_error(self):
        def percent_error(x_hat, x_true):
            diff_mag = np.linalg.norm(x_true - x_hat, axis=1)
            true_mag = np.linalg.norm(x_true, axis=1)
            percent_error = diff_mag/true_mag*100
            return percent_error
        
        self.percent_error_acc = percent_error(self.predicted_accelerations, self.truth_accelerations)
        self.percent_error_pot = percent_error(self.predicted_potentials, self.truth_potentials)

    def compute_RMS(self):
        def RMS(x_hat, x_true):
            return np.sqrt(np.sum(np.square(x_true - x_hat), axis=1))
        
        self.RMS_acc = RMS(self.predicted_accelerations, self.truth_accelerations)
        self.RMS_pot = RMS(self.predicted_potentials, self.truth_potentials)

    def compute_loss(self):

        def rms_percent_loss(x_hat, x_true):
            """ RMS + Percent Error """
            diff_mag = np.linalg.norm(x_true - x_hat, axis=1)
            true_mag = np.linalg.norm(x_true, axis=1)
            percent_error = diff_mag/true_mag
            loss_error = np.sum(np.square(x_true - x_hat),axis=1)
            loss_error += percent_error
            return loss_error

        def percent_loss(x_hat, x_true):
            """ RMS + Percent Error """
            diff_mag = np.linalg.norm(x_true - x_hat, axis=1)
            true_mag = np.linalg.norm(x_true, axis=1)
            percent_error = diff_mag/true_mag
            return percent_error

        def rms_loss(x_hat, x_true):
            """ RMS """
            return np.sqrt(np.sum(np.square(x_true - x_hat), axis=1))
        
        if self.loss_type == "rms_percent":
            loss = rms_percent_loss
        elif self.loss_type == 'rms':
            loss = rms_loss
        elif self.loss_type == 'percent':
            loss = percent_loss
        else: 
            raise NotImplementedError()

        self.loss_acc = loss(self.predicted_accelerations, self.truth_accelerations)
        self.loss_pot = loss(self.predicted_potentials, self.truth_potentials)


    def compute_trend_lines(self):

        def get_rolling_lines(data):
            df = pd.DataFrame(data=data, index=None)
            avg = df.rolling(50, 25).mean()
            std = df.rolling(50, 25).std()
            max = df.rolling(10, 10).max()
            return avg, std, max
        
        acc_avg, acc_std, acc_max = get_rolling_lines(self.percent_error_acc)
        pot_avg, pot_std, pot_max = get_rolling_lines(self.percent_error_pot)

        self.acc_avg_percent_error_line = acc_avg
        self.acc_std_percent_error_line = acc_std
        self.acc_max_percent_error_line = acc_max

        self.pot_avg_percent_error_line = pot_avg
        self.pot_std_percent_error_line = pot_std
        self.pot_max_percent_error_line = pot_max

        acc_avg, acc_std, acc_max = get_rolling_lines(self.loss_acc)
        pot_avg, pot_std, pot_max = get_rolling_lines(self.loss_pot)

        self.acc_avg_loss_line = acc_avg
        self.acc_std_loss_line = acc_std
        self.acc_max_loss_line = acc_max

        self.pot_avg_loss_line = pot_avg
        self.pot_std_loss_line = pot_std
        self.pot_max_loss_line = pot_max

        acc_avg, acc_std, acc_max = get_rolling_lines(self.RMS_acc)
        pot_avg, pot_std, pot_max = get_rolling_lines(self.RMS_pot)

        self.acc_avg_RMS_line = acc_avg
        self.acc_std_RMS_line = acc_std
        self.acc_max_RMS_line = acc_max

        self.pot_avg_RMS_line = pot_avg
        self.pot_std_RMS_line = pot_std
        self.pot_max_RMS_line = pot_max


    def run(self):
        self.get_truth_data()
        self.get_PINN_data()
        self.compute_percent_error()
        self.compute_RMS()
        self.compute_loss()
        self.compute_trend_lines()
