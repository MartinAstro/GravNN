
from GravNN.GravityModels.Polyhedral import get_poly_data, Polyhedral
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Trajectories import SurfaceDist, RandomDist
from GravNN.Networks.utils import _get_loss_fcn
from GravNN.Networks.Data import DataSet

import numpy as np
import pandas as pd
import trimesh
import os
import GravNN

class ExtrapolationExperiment:
    def __init__(self, model, config, points, random_seed=1234):
        self.config = config
        self.model = model
        self.points = points

        self.brillouin_radius = config['planet'][0].radius
        original_max_radius = self.config['radius_max'][0]
        extra_max_radius = self.config.get('extra_radius_max', [0])[0]
        max_radius = np.max([original_max_radius, extra_max_radius])
        self.training_bounds = [config['radius_min'][0], max_radius]

        # attributes to be populated in run()
        self.positions = None
        self.test_accelerations = None
        self.test_potentials = None

        self.predicted_accelerations = None
        self.predicted_potentials = None

        self.percent_error_acc = None
        self.percent_error_pot = None

        np.random.seed(random_seed)


    def get_train_data(self):

        data = DataSet(self.config)
        x_train = data.raw_data['x_train']
        a_train = data.raw_data['a_train']
        train_r_COM = cart2sph(x_train)[:,0] 

        # sort
        self.train_dist_2_COM_idx = np.argsort(train_r_COM)
        self.train_r_COM = train_r_COM[self.train_dist_2_COM_idx]

        grav_file =  self.config.get("grav_file", [None])[0] # asteroids grav_file is the shape model
        obj_file = self.config.get("shape_model", [grav_file])[0] # planets have shape model (sphere currently) 
         
        # Compute distance to surface
        filename, file_extension = os.path.splitext(obj_file)
        mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
        closest, train_r, triangle_id = trimesh.proximity.closest_point(mesh, x_train/ 1000)

        # Sort
        self.train_dist_2_surf_idx = np.argsort(train_r)
        self.train_r_surf = train_r[self.train_dist_2_surf_idx]*1000

    def get_test_data(self):
        planet = self.config['planet'][0]
        original_max_radius = self.config['radius_max'][0]
        augment_data = self.config.get('augment_data_config', [{}])[0]
        extra_max_radius = augment_data.get('radius_max', [0])[0]
        max_radius = np.max([original_max_radius, extra_max_radius])

        min_radius = self.config['radius_min'][0]
        obj_file = self.config.get('grav_file',[None])[0]

        gravity_data_fcn = self.config['gravity_data_fcn'][0]

        interpolation_dist = RandomDist(planet, 
                            radius_bounds=self.training_bounds,
                            points=self.points,
                            **self.config)
        extrapolation_dist = RandomDist(planet, 
                            radius_bounds=[max_radius, max_radius*10],
                            points=self.points,
                            **self.config)

        x, a, u = gravity_data_fcn(interpolation_dist, obj_file, **self.config)
        x_extra, a_extra, u_extra = gravity_data_fcn(extrapolation_dist, obj_file, **self.config)
        
        x = np.append(x, x_extra, axis=0)
        a = np.append(a, a_extra, axis=0)
        u = np.append(u, u_extra, axis=0)

        full_dist = interpolation_dist
        full_dist.positions = np.append(full_dist.positions, extrapolation_dist.positions, axis=0)
        

        # Compute distance to COM
        x_sph = cart2sph(x)
        self.test_dist_2_COM_idx = np.argsort(x_sph[:,0])
        self.test_r_COM = x_sph[self.test_dist_2_COM_idx,0]

        mesh = interpolation_dist.shape_model
        closest, test_r, triangle_id = trimesh.proximity.closest_point(mesh, x / 1000)

        # Sort
        self.test_dist_2_surf_idx = np.argsort(test_r)
        self.test_r_surf = test_r[self.test_dist_2_surf_idx]*1000

        self.positions = x
        self.test_accelerations = a
        self.test_potentials = u

    def get_PINN_data(self):
        positions = self.positions#.astype(self.model.network.compute_dtype)
        self.predicted_accelerations =  self.model.compute_acceleration(positions)
        self.predicted_potentials =  self.model.compute_potential(positions)

    def compute_percent_error(self):
        def percent_error(x_hat, x_true):
            diff_mag = np.linalg.norm(x_true - x_hat, axis=1)
            true_mag = np.linalg.norm(x_true, axis=1)
            percent_error = diff_mag/true_mag*100
            return percent_error
        
        self.percent_error_acc = percent_error(self.predicted_accelerations, self.test_accelerations)
        self.percent_error_pot = percent_error(self.predicted_potentials, self.test_potentials)

    def compute_RMS(self):
        def RMS(x_hat, x_true):
            return np.sqrt(np.sum(np.square(x_true - x_hat), axis=1))
        
        self.RMS_acc = RMS(self.predicted_accelerations, self.test_accelerations)
        self.RMS_pot = RMS(self.predicted_potentials, self.test_potentials)

    def compute_loss(self):

        def compute_errors(y, y_hat):
            rms_error = np.square(y_hat - y)
            percent_error = np.linalg.norm(y - y_hat, axis=1) / np.linalg.norm(y, axis=1)*100
            return rms_error.astype(np.float32), percent_error.astype(np.float32)

        loss_fcn = _get_loss_fcn(self.config.get('loss_fcn',["percent_rms_summed"])[0])

        rms_accelerations, percent_accelerations = compute_errors(self.test_accelerations, self.predicted_accelerations) 
        self.loss_acc = np.array([
            loss_fcn(
                np.array([rms_accelerations[i]]), 
                np.array([percent_accelerations[i]])
                ) 
            for i in range(len(rms_accelerations)) 
            ])

        rms_potentials, percent_potentials = compute_errors(self.test_potentials, self.predicted_potentials) 
        self.loss_pot = np.array([
            loss_fcn(
                np.array([rms_potentials[i]]), 
                np.array([percent_potentials[i]])
                ) 
            for i in range(len(rms_potentials)) 
            ])

    def run(self):
        self.get_train_data()
        self.get_test_data()
        self.get_PINN_data()
        self.compute_percent_error()
        self.compute_RMS()
        self.compute_loss()


def main():
    import pandas as pd
    from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
    import matplotlib.pyplot as plt
    from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle("Data/Dataframes/optimize_20.data")
    model_id = df["id"].values[-1] 
    config, model = load_config_and_model(model_id, df)
    extrapolation_exp = ExtrapolationExperiment(model, config, 500)
    extrapolation_exp.run()
    vis = ExtrapolationVisualizer(extrapolation_exp)
    vis.plot_interpolation_percent_error()
    vis.plot_extrapolation_percent_error()
    # vis.plot_interpolation_loss()
    plt.show()
if __name__ == "__main__":
    main()