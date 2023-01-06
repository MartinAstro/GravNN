
from GravNN.GravityModels.Polyhedral import get_poly_data, Polyhedral
from GravNN.Support.transformations import cart2sph, project_acceleration
from GravNN.Trajectories import PlanesDist, SurfaceDist, RandomAsteroidDist
from GravNN.Networks.utils import _get_loss_fcn
from GravNN.Networks.Losses import get_loss_fcn
from GravNN.Networks.Data import DataSet

import numpy as np
import pandas as pd
import trimesh
import os
import GravNN

class PlanesExperiment:
    def __init__(self, model, config, bounds, samples_1d, **kwargs):
        self.config = config
        self.model = model
        self.bounds = np.array(bounds)
        self.samples_1d = samples_1d
        self.interior_mask = None

        self.brillouin_radius = config['planet'][0].radius
        original_max_radius = self.config['radius_max'][0]
        extra_max_radius = self.config.get('extra_radius_max', [0])[0]
        max_radius = np.max([original_max_radius, extra_max_radius])
        self.training_bounds = np.array([-max_radius, max_radius])

        # attributes to be populated in run()
        self.a_test = None
        self.u_test = None

        self.a_pred = None
        self.u_pred = None

        self.percent_error_acc = None
        self.percent_error_pot = None
        self.loss_fcn_list = ['rms', 'percent', 'angle', 'magnitude']


    def get_train_data(self):
        data = DataSet(self.config)
        self.x_train = data.raw_data['x_train']
        self.a_train = data.raw_data['a_train']
        
    def get_test_data(self):
        planet = self.config['planet'][0]
        obj_file = self.config.get('grav_file',[None])[0]
        gravity_data_fcn = self.config['gravity_data_fcn'][0]
        interpolation_dist = PlanesDist(planet, 
                            bounds=self.bounds,
                            samples_1d=self.samples_1d,
                            **self.config,
                            )

        full_dist = interpolation_dist
        
        x, a, u = gravity_data_fcn(full_dist, obj_file, **self.config)        

        self.x_test = x
        self.a_test = a
        self.u_test = u

    def get_model_data(self):
        try:
            dtype = self.model.network.compute_dtype
        except:
            dtype = float
        positions = self.x_test.astype(dtype)
        self.a_pred =  self.model.compute_acceleration(positions).astype(float)
        self.u_pred =  self.model.compute_potential(positions).numpy().astype(float)

    def compute_percent_error(self):
        def percent_error(x_hat, x_true):
            diff_mag = np.linalg.norm(x_true - x_hat, axis=1)
            true_mag = np.linalg.norm(x_true, axis=1)
            percent_error = diff_mag/true_mag*100
            return percent_error
        
        self.percent_error_acc = percent_error(self.a_pred, self.a_test)
        self.percent_error_pot = percent_error(self.u_pred, self.u_test)

    def compute_RMS(self):
        def RMS(x_hat, x_true):
            return np.sqrt(np.sum(np.square(x_true - x_hat), axis=1))
        
        self.RMS_acc = RMS(self.a_pred, self.a_test)
        self.RMS_pot = RMS(self.u_pred, self.u_test)

    def compute_losses(self, loss_fcn_list):
        losses = {}
        for loss_key in loss_fcn_list:
            loss_fcn = get_loss_fcn(loss_key)
            
            # Compute loss on acceleration and potential
            losses.update({
                f"{loss_fcn.__name__}" : loss_fcn(
                    self.a_pred, 
                    self.a_test
                    )
                })
        self.losses = losses

    def get_planet_mask(self):
        # Don't recompute this
        if self.interior_mask is None:
            grav_file =  self.config.get("grav_file", [None])[0] # asteroids grav_file is the shape model
            self.model_file = self.config.get("shape_model", [grav_file])[0] # planets have shape model (sphere currently)
            filename, file_extension = os.path.splitext(self.model_file)
            self.shape_model = trimesh.load_mesh(self.model_file, file_type=file_extension[1:])
            distances = self.shape_model.nearest.signed_distance(
                        self.x_test / 1e3
                    )
            self.interior_mask = distances > 0
        return self.interior_mask


    def run(self):
        self.get_train_data()
        self.get_test_data()
        self.get_model_data()
        self.compute_percent_error()
        self.compute_RMS()
        self.compute_losses(self.loss_fcn_list)


def main():
    import pandas as pd
    from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
    import matplotlib.pyplot as plt
    from GravNN.Networks.Model import load_config_and_model
    from GravNN.CelestialBodies.Asteroids import Eros
    df = pd.read_pickle("Data/Dataframes/test.data")
    model_id = df["id"].values[-1] 
    config, model = load_config_and_model(model_id, df)

    planet = config['planet'][0]
    planes_exp = PlanesExperiment(model, config, [-planet.radius, planet.radius], 30)
    planes_exp.run()

    vis = PlanesVisualizer(planes_exp)
    vis.plot(percent_max=10)

    planes_exp = PlanesExperiment(model, config, [-10*planet.radius, 10*planet.radius], 30)
    planes_exp.run()

    vis = PlanesVisualizer(planes_exp)
    vis.plot(percent_max=10)

    plt.show()
if __name__ == "__main__":
    main()