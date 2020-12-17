import os
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase

from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Networks import utils
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import sphere2cart, cart2sph, invert_projection, project_acceleration
import matplotlib.pyplot as plt
import numpy as np
class Plotting():
    def __init__(self, model_file, map_trajectories, x_transformer, a_transformer, config, directory):
        self.model_file = model_file
        self.map_trajectories = map_trajectories
        self.config = config
        self.x_transformer = x_transformer
        self.a_transformer = a_transformer
        self.directory = directory
        
    def plot_maps(self,model):
        for name, map_traj in self.map_trajectories.items():
            Call_r0_gm = SphericalHarmonics(self.model_file, degree=int(self.config['max_deg'][0]), trajectory=map_traj)
            Call_a = Call_r0_gm.load()
            
            Clm_r0_gm = SphericalHarmonics(self.model_file, degree=int(self.config['deg_removed'][0]), trajectory=map_traj)
            Clm_a = Clm_r0_gm.load()

            x = Call_r0_gm.positions # position (N x 3)
            a = Call_a - Clm_a

            if self.config['basis'][0] == 'spherical':
                x = cart2sph(x)
                a = project_acceleration(x, a)
                x[:,1:3] = np.deg2rad(x[:,1:3])
            
            x = self.x_transformer.transform(x)
            a = self.a_transformer.transform(a)

            U_pred, acc_pred = model.predict(x.astype('float32'))

            x = self.x_transformer.inverse_transform(x)
            a = self.a_transformer.inverse_transform(a)
            acc_pred = self.a_transformer.inverse_transform(acc_pred)

            if self.config['basis'][0] == 'spherical':
                x[:,1:3] = np.rad2deg(x[:,1:3])
                a = invert_projection(x, a)
                a_pred = invert_projection(x, acc_pred.astype(float))# numba requires that the types are the same 

            grid_true = Grid(trajectory=map_traj, accelerations=a)
            grid_pred = Grid(trajectory=map_traj, accelerations=acc_pred)
            diff = grid_pred - grid_true


            mapUnit = 'mGal'
            map_vis = MapVisualization(mapUnit)
            plt.rc('text', usetex=False)

            fig_true, ax = map_vis.plot_grid(grid_true.total, "True Grid [mGal]")
            fig_pred, ax = map_vis.plot_grid(grid_pred.total, "NN Grid [mGal]")
            fig_pert, ax = map_vis.plot_grid(diff.total, "Acceleration Difference [mGal]")

            fig, ax = map_vis.newFig(fig_size=(5*4,3.5*4))
            vlim = [0, np.max(grid_true.total)*10000.0] 
            plt.subplot(311)
            im = map_vis.new_map(grid_true.total, vlim=vlim, log_scale=False)
            map_vis.add_colorbar(im, '[mGal]', vlim)
            
            plt.subplot(312)
            im = map_vis.new_map(grid_pred.total, vlim=vlim, log_scale=False)
            map_vis.add_colorbar(im, '[mGal]', vlim)
            
            plt.subplot(313)
            im = map_vis.new_map(diff.total, vlim=vlim, log_scale=False)
            map_vis.add_colorbar(im, '[mGal]', vlim)

            if self.directory is not None:
                os.makedirs(self.directory + name, exist_ok=True)
                map_vis.save(fig_true, self.directory + name + "/true.pdf")
                map_vis.save(fig_pred, self.directory + name + "/pred.pdf")
                map_vis.save(fig_pert, self.directory + name + "/diff.pdf")
                map_vis.save(fig, self.directory + name + "/all.pdf")


    def plot_history(self, histories, labels):
        vis = VisualizationBase()
        fig, ax = vis.newFig()
        for i in range(len(histories)):
            plt.plot(histories[i].epoch[50:], histories[i].history['loss'][50:], label=labels[i])
            plt.plot(histories[i].epoch[50:], histories[i].history['val_loss'][50:], label=labels[i])

        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        if self.directory is not None:
            vis.save(fig, self.directory + "loss.pdf")

        

