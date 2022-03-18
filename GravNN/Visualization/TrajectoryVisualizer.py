from multiprocessing.sharedctypes import Value
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Support.transformations import sphere2cart, cart2sph
from GravNN.Networks.Data import get_raw_data
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import copy
import os
import pandas as pd
import seaborn as sns
import sigfig
import GravNN
import OrbitalElements.orbitalPlotting as op

class TrajectoryVisualizer(VisualizationBase):
    def __init__(self, experiment, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.shape_model = kwargs.get("shape_model", 
                                        os.path.dirname(GravNN.__file__) + 
                                        "/Files/ShapeModels/Misc/unit_sphere.obj")
        
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"

    def __annotate_metrics(self, values, xy=(0.3, 0.95)):
        rms_avg = sigfig.round(np.mean(values), sigfigs=2)
        rms_std = sigfig.round(np.std(values), sigfigs=2)
        rms_max = sigfig.round(np.max(values), sigfigs=2)
        metric_str = "%s ± %s (%s)" % (rms_avg, rms_std, rms_max)
        plt.annotate(metric_str, xy=xy, xycoords='axes fraction')

    def __plot_differences_1d(self, value):
        radii = np.linalg.norm(self.experiment.true_sol.y[0:3,:], axis=0)
        idx = np.argsort(radii)
        plt.figure()
        plt.plot(radii[idx], value[idx])
        plt.xlabel("Altitude [km]")

    def plot_trajectory_error(self):
        positions = self.experiment.test_sol.y[0:3,:]
        acc_percent_error = self.experiment.error_acc
        scale = np.max(acc_percent_error) - np.min(acc_percent_error)
        colors = plt.cm.RdYlGn(1 - ((acc_percent_error  - np.min(acc_percent_error)) / scale))   
        op.plot3d(positions, cVec=colors, obj_file=self.shape_model, plot_type='scatter')
    
    def plot_trajectories(self):
        op.plot3d(self.experiment.true_sol.y[0:3,:], plt.cm.Blues, obj_file=self.shape_model)
        op.plot3d(self.experiment.test_sol.y[0:3,:], plt.cm.Oranges, obj_file=self.shape_model, new_fig=False)


        
    def plot_acceleration_differences(self):
        self.__plot_differences_1d(self.experiment.error_acc)
        plt.yscale("log")
        plt.ylabel("Acceleration Percent Error")

    def plot_potential_differences(self):
        self.__plot_differences_1d(self.experiment.error_acc)
        plt.yscale("log")
        plt.ylabel("Potential Percent Error")


    def plot_trajectory_differences(self):
        plt.figure()
        plt.subplot(3,2,1)
        plt.plot(self.experiment.t_mesh, self.experiment.diff_sol.y[0])
        plt.ylabel('x')
        plt.subplot(3,2,3)
        plt.plot(self.experiment.t_mesh, self.experiment.diff_sol.y[1])
        plt.ylabel('y')
        plt.subplot(3,2,5)
        plt.plot(self.experiment.t_mesh, self.experiment.diff_sol.y[2])
        plt.ylabel('z')

        plt.subplot(3,2,2)
        plt.plot(self.experiment.t_mesh, self.experiment.diff_sol.y[3])
        plt.ylabel('vx')
        plt.subplot(3,2,4)
        plt.plot(self.experiment.t_mesh, self.experiment.diff_sol.y[4])
        plt.ylabel('vy')
        plt.subplot(3,2,6)
        plt.plot(self.experiment.t_mesh, self.experiment.diff_sol.y[5])
        plt.ylabel('vz')
        plt.suptitle("Trajectory Differences [m]")

    def plot_trajectories_1d(self):
        plt.figure()
        plt.subplot(3,2,1)
        plt.plot(self.experiment.t_mesh, self.experiment.true_sol.y[0])
        plt.plot(self.experiment.t_mesh, self.experiment.test_sol.y[0])
        plt.ylabel('x')
        plt.subplot(3,2,3)
        plt.plot(self.experiment.t_mesh, self.experiment.true_sol.y[1])
        plt.plot(self.experiment.t_mesh, self.experiment.test_sol.y[1])
        plt.ylabel('y')
        plt.subplot(3,2,5)
        plt.plot(self.experiment.t_mesh, self.experiment.true_sol.y[2])
        plt.plot(self.experiment.t_mesh, self.experiment.test_sol.y[2])
        plt.ylabel('z')

        plt.subplot(3,2,2)
        plt.plot(self.experiment.t_mesh, self.experiment.true_sol.y[3])
        plt.plot(self.experiment.t_mesh, self.experiment.test_sol.y[3])
        plt.ylabel('vx')
        plt.subplot(3,2,4)
        plt.plot(self.experiment.t_mesh, self.experiment.true_sol.y[4])
        plt.plot(self.experiment.t_mesh, self.experiment.test_sol.y[4])
        plt.ylabel('vy')
        plt.subplot(3,2,6)
        plt.plot(self.experiment.t_mesh, self.experiment.true_sol.y[5], label='True')
        plt.plot(self.experiment.t_mesh, self.experiment.test_sol.y[5], label='Test')
        plt.legend()
        plt.ylabel('vz')
        plt.suptitle("Trajectories [m]")



    def plot_scatter_error(self):
        import OrbitalElements.orbitalPlotting as op
        print(len(self.experiment.percent_error_acc[:self.max_idx]))
        error = np.clip(self.experiment.percent_error_acc[:self.max_idx], 0, 10)
        error = self.experiment.percent_error_acc[:self.max_idx]
        scale = np.max(error) - np.min(error)
        colors = plt.cm.RdYlGn(1 - ((error  - np.min(error)) / scale))   
        op.plot3d(self.experiment.positions[:self.max_idx].T, cVec=colors, obj_file=self.experiment.config['grav_file'][0], plot_type='scatter', alpha=0.2)

        # self.new3DFig()
        # x = (self.r / self.radius)

        # scale = np.max(diff_acc_mag_percent) - np.min(diff_acc_mag_percent)
        # colors = plt.cm.RdYlGn(1 - ((diff_acc_mag_percent  - np.min(diff_acc_mag_percent)) / scale))  
        # training_bounds = self.training_bounds / self.radius
        # x, y, z = self.experiment.positions
        # plt.scatter3d(x, y, z, alpha=0.2, s=2)
