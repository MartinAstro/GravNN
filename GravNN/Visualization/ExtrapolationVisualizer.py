from multiprocessing.sharedctypes import Value
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Support.transformations import sphere2cart, cart2sph
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import copy
import os
import pandas as pd
import seaborn as sns
import sigfig
from GravNN.Networks.Model import load_config_and_model


class ExtrapolationVisualizer(VisualizationBase):
    def __init__(self, experiment, **kwargs):
        super().__init__(**kwargs)
        plt.rc('font', size= 7.0)
        self.experiment = experiment
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.radius = self.experiment.config['planet'][0].radius
        self.training_bounds = np.array(self.experiment.training_bounds)
        self.max_idx = np.where(self.experiment.test_r_COM > self.training_bounds[1])[0][0]
        self.set_x_axis(kwargs.get('x_axis', 'dist_2_surf'))
        self.annotate = kwargs.get('annotate', True)
        self.plot_fcn = kwargs.get('plot_fcn', plt.plot)

    def set_x_axis(self, x_type):
        if x_type == "dist_2_COM":
            self.x_test = self.experiment.test_r_COM / self.radius
            self.x_train = self.experiment.train_r_COM / self.radius
            self.idx_test = self.experiment.test_dist_2_COM_idx
            self.idx_train = self.experiment.train_dist_2_COM_idx
            self.x_label = "Distance to COM [R]"
        elif x_type == 'dist_2_surf':
            self.x_test = self.experiment.test_r_surf / self.radius
            self.x_train = self.experiment.train_r_surf / self.radius
            self.idx_test = self.experiment.test_dist_2_surf_idx
            self.idx_train = self.experiment.train_dist_2_surf_idx
            self.x_label = "Distance to Surface [R]"

        else:
            raise ValueError()

    def annotate_metrics(self, values, xy=(0.3, 0.95)):
        rms_avg = sigfig.round(np.mean(values), sigfigs=2)
        rms_std = sigfig.round(np.std(values), sigfigs=2)
        rms_max = sigfig.round(np.max(values), sigfigs=2)
        metric_str = "%s ± %s (%s)" % (rms_avg, rms_std, rms_max)
        plt.annotate(metric_str, xy=xy, xycoords='axes fraction')

    def plot_histogram(self, x):
        ax = plt.twinx()
        plt.hist(x, 50, alpha=0.2)
        plt.ylabel('Frequency')
        ax.set_zorder(1)

    def plot(self, x, value, **kwargs):
        # compute trend lines
        def get_rolling_lines(data):
            df = pd.DataFrame(data=data, index=None)
            avg = df.rolling(50, 25).mean()
            std = df.rolling(50, 25).std()
            max = df.rolling(10, 10).max()
            return avg, std, max
        
        # sort entries
        avg_line, std_line, max_line = get_rolling_lines(value)
        
        self.newFig()
        plt.scatter(x, value, alpha=0.2, s=2)
        self.plot_fcn(x, avg_line)

        if kwargs.get("plot_std", True):
            y_std_upper = np.squeeze(avg_line + 1*std_line)
            y_std_lower = np.squeeze(avg_line - 1*std_line)
            plt.fill_between(x, y_std_lower, y_std_upper, color='C0', alpha=0.5)
            
        self.plot_fcn(x, max_line, color='red')
        
        training_bounds = self.training_bounds / self.radius
        plt.vlines(training_bounds[0], ymin=0, ymax=np.max(value), color='green')
        plt.vlines(training_bounds[1], ymin=0, ymax=np.max(value), color='green')
        plt.vlines(1, ymin=0, ymax=np.max(value), color='grey')
        if self.annotate:
            self.annotate_metrics(value)
        plt.tight_layout()

    def plot_interpolation_loss(self, **kwargs):
        self.plot(
            self.x_test[:self.max_idx],
            self.experiment.loss_acc[self.idx_test][:self.max_idx],
            **kwargs)       
        plt.gca().set_yscale('log')
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("Loss")
        plt.xlabel(self.x_label)
        plt.ylim([0,None])
        plt.xlim([0,2])
        self.plot_histogram(self.x_train)

    def plot_extrapolation_loss(self, **kwargs):
        self.plot(
            self.x_test,
            self.experiment.loss_acc[self.idx_test],
            **kwargs)       
        plt.gca().set_yscale('log')
        plt.ylabel("Loss")
        plt.xlabel(self.x_label)

    def plot_interpolation_rms(self, **kwargs):
        self.plot(
            self.x_test[:self.max_idx],
            self.experiment.losses['rms'][self.idx_test][:self.max_idx],
            **kwargs)       
        plt.gca().set_yscale('log')
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("RMS [$m/s^2$]")
        plt.xlabel(self.x_label)
        plt.ylim([0,None])
        self.plot_histogram(self.x_train)

    def plot_extrapolation_rms(self, **kwargs):
        self.plot(
            self.x_test,
            self.experiment.losses['rms'][self.idx_test],
            **kwargs)       
        plt.gca().set_yscale('log')
        plt.ylabel("RMS [$m/s^2$]")
        plt.xlabel(self.x_label)

    def plot_interpolation_percent_error(self, **kwargs):
        self.plot(
            self.x_test[:self.max_idx],
            self.experiment.losses['percent'][self.idx_test][:self.max_idx],
            **kwargs)       
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("Percent Error")
        plt.xlabel(self.x_label)
        plt.ylim([0,None])
        self.plot_histogram(self.x_train)

    def plot_extrapolation_percent_error(self, **kwargs):
        self.plot(
            self.x_test,
            self.experiment.losses['percent'][self.idx_test],
            **kwargs)       
        plt.ylabel("Percent Error")
        plt.xlabel(self.x_label)



    def plot_scatter_error(self):

        import OrbitalElements.orbitalPlotting as op
        print(len(self.experiment.losses['percent'][:self.max_idx]))
        error = np.clip(self.experiment.losses['percent'][:self.max_idx], 0, 10)
        error = self.experiment.losses['percent'][:self.max_idx]
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

def main():
    df = pd.read_pickle("Data/Dataframes/high_altitude_behavior.data")

    model_id = df["id"].values[-1] # with scaling
    config, model = load_config_and_model(model_id, df)

    # evaluate the error at "training" altitudes and beyond
    extrapolation_exp = ExtrapolationExperiment(model, config, 10000)
    extrapolation_exp.run()

    # visualize error @ training altitude and beyond
    vis = ExtrapolationVisualizer(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy)
    vis.plot_interpolation_percent_error()
    vis.plot_extrapolation_percent_error()
    vis.plot_interpolation_rms()
    vis.plot_extrapolation_rms()


if __name__ == "__main__":
    main()