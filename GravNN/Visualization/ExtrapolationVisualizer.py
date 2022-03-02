from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Support.transformations import sphere2cart, cart2sph
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import copy
import os
import pandas as pd
import seaborn as sns
import sigfig


class ExtrapolationVisualizer(VisualizationBase):
    def __init__(self, experiment, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"

        x_sph = cart2sph(self.experiment.positions)
        self.r = x_sph[:,0]

        self.radius = self.experiment.config['planet'][0].radius
        self.training_bounds = np.array(self.experiment.training_bounds)
        self.max_idx = np.where(self.r > self.training_bounds[1])[0][0]

    def annotate_metrics(self, values, xy=(0.3, 0.95)):
        rms_avg = sigfig.round(np.mean(values), sigfigs=2)#, notation='scientific')
        rms_std = sigfig.round(np.std(values), sigfigs=2)#, notation='scientific')
        rms_max = sigfig.round(np.max(values), sigfigs=2)#, notation='scientific')
        metric_str = "%s ± %s (%s)" % (rms_avg, rms_std, rms_max)
        plt.annotate(metric_str, xy=xy, xycoords='axes fraction')

    def plot_histogram(self, x):
        ax = plt.twinx()
        plt.hist(x, 50, alpha=0.2)
        plt.ylabel('Frequency')
        ax.set_zorder(1)

    def plot(self, value, avg_line, std_line, max_line):
        self.newFig()
        x = (self.r / self.radius)[:len(value)]
        training_bounds = self.training_bounds / self.radius

        plt.scatter(x, value, alpha=0.2, s=2)
        plt.plot(x, avg_line)

        y_std_upper = np.squeeze(avg_line + 1*std_line)
        y_std_lower = np.squeeze(avg_line - 1*std_line)
        plt.fill_between(x, y_std_lower, y_std_upper, color='C0', alpha=0.5)
        plt.plot(x, max_line, color='red')
        
        plt.vlines(training_bounds[0], ymin=0, ymax=np.max(value), color='green')
        plt.vlines(training_bounds[1], ymin=0, ymax=np.max(value), color='green')
        plt.vlines(1, ymin=0, ymax=np.max(value), color='grey')


    def plot_interpolation_loss(self):
        self.plot(self.experiment.loss_acc[:self.max_idx],
                    self.experiment.acc_avg_loss_line[:self.max_idx],
                    self.experiment.acc_std_loss_line[:self.max_idx],
                    self.experiment.acc_max_loss_line[:self.max_idx]
                    )
        plt.gca().set_yscale('log')
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("Loss")
        plt.xlabel("Radius")
        plt.ylim([0,None])
        self.annotate_metrics(self.experiment.loss_acc[:self.max_idx])
        self.plot_histogram(self.r / self.radius)


    def plot_extrapolation_loss(self):
        self.plot(self.experiment.loss_acc,
                    self.experiment.acc_avg_loss_line,
                    self.experiment.acc_std_loss_line,
                    self.experiment.acc_max_loss_line
                    )       
        plt.gca().set_yscale('log')
        plt.ylabel("Loss")
        plt.xlabel("Radius")
        self.annotate_metrics(self.experiment.loss_acc)

    def plot_interpolation_rms(self):
        self.plot(self.experiment.RMS_acc[:self.max_idx],
                    self.experiment.acc_avg_RMS_line[:self.max_idx],
                    self.experiment.acc_std_RMS_line[:self.max_idx],
                    self.experiment.acc_max_RMS_line[:self.max_idx]
                    )
        plt.gca().set_yscale('log')
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("Loss")
        plt.xlabel("Radius")
        plt.ylim([0,None])
        self.annotate_metrics(self.experiment.RMS_acc[:self.max_idx])
        self.plot_histogram(self.r / self.radius)


    def plot_extrapolation_rms(self):
        self.plot(self.experiment.RMS_acc,
                    self.experiment.acc_avg_RMS_line,
                    self.experiment.acc_std_RMS_line,
                    self.experiment.acc_max_RMS_line
                    )       
        plt.gca().set_yscale('log')
        plt.ylabel("Loss")
        plt.xlabel("Radius")
        self.annotate_metrics(self.experiment.RMS_acc)


    def plot_interpolation_percent_error(self):
        self.plot(self.experiment.percent_error_acc[:self.max_idx],
                    self.experiment.acc_avg_percent_error_line[:self.max_idx],
                    self.experiment.acc_std_percent_error_line[:self.max_idx],
                    self.experiment.acc_max_percent_error_line[:self.max_idx]
                    )       
        plt.xlim(self.training_bounds / self.radius)
        plt.ylabel("Percent Error")
        plt.xlabel("Radius")
        plt.ylim([0,None])
        self.annotate_metrics(self.experiment.percent_error_acc[:self.max_idx])
        self.plot_histogram((self.r / self.radius)[:self.max_idx])


    def plot_extrapolation_percent_error(self):
        self.plot(self.experiment.percent_error_acc,
            self.experiment.acc_avg_percent_error_line,
            self.experiment.acc_std_percent_error_line,
            self.experiment.acc_max_percent_error_line
            )       
        plt.ylabel("Percent Error")
        plt.xlabel("Radius")
        self.annotate_metrics(self.experiment.percent_error_acc)


