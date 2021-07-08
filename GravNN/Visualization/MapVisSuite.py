import os

import matplotlib.pyplot as plt
import numpy as np
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from GravNN.Support.transformations import cart2sph, check_fix_radial_precision_errors
from matplotlib.cm import get_cmap
from GravNN.Visualization.FigureSupport import get_vlim_bounds

class MapVisSuite(MapVisualization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
    
    def true_pred_acc_pot_plots(self, grid_pot_true, grid_pot_pred, grid_acc_true, grid_acc_pred, title=None, percent=False, sigma_vlim=3):
        self.newFig()
        plt.subplot(2,3,1)
        self.plot_grid(grid_pot_true.total, "$m^2/s^2$", new_fig=False, ticks=False, loc='left', labels=False, title="$U$ True")

        ax = plt.gcf().axes[0]
        plt.subplot(2,3,2)
        self.plot_grid(grid_pot_pred.total, "$m^2/s^2$", new_fig=False, ticks=False, vlim=[ax.images[0].colorbar.vmin, ax.images[0].colorbar.vmax], colorbar=False, labels=False, title="$U$ NN")
        plt.subplot(2,3,3)
        if percent:
            diff = np.abs(((grid_pot_true - grid_pot_pred)/grid_pot_true).total)*100
            self.plot_grid(diff, "Percent", new_fig=False, ticks=False, vlim=get_vlim_bounds(diff, sigma_vlim), labels=False, format='$%.2f$', title="$U$ Difference", xlabel="$\delta U$: " +str(np.round(np.average(diff),2)))
        else:
            diff = (grid_pot_true - grid_pot_pred).total
            self.plot_grid(diff, "$m^2/s^2$", new_fig=False, ticks=False, vlim=get_vlim_bounds(diff, sigma_vlim), labels=False,title="$U$ Difference", xlabel="$\delta U$: " +str(np.round(np.average(diff),2)))

        plt.subplot(2,3,4)
        self.plot_grid(grid_acc_true.total, "$m/s^2$", new_fig=False, ticks=False,  loc='left', labels=False, title="$\mathbf{a}$ True")  
        ax = plt.gcf().axes[5] # the colorbars are their own axes and one is skipped in the middle figure
        plt.subplot(2,3,5)
        self.plot_grid(grid_acc_pred.total, "$m/s^2$", new_fig=False, ticks=False, vlim=[ax.images[0].colorbar.vmin, ax.images[0].colorbar.vmax], labels=False, colorbar=False,title="$\mathbf{a}$ NN")

        plt.subplot(2,3,6)
        if percent:
            diff = np.abs(((grid_acc_true - grid_acc_pred)/grid_acc_true).total)*100
            self.plot_grid(diff, "Percent", new_fig=False, ticks=False, vlim=get_vlim_bounds(diff, sigma_vlim), labels=False, format='$%.2f$', title="$\mathbf{a}$ Difference", xlabel="$\delta \mathbf{a}$: " +str(np.round(np.average(diff),2)))
        else:
            diff = (grid_acc_true - grid_acc_pred).total
            self.plot_grid(diff, "$m/s^2$", new_fig=False, ticks=False, vlim=get_vlim_bounds(diff, sigma_vlim), labels=False, title="$\mathbf{a}$ Difference", xlabel="$\delta \mathbf{a}$: " +str(np.round(np.average(diff),2)))
        plt.suptitle(title)


    def plot_acceleration_comp(self, grid_acc, title, ar_vlim=None, atheta_vlim=None, aphi_vlim=None):
        self.newFig()
        plt.subplot(3, 1, 1)
        self.plot_grid(grid_acc.r, r"$a_r$", new_fig=False, vlim=ar_vlim)   
        plt.subplot(3, 1, 2)
        self.plot_grid(grid_acc.theta, r"$a_{\theta}$", new_fig=False, vlim=atheta_vlim)   
        plt.subplot(3, 1, 3)
        self.plot_grid(grid_acc.phi, r"$a_{\phi}$", new_fig=False,  vlim=aphi_vlim)   
        plt.suptitle(title)
