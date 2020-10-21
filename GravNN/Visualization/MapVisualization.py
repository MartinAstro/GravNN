from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Support.transformations import sphere2cart
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import copy
import os
from enum import Enum
from matplotlib.colors import LogNorm, SymLogNorm

class MapVisualization(VisualizationBase):
    def __init__(self, unit='m/s^2'):
        super().__init__()
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        if unit == "mGal":
            # https://en.wikipedia.org/wiki/Gal_(unit)
            # 1 Gal == 0.01 m/s^2
            # 1 mGal == 1E-2 * 10E-3 = 10E-5 or 10000 mGal per m/s^2
            self.scale = 10000.0
        elif unit == "m/s^2":
            self.scale = 1.0
        pass

    def new_map(self, grid, vlim=None, log_scale=False):
        tick_interval = [30,30] # [45, 45]
        yticks = np.linspace(-90, 90, num=180//tick_interval[1]+1, endpoint=True, dtype=int)
        xticks = np.linspace(0, 360, num=360//tick_interval[0]+1, endpoint=True, dtype=int)
        xloc = np.linspace(0, len(grid)-1, num=len(xticks), endpoint=True, dtype=int)
        yloc = np.linspace(0, len(grid[0]), num=len(yticks), endpoint=True, dtype=int)
        
        # xticks_labels = [ r'$' + str(xtick) + r'\degree$' for xtick in xticks]
        # yticks_labels = [ r'$' + str(ytick) + r'\degree$' for ytick in yticks]
        xticks_labels = [ r'$' + str(xtick) + r'^\circ$' for xtick in xticks]
        yticks_labels = [ r'$' + str(ytick) + r'^\circ$' for ytick in yticks]

        ax = plt.gca()
        ax.set_xlabel("Longitude",  fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        plt.xticks(xloc,labels=xticks_labels, fontsize=6)
        plt.yticks(yloc, labels=yticks_labels, fontsize=6)
        # plt.xticks([])
        # plt.yticks([])
        grid = np.transpose(grid) # imshow takes (MxN)
        if log_scale:
            if vlim is not None:
                norm = SymLogNorm(linthresh=1E-4, vmin=vlim[0], vmax=vlim[1])
            else:
                norm = SymLogNorm(1E-4)
        else:
            norm = None

        if vlim is not None:
            im = plt.imshow(grid*self.scale, vmin=vlim[0], vmax=vlim[1], norm=norm)
        else:
            im = plt.imshow(grid*self.scale, norm=norm)
        return im


    def add_colorbar(self,im, label, vlim=None, extend='neither'):
        ax = plt.gca()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if vlim is not None:
            t = np.linspace(vlim[0], vlim[1], 5)
            cBar = plt.colorbar(im,  cax=cax, ticks=t, format='$%.4f$', extend=extend)
        else:
            cBar = plt.colorbar(im, cax=cax)
        cBar.ax.set_ylabel(label)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    def plot_grid(self, grid, label, vlim=None):
        fig,ax = self.newFig()
        im = self.new_map(grid, vlim)
        self.add_colorbar(im, label,vlim)
        return fig, ax
       
    def plot_grid_rmse(self, grid, grid_true, vlim=None, log_scale=False):
        error_grid = copy.deepcopy(grid)
        error_grid -= grid_true
        error_grid *= error_grid
        error_grid.total = np.sqrt(error_grid.total)
        fig, ax = self.newFig()
        im = self.new_map(error_grid.total, vlim, log_scale)
        self.add_colorbar(im,  'RSE [mGal]', vlim=vlim, extend='max')
        return fig, ax

