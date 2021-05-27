import os

import matplotlib.pyplot as plt
import numpy as np
from GravNN.Visualization.VisualizationBase import VisualizationBase
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        self.tick_interval = [30,30]
        self.unit = unit

    def new_map(self, grid, **kwargs):#vlim=None, log_scale=False, alpha=None, cmap=None):
        vlim = kwargs.get('vlim', None)
        log_scale = kwargs.get('log_scale', False)
        alpha = kwargs.get('alpha', None)
        cmap = kwargs.get('cmap', None)

        yticks = np.linspace(-90, 90, num=180//self.tick_interval[1]+1, endpoint=True, dtype=int)
        xticks = np.linspace(0, 360, num=360//self.tick_interval[0]+1, endpoint=True, dtype=int)
        
        xloc = np.linspace(0, len(grid)-1, num=len(xticks), endpoint=True, dtype=int)
        yloc = np.linspace(0, len(grid[0]), num=len(yticks), endpoint=True, dtype=int)

        xticks_labels = [ r'$' + str(xtick) + r'^\circ$' for xtick in xticks]
        yticks_labels = [ r'$' + str(ytick) + r'^\circ$' for ytick in yticks]

        ax = plt.gca()
        ax.set_xlabel("Longitude",  fontsize=11)
        ax.set_ylabel("Latitude",  fontsize=11)
        try:
            plt.xticks(xloc, labels=xticks_labels, fontsize=11)
            plt.yticks(yloc, labels=yticks_labels, fontsize=11)
        except:
            plt.xticks([])
            plt.yticks([])
        grid = np.transpose(grid) # imshow takes (MxN)
        if log_scale:
            if vlim is not None:
                norm = SymLogNorm(linthresh=1E-4, vmin=vlim[0], vmax=vlim[1])
            else:
                norm = SymLogNorm(1E-4)
        else:
            norm = None

        if vlim is not None:
            im = plt.imshow(grid*self.scale, vmin=vlim[0], vmax=vlim[1], norm=norm, alpha=alpha, cmap=cmap)
        else:
            im = plt.imshow(grid*self.scale, norm=norm, alpha=alpha, cmap=cmap)
        return im


    def add_colorbar(self,im, label, **kwargs):
        extend = kwargs.get('extend', 'neither')
        vlim = kwargs.get('vlim', None)
        ax = plt.gca()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if vlim is not None:
            t = np.linspace(vlim[0], vlim[1], 5)
            if self.unit == 'mGal':
                cBar = plt.colorbar(im, cax=cax, ticks=t, format='$%.2f$', extend=extend)
            else:
                cBar = plt.colorbar(im, cax=cax, ticks=t, format='$%.4f$', extend=extend)

        else:
            cBar = plt.colorbar(im, cax=cax)
        cBar.ax.set_ylabel(label)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    def plot_grid(self, grid, label, **kwargs): # vlim=None, cmap=None):
        fig, ax = self.newFig()
        im = self.new_map(grid, **kwargs)
        self.add_colorbar(im, label, **kwargs)
        return fig, ax
       
