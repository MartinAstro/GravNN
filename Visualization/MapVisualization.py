from Visualization.VisualizationBase import VisualizationBase
from support.transformations import sphere2cart
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import copy
import os
class MapVisualization(VisualizationBase):
    def __init__(self):
        super().__init__()
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        pass

    def new_map(self, grid, vlim=None):
        tick_interval = [30,30]
        yticks = np.linspace(-90, 90, num=180//tick_interval[1]+1, endpoint=True, dtype=int)
        xticks = np.linspace(0, 360, num=360//tick_interval[0]+1, endpoint=True, dtype=int)
        xloc = np.linspace(0, len(grid), num=len(xticks), endpoint=True, dtype=int)
        yloc = np.linspace(0, len(grid[0]), num=len(yticks), endpoint=True, dtype=int)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.xticks(xloc,labels=xticks)
        plt.yticks(yloc, labels=yticks)
        grid = np.transpose(grid) # imshow takes (MxN)
        if vlim is not None:
            im = plt.imshow(grid, vmin=-vlim, vmax=vlim)
        else:
            im = plt.imshow(grid)
        return im


    def add_colorbar(self,im, label):
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cBar = plt.colorbar(im, cax=cax)
        cBar.ax.set_ylabel(label)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    def plot_grid(self, grid, label):
        fig,ax = self.newFig()
        im = self.new_map(grid)
        self.add_colorbar(im, label)
        return plt.gcf(), plt.gca()


    def plot_grid_error(self, grid, grid_true, vlim=None):
        error_grid = copy.deepcopy(grid)
        error_grid -= grid_true
        error_grid = error_grid/grid_true * 100.0
        fig, ax = self.newFig()
        im = self.new_map(error_grid, vlim)
        self.add_colorbar(im,  'Acceleration, $\%$ Error')
        return plt.gcf(), plt.gca()
    
    def plot_component_errors(self, deg_list, error_list, color):
        fig, ax = self.newFig()
        linetypes = ['solid', 'dashed', 'dotted', 'dashdot']
        for i in range(len(error_list[0])):
            plt.plot(deg_list, error_list[:,i], linestyle=linetypes[i], color=color) # total error
        
        plt.legend(['Total', 'r', 'theta', 'phi'])
        return  plt.gcf(), plt.gca()