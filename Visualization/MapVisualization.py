from Visualization.VisualizationBase import VisualizationBase
from Support.transformations import sphere2cart
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import copy
import os
from enum import Enum



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
            im = plt.imshow(grid, vmin=vlim[0], vmax=vlim[1])
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
        return fig, ax


    def plot_grid_error(self, grid, grid_true, vlim=None):
        error_grid = copy.deepcopy(grid)
        error_grid -= grid_true
        error_grid = error_grid/grid_true * 100.0
        fig, ax = self.newFig()
        im = self.new_map(error_grid, vlim)
        self.add_colorbar(im,  'Acceleration, $\%$ Error')
        return fig, ax
        
    def plot_component_errors(self, x_list, error_list, color):
        """Plot the errors of each component over some type of interval (be it SH degree or number of data points to train the NN). X-labels are specifically omitted and to be later specified by the user. 

        Args:
            x_list (np.array): intervals over which the errors are plotted  [Nx1]
            error_list (np.array): percent error for each component [Nx4]
            color ([str]): [color of lines to be plotted]

        """
        fig, ax = self.newFig()
        linetypes = ['solid', 'dashed', 'dotted', 'dashdot']
        for i in range(len(error_list[0])):
            plt.plot(x_list, error_list[:,i], linestyle=linetypes[i], color=color) # total error
        plt.legend([r'$r$', r'$\theta$', r'$\phi$', r'Total'])
        plt.ylabel(r"\% Error")
        return  plt.gcf(), plt.gca()


    def percent_maps(self, true_grid, grid, param="total", vlim=None):
        """
        Generates highest fidelity map of the planet gravity field provided. This includes the full map as well as the perturbations above C20.
        """
        percent_grid = (true_grid - grid)/true_grid*100
        fig,ax = self.newFig()
        if param == "total":
            data = percent_grid.total
        if param == "r":
            data = percent_grid.r
        if param == "theta":
            data = percent_grid.theta       
        if param == "phi":
            data = percent_grid.phi
        
        im = self.new_map(data, vlim)
        self.add_colorbar(im,  "Acceleration Perturbation \%")
        return plt.gcf(), plt.gca()

    def component_error(self,grid_list, true_grid, x_axis, color, C20_grid=None):
        """
        Shows the relative error of acceleration components for a particular model
        """
        error_list = np.zeros((len(grid_list),4))

        idx = 0
        for grid in grid_list:
            if C20_grid is not None:
                true_grid_final = true_grid - C20_grid
                grid_final = grid - C20_grid
            else:
                true_grid_final = true_grid 
                grid_final = grid 

            error_grid = (grid_final - true_grid_final)/true_grid_final * 100

            error_list[idx, 0] = np.average(abs(error_grid.r))
            error_list[idx, 1] = np.average(abs(error_grid.theta))
            error_list[idx, 2] = np.average(abs(error_grid.phi))
            error_list[idx, 3] = np.average(abs(error_grid.total))

            idx += 1

        fig, ax = self.plot_component_errors(x_axis, error_list, color)    
        return fig, ax