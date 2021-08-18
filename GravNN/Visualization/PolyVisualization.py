from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Support.transformations import sphere2cart
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import copy
import os
from enum import Enum
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from matplotlib import cm


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class PolyVisualization(VisualizationBase):
    def __init__(self, unit='m/s^2', **kwargs):
        """Visualization class for polyhedral shape models

        Args:
            unit (str, optional): acceleration unit ('m/s^2' or 'mGal'). Defaults to 'm/s^2'.
        """
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        if unit == "mGal":
            # https://en.wikipedia.org/wiki/Gal_(unit)
            # 1 Gal == 0.01 m/s^2
            # 1 mGal == 1E-2 * 10E-3 = 10E-5 or 10000 mGal per m/s^2
            self.scale = 10000.0
        elif unit == "m/s^2":
            self.scale = 1.0
        pass

    def plot_polyhedron(self, mesh, accelerations, label=None, vlim=None,log=False):
        fig, ax = self.new3DFig()
        cmap = plt.get_cmap('bwr')
        tri = Poly3DCollection(mesh.triangles*1000, cmap=cmap)
        vlim_min = np.min(accelerations)
        vlim_max = np.max(accelerations)
        scaled_acceleration = (accelerations - np.min(accelerations))/(np.max(accelerations) - np.min(accelerations))
        for i in range(len(mesh.faces)):
            color = np.array(cmap(int(scaled_acceleration[i]*255)))*255
            mesh.visual.face_colors[i] = color
        tri.set_facecolor(mesh.visual.face_colors/255)
        tri.set_edgecolor(mesh.visual.face_colors/255)
        p = ax.add_collection3d(tri)
        min_lim = np.min(mesh.vertices*1000)
        max_lim = np.max(mesh.vertices*1000)
        ax.axes.set_xlim3d(left=min_lim, right=max_lim) 
        ax.axes.set_ylim3d(bottom=min_lim, top=max_lim) 
        ax.axes.set_zlim3d(bottom=min_lim, top=max_lim) 
        
        if log:
            norm = SymLogNorm(linthresh=1E-4, vmin=vlim_min, vmax=vlim_max)
        else:
            norm = Normalize(vmin=vlim_min, vmax=vlim_max)
        arg = cm.ScalarMappable(norm=norm, cmap=cmap)
        ticks = np.linspace(vlim_min, vlim_max, 5)

        cBar = plt.colorbar(arg,  pad=0.20, fraction=0.15, norm=norm)#ticks=ticks,boundaries=vlim,
        if label is not None:
            cBar.ax.set_ylabel(label)

        return fig, ax

    def plot_position_data(self, data):
        x = data/1000.0
        ax = plt.gca()
        plt.gcf().axes[0].scatter(x[:,0], x[:,1], x[:,2],s=1)
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(x[:,0], x[:,1], x[:,2], s=1)