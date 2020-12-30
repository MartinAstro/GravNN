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

    def plot_polyhedron(self, mesh, accelerations, label=None, vlim=None):
        fig, ax = self.new3DFig()
        cmap = plt.get_cmap('bwr')
        tri = Poly3DCollection(mesh.triangles, cmap=cmap)
        for i in range(len(mesh.faces)):
            color = cmap(accelerations[i])[0]*255
            color[3] = 255
            mesh.visual.face_colors[i] = color
        tri.set_facecolor(mesh.visual.face_colors/255)
        tri.set_edgecolor(mesh.visual.face_colors/255)
        p = ax.add_collection3d(tri)
        min_lim = np.min(mesh.vertices)
        max_lim = np.max(mesh.vertices)
        ax.axes.set_xlim3d(left=min_lim, right=max_lim) 
        ax.axes.set_ylim3d(bottom=min_lim, top=max_lim) 
        ax.axes.set_zlim3d(bottom=min_lim, top=max_lim) 
        
        if vlim is not None:
            norm = Normalize(vmin=vlim[0], vmax=vlim[1])
            arg = cm.ScalarMappable(norm=norm, cmap=cmap)
            ticks = np.linspace(vlim[0], vlim[1], 5)
            cBar = plt.colorbar(arg,  pad=0.20, fraction=0.15, norm=norm)#ticks=ticks,boundaries=vlim,
            if label is not None:
                cBar.ax.set_ylabel(label)

        return fig, ax
