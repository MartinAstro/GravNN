import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import cm
from matplotlib.colors import Normalize, SymLogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from GravNN.Visualization.VisualizationBase import VisualizationBase


class PolyVisualization(VisualizationBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        pass

    def plot_polyhedron(
        self,
        obj_file,
        accelerations=None,
        label=None,
        vlim=None,
        log=False,
        cbar=True,
        surface_colors=True,
    ):
        fig, ax = self.new3DFig()

        filename, file_extension = os.path.splitext(obj_file)
        mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])

        if surface_colors and accelerations is not None:
            cmap = plt.get_cmap("bwr")
            tri = Poly3DCollection(
                mesh.triangles * 1000,
                cmap=cmap,
                alpha=0.4,
                # shade=True,
            )

            x = np.linalg.norm(accelerations, axis=1)

            # Normalize the color from 0 - 1
            x_normal = (x - x.min()) / (x.max() - x.min())
            mesh.visual.face_colors = cmap(x_normal)

            # place color on the collection
            tri.set_facecolor(mesh.visual.face_colors / 255)
            tri.set_edgecolor(mesh.visual.face_colors / 255)
        else:
            cmap = plt.get_cmap("Greys")
            tri = Poly3DCollection(
                mesh.triangles * 1000,
                cmap=cmap,
                alpha=0.4,
                # shade=True,
            )

        ax.add_collection3d(tri)
        min_lim = np.min(mesh.vertices * 1000)
        max_lim = np.max(mesh.vertices * 1000)
        ax.axes.set_xlim3d(left=min_lim, right=max_lim)
        ax.axes.set_ylim3d(bottom=min_lim, top=max_lim)
        ax.axes.set_zlim3d(bottom=min_lim, top=max_lim)

        if cbar and accelerations is not None:
            vlim_min = np.min(x)
            vlim_max = np.max(x)
            if log:
                norm = SymLogNorm(linthresh=1e-4, vmin=vlim_min, vmax=vlim_max)
            else:
                norm = Normalize(vmin=vlim_min, vmax=vlim_max)

            arg = cm.ScalarMappable(norm=norm, cmap=cmap)
            cBar = plt.colorbar(
                arg,
                pad=0.20,
                fraction=0.15,
                norm=norm,
            )  # ticks=ticks,boundaries=vlim,
            if label is not None:
                cBar.ax.set_ylabel(label)

        return fig, ax

    def plot_position_data(self, data, alpha=1.0, color="blue"):
        x = data  # /1000.0
        plt.gca()
        plt.gcf().axes[0].scatter(x[:, 0], x[:, 1], x[:, 2], s=1, c=color, alpha=alpha)


if __name__ == "__main__":
    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.GravityModels.Polyhedral import Polyhedral
    from GravNN.Trajectories.SurfaceDist import SurfaceDist

    planet = Eros()
    traj = SurfaceDist(planet, planet.obj_8k)
    gravity_model = Polyhedral(planet, planet.obj_8k, traj).load()
    acc = gravity_model.accelerations

    vis = PolyVisualization()
    vis.plot_polyhedron(planet.obj_8k, acc)
    plt.show()
