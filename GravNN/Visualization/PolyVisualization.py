import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import cm, ticker
from matplotlib.colors import Normalize, SymLogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.preprocessing import MinMaxScaler

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
        cmap="RdYlGn",
        cmap_reverse=True,
        percent=False,
        min_percent=0,
        max_percent=1,
        z_min=None,
        z_max=None,
        alpha=0.4,
        surface_colors=True,
        cbar_orientation="horizontal",
    ):
        fig, ax = self.new3DFig()

        filename, file_extension = os.path.splitext(obj_file)
        mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])

        if surface_colors and accelerations is not None:
            cmap = plt.get_cmap(cmap)

            if cmap_reverse:
                cmap = cmap.reversed()
            tri = Poly3DCollection(
                mesh.triangles * 1000,
                cmap=cmap,
                alpha=alpha,
            )

            if len(accelerations.shape) == 2:
                x = np.linalg.norm(accelerations, axis=1)
            else:
                x = accelerations

            if percent:
                # Clip the values to prescribed range

                x_min = min_percent * 100
                x_max = max_percent * 100
                x_normal = np.clip(x, x_min, x_max)  # Units of % (0 - 100 typically)
                x_range = np.array([[x_min, x_max]])

                if log:
                    x_normal = np.log10(x_normal)
                    x_range = np.log10(x_range)

                # scale the clipped values to 0 - 1
                scaler = MinMaxScaler((0, 1))
                scaler.fit(x_range.reshape((-1, 1)))
                x_normal = scaler.transform(x_normal.reshape((-1, 1))).flatten()

                # If all values exceed the clip bounds, ensure scalar forces to 1 rather
                # than 0.
                if np.all(x_normal == 0.0):
                    x_normal = np.ones_like(x_normal)
            else:
                # Normalize the color from 0 - 1
                x_min = np.min(x) if z_min is None else z_min
                x_max = np.max(x) if z_max is None else z_max
                x_normal = (x - x_min) / (x_max - x_min)

            mesh.visual.face_colors = cmap(x_normal)

            # place color on the collection
            tri.set_facecolor(mesh.visual.face_colors / 255)
            tri.set_edgecolor(mesh.visual.face_colors / 255)
        else:
            cmap = plt.get_cmap("Greys")
            tri = Poly3DCollection(
                mesh.triangles * 1000,
                cmap=cmap,
                alpha=alpha,
                # shade=True,
            )
            mesh.visual.face_colors = cmap(np.zeros((len(mesh.triangles),)) + 0.5)

            # place color on the collection
            tri.set_facecolor(mesh.visual.face_colors / 255)
            tri.set_edgecolor(mesh.visual.face_colors / 255)

        ax.add_collection3d(tri)
        min_lim = np.min(mesh.vertices * 1000)
        max_lim = np.max(mesh.vertices * 1000)
        ax.axes.set_xlim3d(left=min_lim, right=max_lim)
        ax.axes.set_ylim3d(bottom=min_lim, top=max_lim)
        ax.axes.set_zlim3d(bottom=min_lim, top=max_lim)

        vlim_min = x_min
        vlim_max = x_max

        if log:
            norm = SymLogNorm(linthresh=1e-4, vmin=vlim_min, vmax=vlim_max)
        else:
            norm = Normalize(vmin=vlim_min, vmax=vlim_max)

        arg = cm.ScalarMappable(norm=norm, cmap=cmap)
        if cbar and accelerations is not None:
            cbformat = ticker.ScalarFormatter()
            cbformat.set_scientific("%.2e")
            cbformat.set_useMathText(True)
            cbformat.set_powerlimits((-2, 2))

            cBar = plt.colorbar(
                arg,
                pad=0.05,
                fraction=0.10,
                norm=norm,
                orientation=cbar_orientation,
                format=cbformat,
            )  # ticks=ticks,boundaries=vlim,

            if label is not None:
                cBar.ax.set_xlabel(label)

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
    vis.plot_polyhedron(planet.obj_8k, acc, cmap="bwr")
    plt.show()
