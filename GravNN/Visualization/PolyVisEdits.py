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

    def normalize_percent(self, x, min_percent, max_percent):
        # Clip the values to prescribed range
        x_normal = np.clip(x / 100, min_percent, max_percent)
        x_min = 0
        x_max = np.max(x_normal) * 100

        # scale the clipped values to 0 - 1
        scaler = MinMaxScaler((0, 1))
        x_normal = scaler.fit_transform(x_normal.reshape((-1, 1))).flatten()

        # If all values exceed the clip bounds,
        # ensure scalar forces to 1 rather than 0.
        if np.all(x_normal == 0.0):
            x_normal = np.ones_like(x_normal)
        return x_min, x_max, x_normal

    def normalize_acceleration(self, x, z_min, z_max):
        # Normalize the color from 0 - 1
        x_min = np.min(x) if z_min is None else z_min
        x_max = np.max(x) if z_max is None else z_max
        x_normal = (x - x_min) / (x_max - x_min)
        return x_min, x_max, x_normal

    def apply_colors(self, tri, cmap, x_norm):
        colors = cmap(x_norm)

        # place color on the collection
        tri.set_facecolor(colors / 255)
        tri.set_edgecolor(colors / 255)
        return tri

    def get_polyhedron(self, obj_file):
        # Load Mesh
        filename, file_extension = os.path.splitext(obj_file)
        mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])

        # Default cmap is Grey
        cmap = plt.get_cmap("Greys")
        colors = cmap(np.zeros((len(mesh.triangles),)) + 0.5)

        # Load collection
        tri = Poly3DCollection(
            mesh.triangles * 1000,
            cmap=cmap,
            alpha=0.4,
        )

        self.apply_colors(tri, colors)
        return mesh, tri

    def format_axes(self, ax, tri, mesh):
        ax.add_collection3d(tri)
        min_lim = np.min(mesh.vertices * 1000)
        max_lim = np.max(mesh.vertices * 1000)
        ax.axes.set_xlim3d(left=min_lim, right=max_lim)
        ax.axes.set_ylim3d(bottom=min_lim, top=max_lim)
        ax.axes.set_zlim3d(bottom=min_lim, top=max_lim)

    def add_colorbar(self, cmap, x_min, x_max, **kwargs):
        log = kwargs.get("log", False)
        if log:
            norm = SymLogNorm(linthresh=1e-4, vmin=x_min, vmax=x_max)
        else:
            norm = Normalize(vmin=x_min, vmax=x_max)

        arg = cm.ScalarMappable(norm=norm, cmap=cmap)

        cbformat = ticker.ScalarFormatter()
        cbformat.set_scientific("%.2e")
        cbformat.set_useMathText(True)
        cbformat.set_powerlimits((-2, 2))

        cbar_orientation = kwargs.get("cbar_orientation", "horizontal")
        label = kwargs.get("label", None)

        cBar = plt.colorbar(
            arg,
            pad=0.05,
            fraction=0.10,
            norm=norm,
            orientation=cbar_orientation,
            format=cbformat,
        )

        if label is not None:
            cBar.ax.set_xlabel(label)

    def plot_poly_errors(self, obj_file, errors, cbar=True, **kwargs):
        cmap = plt.get_cmap("RdYlGn")

        fig, ax = self.new3DFig()
        mesh, tri = self.get_polyhedron(self, obj_file)
        x_data = self.normalize_percent(errors, 0, 1)

        x_min = x_data[0]
        x_max = x_data[1]
        x_norm = x_data[2]

        tri = self.apply_colors(tri, cmap, x_norm)
        self.format_axes(ax, tri, mesh)
        if cbar:
            self.add_colorbar(cmap, x_min, x_max, **kwargs)

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
