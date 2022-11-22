from multiprocessing.sharedctypes import Value
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Support.transformations import sphere2cart, cart2sph
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter
from GravNN.CelestialBodies.Asteroids import Eros
import trimesh
import matplotlib

class PlanesVisualizer(VisualizationBase):
    def __init__(self, experiment, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.radius = self.experiment.config['planet'][0].radius
        self.training_bounds = np.array(self.experiment.training_bounds)
        self.planet = experiment.config['planet'][0]
        self.interior_mask = None

    def plane_mask(self, plane):
        mask = np.array([False, False, False])
        if 'x' in plane:
            mask = np.array([True, False, False]) + mask
        if 'y' in plane: 
            mask = np.array([False, True, False]) + mask
        if 'z' in plane: 
            mask = np.array([False, False, True]) + mask
        mask = mask.astype(bool)
        return mask 
    
    def get_plane_idx(self, plane):
        N = len(self.experiment.x_test)
        M = N // 3
        if plane == 'xy':
            idx_start = 0
            idx_end = M
        elif plane == 'xz':
            idx_start = M
            idx_end = 2*M
        elif plane == 'yz':
            idx_start = 2*M
            idx_end = 3*M
        else:
            raise ValueError(f"Invalid Plane: {plane}")
        return idx_start, idx_end

    def get_planet_mask(self):
        # Don't recompute this
        if self.interior_mask is None:
            grav_file =  self.experiment.config.get("grav_file", [None])[0] # asteroids grav_file is the shape model
            self.model_file = self.experiment.config.get("shape_model", [grav_file])[0] # planets have shape model (sphere currently)
            filename, file_extension = os.path.splitext(self.model_file)
            self.shape_model = trimesh.load_mesh(self.model_file, file_type=file_extension[1:])
            distances = self.shape_model.nearest.signed_distance(
                        self.experiment.x_test / 1e3
                    )
            self.interior_mask = distances > 0

    def set_SRP_contour(self, a_srp):
        self.r_srp = np.sqrt(self.planet.mu/a_srp)


    def plot_density_map(self, x_vec, plane='xy'):
        x, y, z = x_vec.T
        mask = self.plane_mask(plane)
        train_data = x_vec.T[mask]
        test_data = self.experiment.x_test.T[mask]
        k = gaussian_kde(train_data)
        nbins = 50

        x_0, x_1 = train_data[0], train_data[1]
        x_test_0, x_test_1 = test_data[0], test_data[1]
        
        x_0_i, x_1_i = np.mgrid[x_0.min():x_0.max():nbins*1j, x_1.min():x_1.max():nbins*1j]
        zi = k(np.vstack([x_0_i.flatten(), x_1_i.flatten()]))
        # plt.gca().pcolormesh(x_0_i, x_1_i, zi.reshape(x_0_i.shape))

        min_x_test_0 = np.min(x_test_0) / self.radius
        max_x_test_0 = np.max(x_test_0) / self.radius

        min_x_test_1 = np.min(x_test_1) / self.radius
        max_x_test_1 = np.max(x_test_1) / self.radius

        min_x_0 = np.min(x_0) / self.radius
        max_x_0 = np.max(x_0) / self.radius

        min_x_1 = np.min(x_1) / self.radius
        max_x_1 = np.max(x_1) / self.radius

        heatmap, xedges, yedges = np.histogram2d(x_0 / self.radius,
                                                 x_1 / self.radius,
                                                 nbins,
                                            # range=[
                                            #     [min_x_0, max_x_0],
                                            #     [min_x_1, max_x_1]
                                            #     ]
                                            range=[
                                                [min_x_test_0, max_x_test_0],
                                                [min_x_test_1, max_x_test_1]
                                                ]
                                                )
                                
        heatmap = gaussian_filter(heatmap, sigma=1)

        extent = [min_x_test_0, max_x_test_0, min_x_test_1, max_x_test_1]
        # extent = [min_x_0, max_x_0, min_x_1, max_x_1]
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet)
        cbar = plt.colorbar(fraction=0.15,)
        cbar.set_label('Samples')
        # plt.scatter(x_0 / self.radius, 
        #             x_1 / self.radius,
        #              s=5, c='r', alpha=0.5)
        # plt.scatter(x_0, x_1, c='r', alpha=1)
        # plt.xlim([min_x_test_0, max_x_test_0])
        # plt.ylim([min_x_test_1, max_x_test_1])

        plt.xlabel(plane[0])
        plt.ylabel(plane[1])        

    def plot_plane(self, x_vec, z_vec, plane='xy', nan_interior=True, colorbar_label=None, srp_sphere=False):
        
        mask = self.plane_mask(plane)
        idx_start, idx_end = self.get_plane_idx(plane)
        x = x_vec[idx_start:idx_end, mask]
        try:
            z = z_vec[idx_start:idx_end, mask]
        except:
            z = z_vec[idx_start:idx_end]

        if nan_interior:
            self.get_planet_mask()
            z[self.interior_mask[idx_start:idx_end]] = np.nan

        min_x_0 = np.min(x[:,0]) / self.radius
        max_x_0 = np.max(x[:,0]) / self.radius

        min_x_1 = np.min(x[:,1]) / self.radius
        max_x_1 = np.max(x[:,1]) / self.radius

        N = np.sqrt(len(z)).astype(int)
    
        im = plt.imshow(z.reshape((N,N)), extent=[min_x_0, max_x_0, min_x_1, max_x_1], origin='lower', cmap=cm.jet, vmin=0, vmax=self.max)
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.05)
        cBar = plt.colorbar(im, cax=cax)

        # cbar = plt.colorbar(fraction=0.15,)
        cBar.set_label(colorbar_label)
        plt.sca(plt.gcf().axes[0])

        plt.xlabel(plane[0])
        plt.ylabel(plane[1])

        if srp_sphere:
            circ = matplotlib.patches.Circle((0,0), self.r_srp/self.radius, fill=False, edgecolor='white')
            plt.gca().add_patch(circ)


    def plot(self, percent_max=100):
        self.max = percent_max
        plt.figure()
        plt.subplot(2,2,1)
        self.plot_density_map(self.experiment.x_train)

        x = self.experiment.x_test
        y = self.experiment.percent_error_acc
        cbar_label = "Acceleration Percent Error"
        plt.subplot(2,2,2)
        self.plot_plane(x,y, plane='xy', colorbar_label=cbar_label)
        plt.subplot(2,2,3)
        self.plot_plane(x,y, plane='xz', colorbar_label=cbar_label)
        plt.subplot(2,2,4)
        self.plot_plane(x,y, plane='yz', colorbar_label=cbar_label)
        plt.tight_layout()

    def plot_RMS(self, max=None):
        self.max = max
        plt.figure()
        plt.subplot(2,2,1)
        self.plot_density_map(self.experiment.x_train)

        x = self.experiment.x_test
        y = self.experiment.RMS_acc
        cbar_label = "Acceleration RMS Error"
        plt.subplot(2,2,2)
        self.plot_plane(x,y, plane='xy', colorbar_label=cbar_label)
        plt.subplot(2,2,3)
        self.plot_plane(x,y, plane='xz', colorbar_label=cbar_label)
        plt.subplot(2,2,4)
        self.plot_plane(x,y, plane='yz', colorbar_label=cbar_label)
        plt.tight_layout()

    def plot_scatter_error(self):
        import OrbitalElements.orbitalPlotting as op
        print(len(self.experiment.percent_error_acc[:self.max_idx]))
        error = np.clip(self.experiment.percent_error_acc[:self.max_idx], 0, 10)
        error = self.experiment.percent_error_acc[:self.max_idx]
        scale = np.max(error) - np.min(error)
        colors = plt.cm.RdYlGn(1 - ((error  - np.min(error)) / scale))   
        op.plot3d(self.experiment.positions[:self.max_idx].T, cVec=colors, obj_file=self.experiment.config['grav_file'][0], plot_type='scatter', alpha=0.2)

