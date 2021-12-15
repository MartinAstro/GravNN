import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from GravNN.CelestialBodies.Planets import Moon
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import (cart2sph,
                                            check_fix_radial_precision_errors)
from GravNN.Trajectories import DHGridDist, RandomDist
from GravNN.Visualization.MapVisualization import MapVisualization
from matplotlib.colors import ListedColormap


def main():

    N_data = 95000
    planet = Moon()
    trajectory = RandomDist(planet, [planet.radius, planet.radius+50000.0], 1000000)
    x_unscaled, a_unscaled, u_unscaled = get_sh_data(trajectory, planet.sh_file, deg_removed=2, max_deg=1000)

    vis = MapVisualization()
    vis.newFig()
    map_traj = DHGridDist(planet, planet.radius, degree=180)
    x_map, a_map, u_map = get_sh_data(map_traj, planet.sh_file, deg_removed=2, max_deg=1000)

    grid = Grid(trajectory=map_traj, accelerations=a_map)
    vis.new_map(grid.total)

    pos_sph = cart2sph(np.array(trajectory.positions))
    pos_sph = check_fix_radial_precision_errors(pos_sph)

    # Have to transform the original coordinates into the proper bin locations of the image (724, 362)
    longitude_bins = pos_sph[:,1]/360.0*724.0
    latitude_bins = pos_sph[:,2]/180.0*362.0

    # Choose colormap
    c = (pos_sph[:,0] - planet.radius)/50000 # Scale the altitude range into a color range. 
    cmap = pl.cm.Reds

    radial_samples = pos_sph[:N_data, 0]
    longitude_samples = longitude_bins[:N_data]
    latitude_samples = latitude_bins[:N_data]
    color_samples = cmap(c[:N_data])


    # Further filtering
    idx = np.where(radial_samples < planet.radius + 50000)
    print(len(idx[0]))
    print(len(idx[0])/N_data)
    longitude_samples = longitude_samples[idx]
    latitude_samples = latitude_samples[idx]
    color_samples =  color_samples[idx]

    plt.scatter(longitude_samples, latitude_samples, s=1, c=color_samples)

    plt.show()
main()
