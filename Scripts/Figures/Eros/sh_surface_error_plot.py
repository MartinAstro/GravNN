from GravNN.Trajectories import SurfaceDist
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Visualization.PolyVisualization import PolyVisualization
import numpy as np
import matplotlib.pyplot as plt
import os
def main():
    directory = os.path.abspath('.') +"/Plots/Asteroid/"
    os.makedirs(directory, exist_ok=True)
    planet = Eros()
    trajectory = SurfaceDist(planet, planet.obj_8k)
    x_sh, a_sh, u_sh = get_sh_data(trajectory, planet.sh_file, max_deg=4, deg_removed=-1)
    x_poly, a_poly, u_poly = get_poly_data(trajectory, planet.obj_8k, remove_point_mass=[False])

    a_error = np.linalg.norm(a_sh - a_poly,axis=1)/np.linalg.norm(a_poly,axis=1)
    vis = PolyVisualization()
    vis.plot_polyhedron(trajectory.mesh, a_error, label='Acceleration Errors', log=True)
    vis.save(plt.gcf(), directory + "sh_surface_error.pdf")
    plt.show()

if __name__ == "__main__":
    main()