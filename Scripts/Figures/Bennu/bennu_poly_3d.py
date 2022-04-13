        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import SurfaceDist

from GravNN.Visualization.PolyVisualization import PolyVisualization

import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

gradient = np.linspace(0, 1, 256)

def main():
    # https://www.asteroidmission.org/updated-bennu-shape-model-3d-files/
    planet = Bennu()
    obj_file = planet.obj_file
    sh_file = planet.sh_file
    density_deg = 180
    poly_vis = PolyVisualization()

    trajectory = SurfaceDist(planet, obj_file)
    poly_gm = Polyhedral(planet, obj_file, trajectory)
    acc_poly = poly_gm.load().accelerations

    # max_deg = 37
    # Call_r0_gm = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    # acc_sh = Call_r0_gm.load().accelerations

    # diff = acc_poly - acc_sh

    # Polyhedral Results
    totals = np.linalg.norm(acc_poly, axis=1).reshape((-1,1))
    vlim = [np.min(totals), np.max(totals)]
    minmax = MinMaxScaler()

    # Truth Model
    totals_normalized = minmax.fit_transform(totals)
    poly_vis.plot_polyhedron(poly_gm.mesh, totals_normalized, vlim)

    # # Spherical Harmonic Model
    # totals = np.linalg.norm(acc_sh, axis=1).reshape((-1,1))
    # totals_normalized = minmax.transform(totals)
    # poly_vis.plot_polyhedron(poly_gm.mesh, totals_normalized, vlim)

    # # Difference 
    # totals = np.linalg.norm(diff, axis=1).reshape((-1,1))
    # totals_normalized = minmax.transform(totals)
    # poly_vis.plot_polyhedron(poly_gm.mesh, totals_normalized, vlim)
    plt.show()


if __name__ == "__main__":
    main()
