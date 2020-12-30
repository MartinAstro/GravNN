import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.CelestialBodies.Asteroids import Bennu

from numba import jit, njit, prange

class PointMass(GravityModelBase):
    def __init__(self, celestial_body, trajectory=None): 
        super().__init__()
        self.celestial_body = celestial_body
        self.mu = celestial_body.mu

    def generate_full_file_directory(self):
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "_" + os.path.basename(self.obj_file).split('.')[0] + "/"
        pass


    def compute_acc(self, positions=None):
        "Compute the acceleration for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions

        self.accelerations = np.zeros(positions.shape)
        for i in range(len(self.accelerations)):
            self.accelerations[i] = self.compute_acceleration(positions[i])      
        
        return self.accelerations

    def compute_acceleration(self, position):
        G = 6.67408*1E-11 #m^3/(kg s^2)
        acc = -self.mu*np.square(np.pow(positions,-2))
        return acc


def main():
    density = 1260.0 #kg/m^3 bennu https://github.com/bbercovici/SBGAT/blob/master/SbgatCore/include/SbgatCore/Constants.hpp 
    import time 
    start = time.time()
    asteroid = Bennu()
    poly_model = Polyhedral(asteroid, asteroid.obj_file)
    print(time.time() - start)

    position = np.array([[1.,1.,1.],[1.,1.,1.]])*1E3 # Must be in meters

    print(position)
    start = time.time()
    print(poly_model.compute_acc(position))
    print(time.time() - start)


if __name__ == '__main__':
    main()
