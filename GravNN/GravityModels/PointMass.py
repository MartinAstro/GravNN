import os

import matplotlib.pyplot as plt
import numpy as np
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Support.transformations import sphere2cart, cart2sph, project_acceleration, invert_projection

from numba import jit, njit, prange

class PointMass(GravityModelBase):
    def __init__(self, celestial_body, trajectory=None): 
        super().__init__()
        self.configure(trajectory)

        self.celestial_body = celestial_body
        self.mu = celestial_body.mu

    def generate_full_file_directory(self):
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "_PointMass" + "/"
        pass


    def compute_acceleration(self, positions=None):
        "Compute the acceleration for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions

        positions = cart2sph(positions)
        self.accelerations = np.zeros(positions.shape)
        for i in range(len(self.accelerations)):
            self.accelerations[i] = self.compute_acceleration_value(positions[i])      
        
        # accelerations are in spherical coordinates in the hill frame. Need to change to inertial frame
        self.accelerations = invert_projection(positions,self.accelerations) 
        return self.accelerations

    def compute_potential(self, positions=None):
        "Compute the potential for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions

        positions = cart2sph(positions)
        self.potentials = np.zeros(len(positions))
        for i in range(len(self.potentials)):
            self.potentials[i] = self.compute_potential_value(positions[i])      
        
        return self.potentials

    def compute_acceleration_value(self, position):
        # remember that a = -dU/dx 
        # U = -mu/r
        # dU/dx = mu/r^2
        # a = -dU/dx = -mu/r^2
        return np.array([-self.mu/position[0]**2, 0, 0]) #[a_r, theta, phi] -- theta and phi are needed to convert back to cartesian

    def compute_potential_value(self, position):
        return -self.mu/position[0]

def main():
    import time 
    start = time.time()
    planet = Earth()
    point_mass = PointMass(planet)
    print(time.time() - start)

    position = np.array([[1.,0.,0.],[0.,1.,0.], [1., 1., 1.]])*planet.radius # Must be in meters

    print(position)
    start = time.time()
    accelerations = point_mass.compute_acceleration(position)
    print(accelerations)
    print(np.linalg.norm(accelerations, axis=1))
    print(time.time() - start)

    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
    model = SphericalHarmonics(planet.sh_hf_file, 1000)
    sh_results = model.compute_acceleration(position)

    print(sh_results)

if __name__ == '__main__':
    main()
