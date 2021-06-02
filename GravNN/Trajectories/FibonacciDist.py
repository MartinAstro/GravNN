import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np
from numba import njit, prange


@njit
def fibonacci_spiral_sphere(num_points, r):
    '''
    Distribute points evenly across the surface of a sphere
    https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
    http://www.math.vanderbilt.edu/saffeb/texts/161.pdf
    '''
    vectors = np.zeros((num_points, 3))
    gr=(np.sqrt(5.0) + 1.0) / 2.0  # golden ratio = 1.6180339887498948482
    ga=(2.0 - gr) * (2.0*np.pi)  # golden angle = 2.39996322972865332

    for i in prange(num_points):
        lat = np.arcsin(-1.0 + 2.0*i/(num_points))
        lon = ga * i

        x = r*np.cos(lon)*np.cos(lat)
        y = r*np.sin(lon)*np.cos(lat)
        z = r*np.sin(lat)

        vectors[i, :] = x, y, z
    return vectors


class FibonacciDist(TrajectoryBase):
    def __init__(self, celestial_body, radius, points):
        self.radius = radius
        self.celestial_body = celestial_body
        self.points = points
        super().__init__()
        pass

    def generate_full_file_directory(self):
        self.trajectory_name = os.path.splitext(os.path.basename(__file__))[0] +  "/"  + \
                self.celestial_body.body_name + \
                "_Points" +   str(self.points) + \
            "_Rad" + str(self.radius) 
        self.file_directory += self.trajectory_name +  "/"
        pass

    def generate(self):
        self.positions = fibonacci_spiral_sphere(self.points, self.radius)
        return self.positions 
