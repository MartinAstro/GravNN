import os
from TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np

class RandomDist(TrajectoryBase):
    radiusBounds = None # [m]
    points = None  # Total points to distribute

    def __init__(self, celestial_body, radiusBounds, points):
        if points % np.sqrt(points) != 0:
            print("The total number of points is not a perfect square")
            exit()
        super().__init__(celestial_body)
        self.radiusBounds = radiusBounds
        self.points = points
        pass

    def generate_full_file_directory(self):
        self.file_directory  += os.path.splitext(os.path.basename(__file__))[0] +  "/" + \
                                                self.celestial_body.body_name + \
                                                "N_" + str(self.points) + \
                                                "_RadBounds" + str(self.radiusBounds) + "/"
        pass
    
    def generate(self):
        '''r ∈ [0, ∞), φ ∈ [-π/2, π/2],  θ ∈ [0, 2π)'''
        X = []
        Y = []
        Z = []
        idx = 0
        eps = 1E-2
        radTrue = self.radius
        X.extend(np.zeros((self.points,)).tolist())
        Y.extend(np.zeros((self.points,)).tolist())
        Z.extend(np.zeros((self.points,)).tolist())

        for i in range(self.points):
            phi = np.random.uniform(0, np.pi)
            theta = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(self.radiusBounds[0], self.radiusBounds[1])
            X[idx] = r*np.sin(phi)*np.cos(theta)
            Y[idx] = r*np.sin(phi)*np.sin(theta)
            Z[idx] = r*np.cos(phi)
            idx += 1
        return np.transpose(np.array([X, Y, Z]))
