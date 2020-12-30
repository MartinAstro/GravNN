import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np

class UniformDist(TrajectoryBase):
    radius = None # [m]
    points = None # Total points to distribute

    def __init__(self, celestial_body, radius, points):
        if points % np.sqrt(points/2) != 0:
            print("The total number of points is not a perfect square")
            N = int(np.sqrt(points/2))
            points = 2*N**2
            print("The total number of points changed to " + str(points) + ". N = " +str(N))
        self.radius = radius
        self.points = points
        self.celestial_body = celestial_body
        super().__init__()
        pass

    def generate_full_file_directory(self):
        self.trajectory_name =   os.path.splitext(os.path.basename(__file__))[0] +  "/"  + \
            self.celestial_body.body_name + \
             "_N" +   str(self.points) + \
            "_Rad" + str(self.radius) 
        self.file_directory += self.trajectory_name + "/"
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

        N = int(np.sqrt(self.points/2))
        phi = np.linspace(0, np.pi, N, endpoint=True)
        #phi = np.linspace(-np.pi/2, np.pi/2, numPoints, endpoint=False)
        theta = np.linspace(0, 2*np.pi, 2*N, endpoint=True)
        for i in range(0,len(phi)): #Theta Loop
            for j in range(0, len(theta)):
                X[idx] = (radTrue)*np.sin(phi[i])*np.cos(theta[j])
                Y[idx] = (radTrue)*np.sin(phi[i])*np.sin(theta[j])
                Z[idx] = (radTrue)*np.cos(phi[i])
                idx += 1
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
