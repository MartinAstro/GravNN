import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np

class DHGridDist(TrajectoryBase):
    def __init__(self, celestial_body, radius, degree):
        self.radius = radius
        self.degree = degree
        n =  2*degree + 2
        self.N_lon = 2*n
        self.N_lat = n
        self.points = self.N_lon*self.N_lat
        self.celestial_body = celestial_body
        super().__init__()
        pass

    def generate_full_file_directory(self):
        self.trajectory_name = os.path.splitext(os.path.basename(__file__))[0] +  "/"  + \
                self.celestial_body.body_name + \
             "_Deg" +   str(self.degree) + \
            "_Rad" + str(self.radius) 
        self.file_directory += self.trajectory_name +  "/"
        pass
    
    def generate(self):
        '''r ∈ [0, ∞), φ ∈ [-π/2, π/2],  θ ∈ [0, 2π)'''
        X = []
        Y = []
        Z = []
        idx = 0
        radTrue = self.radius
        X.extend(np.zeros((self.N_lat*self.N_lon,)).tolist())
        Y.extend(np.zeros((self.N_lat*self.N_lon,)).tolist())
        Z.extend(np.zeros((self.N_lat*self.N_lon,)).tolist())

        phi = np.linspace(0, np.pi, self.N_lat, endpoint=True)
        theta = np.linspace(0, 2*np.pi, self.N_lon, endpoint=True)
        for i in range(0, self.N_lon): #Theta Loop
            for j in range(0,self.N_lat):
                X[idx] = (radTrue)*np.sin(phi[j])*np.cos(theta[i])
                Y[idx] = (radTrue)*np.sin(phi[j])*np.sin(theta[i])
                Z[idx] = (radTrue)*np.cos(phi[j])
                idx += 1
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
