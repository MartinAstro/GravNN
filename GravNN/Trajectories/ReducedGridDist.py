import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np

class ReducedGridDist(TrajectoryBase):
    def __init__(self, celestial_body, radius, degree, reduction=0.25):
        self.radius = radius
        self.degree = degree
        self.reduction = reduction
        n =  2*degree + 2
        self.N_lon = int(2*n*self.reduction)
        self.N_lat = int(n*self.reduction)
        self.points = self.N_lat*self.N_lon
        super().__init__(celestial_body)
        pass

    def generate_full_file_directory(self):
        self.trajectory_name = os.path.splitext(os.path.basename(__file__))[0] +  "/"  + \
                self.celestial_body.body_name + \
             "_Deg" +   str(self.degree) + \
            "_Rad" + str(self.radius)  + \
                "_Reduct" +  str(self.reduction)
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

        # Center the reduction
        phi = np.linspace(np.pi/2-(np.pi*self.reduction)/2, np.pi/2+(np.pi*self.reduction)/2, self.N_lat, endpoint=True)
        theta = np.linspace(np.pi-(2*np.pi*self.reduction)/2, np.pi+(2*np.pi*self.reduction)/2, self.N_lon, endpoint=True)

        # Shift to interesting feature
        theta -= np.pi/3 # Indonesia area

        for i in range(0, self.N_lon): #Theta Loop
            for j in range(0,self.N_lat):
                X[idx] = (radTrue)*np.sin(phi[j])*np.cos(theta[i])
                Y[idx] = (radTrue)*np.sin(phi[j])*np.sin(theta[i])
                Z[idx] = (radTrue)*np.cos(phi[j])
                idx += 1
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
