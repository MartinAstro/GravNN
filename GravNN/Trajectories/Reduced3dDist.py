import os
import pathlib

import numpy as np
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase


class Reduced3dDist(TrajectoryBase):
    def __init__(self, celestial_body, radiusBounds, layers, degree, reduction=0.25):
        self.radiusBounds = radiusBounds
        self.degree = degree
        self.layers = layers
        self.reduction = reduction
        n =  2*degree + 2
        self.N_lon = int(2*n*self.reduction)
        self.N_lat = int(n*self.reduction)
        self.points = self.N_lon*self.N_lat*self.layers
        self.celestial_body = celestial_body
        super().__init__()
        pass

    def generate_full_file_directory(self):
        self.trajectory_name = os.path.splitext(os.path.basename(__file__))[0] +  "/"  + \
                self.celestial_body.body_name + \
             "_Deg" +   str(self.degree) + \
             "_RadBounds" + str(self.radiusBounds) + \
             "_Layers" + str(self.layers) + \
                "_Reduct" +  str(self.reduction)
        self.file_directory += self.trajectory_name +  "/"
        pass
    
    def generate(self):
        '''r ∈ [0, ∞), φ ∈ [-π/2, π/2],  θ ∈ [0, 2π)'''
        X = []
        Y = []
        Z = []
        idx = 0

        X.extend(np.zeros((self.N_lat*self.N_lon*self.layers,)).tolist())
        Y.extend(np.zeros((self.N_lat*self.N_lon*self.layers,)).tolist())
        Z.extend(np.zeros((self.N_lat*self.N_lon*self.layers,)).tolist())

        # Center the reduction
        phi = np.linspace(np.pi/2-(np.pi*self.reduction)/2, np.pi/2+(np.pi*self.reduction)/2, self.N_lat, endpoint=True)
        theta = np.linspace(np.pi-(2*np.pi*self.reduction)/2, np.pi+(2*np.pi*self.reduction)/2, self.N_lon, endpoint=True)
        r = np.linspace(self.radiusBounds[0], self.radiusBounds[1], self.layers, endpoint=True)

        # Shift to interesting feature
        theta -= np.pi/3 # Indonesia area
        for k in range(0, self.layers):
            for i in range(0, self.N_lon): #Theta Loop
                for j in range(0,self.N_lat):
                    X[idx] = r[k]*np.sin(phi[j])*np.cos(theta[i])
                    Y[idx] = r[k]*np.sin(phi[j])*np.sin(theta[i])
                    Z[idx] = r[k]*np.cos(phi[j])
                    idx += 1
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
