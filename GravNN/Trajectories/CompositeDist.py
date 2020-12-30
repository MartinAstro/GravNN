import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np

class CompositeDist():
    def __init__(self):
        self.trajectories = []
        self.positions = np.array([],dtype=float)
        pass

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
        self.positions = np.append
    
    def generate(self):
        '''r ∈ [0, ∞), φ ∈ [-π/2, π/2],  θ ∈ [0, 2π)'''
        X = []
        Y = []
        Z = []
        idx = 0
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
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
