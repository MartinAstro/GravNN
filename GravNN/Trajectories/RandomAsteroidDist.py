import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np
import trimesh
from numba import njit, prange

# @njit(parallel=True, cache=True)
# def generate_points(radius_bounds)
#     phi = np.random.uniform(0, np.pi)
#     theta = np.random.uniform(0, 2*np.pi)
#     r = np.random.uniform(radius_bounds[0], radius_bounds[1])
#     X_inst = r*np.sin(phi)*np.cos(theta)
#     Y_inst = r*np.sin(phi)*np.sin(theta)
#     Z_inst = r*np.cos(phi)

class RandomAsteroidDist(TrajectoryBase):
    def __init__(self, celestial_body, radius_bounds, points, **kwargs):
        self.radius_bounds = radius_bounds
        self.shape_model = trimesh.load_mesh(kwargs['grav_file'][0])
        self.model_file = shape_model
        self.points = points
        self.celestial_body = celestial_body

        super().__init__()

        pass

    def generate_full_file_directory(self):
        self.trajectory_name =  os.path.splitext(os.path.basename(__file__))[0] +  "/" + \
                                                self.celestial_body.body_name + \
                                                "N_" + str(self.points) + \
                                                "_RadBounds" + str(self.radius_bounds) + \
                                                "_shape_model" +str(self.model_file.split('Blender_')[1]).split('.')[0]
        self.file_directory  += self.trajectory_name +  "/"
        pass
    
    def generate(self):
        '''r ∈ [0, ∞), φ ∈ [-π/2, π/2],  θ ∈ [0, 2π)'''
        X = []
        Y = []
        Z = []
        idx = 0
        X.extend(np.zeros((self.points,)).tolist())
        Y.extend(np.zeros((self.points,)).tolist())
        Z.extend(np.zeros((self.points,)).tolist())

        while idx < self.points:
            phi = np.random.uniform(0, np.pi)
            theta = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(self.radius_bounds[0], self.radius_bounds[1])
            X_inst = r*np.sin(phi)*np.cos(theta)
            Y_inst = r*np.sin(phi)*np.sin(theta)
            Z_inst = r*np.cos(phi)

            distance = self.shape_model.nearest.signed_distance(np.array([[X_inst, Y_inst, Z_inst]])/1E3)
            # ensure that the point is outside of the body
            while distance > 0:
                r = np.random.uniform(self.radius_bounds[0], self.radius_bounds[1])
                X_inst = r*np.sin(phi)*np.cos(theta)
                Y_inst = r*np.sin(phi)*np.sin(theta)
                Z_inst = r*np.cos(phi)
                distance = self.shape_model.nearest.signed_distance(np.array([[X_inst, Y_inst, Z_inst]])/1E3)

            # if distance > 0:
            #     continue
            #else:
            X[idx] = X_inst
            Y[idx] = Y_inst
            Z[idx] = Z_inst
            idx += 1
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
