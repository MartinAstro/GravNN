import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np
import trimesh

class SurfaceDist(TrajectoryBase):
    radiusBounds = None # [m]
    points = None  # Total points to distribute

    def __init__(self, celestial_body, obj_file):
        self.mesh = trimesh.load_mesh(obj_file)
        self.points = len(self.mesh.faces)
        super().__init__(celestial_body)

        pass

    def generate_full_file_directory(self):
        self.trajectory_name =  os.path.splitext(os.path.basename(__file__))[0] +  "/" + \
                                                self.celestial_body.body_name + \
                                                "N_" + str(self.points)
        self.file_directory  += self.trajectory_name +  "/"
        pass
    
    def generate(self):
        ''' position [m] of the center of each facet '''
        X = []
        Y = []
        Z = []
        X.extend(np.zeros((self.points,)).tolist())
        Y.extend(np.zeros((self.points,)).tolist())
        Z.extend(np.zeros((self.points,)).tolist())
        for i in range(len(self.mesh.faces)):
            face = self.mesh.faces[i]
            face_c = (self.mesh.vertices[face[0]]+ \
                    self.mesh.vertices[face[1]]+\
                    self.mesh.vertices[face[2]])/3.0*1E3 
            X[i] = face_c[0]
            Y[i] = face_c[1]
            Z[i] = face_c[2]

        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
