import os

import numpy as np
import trimesh

from GravNN.Trajectories.TrajectoryBase import TrajectoryBase


class SurfaceDist(TrajectoryBase):
    def __init__(self, celestial_body, obj_file, **kwargs):
        """Distribution that generates samples from the center of each facet of a
        shape model.

        Args:
            celestial_body (CelestialBody): body from which points will be sampled
            obj_file (str): path to the file that contains the shape model
        """
        filename, file_extension = os.path.splitext(obj_file)
        self.mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
        self.points = len(self.mesh.faces)  # + self.mesh.vertices)
        self.celestial_body = celestial_body
        self.obj_file = obj_file
        super().__init__()

        pass

    def generate_full_file_directory(self):
        self.trajectory_name = (
            os.path.splitext(os.path.basename(__file__))[0]
            + "/"
            + self.celestial_body.body_name
            + "N_"
            + str(self.points)
        )
        self.file_directory += self.trajectory_name + "/"
        pass

    def generate(self):
        """Generate positions [m] of the center of each facet"""
        X = []
        Y = []
        Z = []
        X.extend(np.zeros((self.points,)).tolist())
        Y.extend(np.zeros((self.points,)).tolist())
        Z.extend(np.zeros((self.points,)).tolist())
        for i in range(len(self.mesh.faces)):
            face = self.mesh.faces[i]
            face_c = (
                (
                    self.mesh.vertices[face[0]]
                    + self.mesh.vertices[face[1]]
                    + self.mesh.vertices[face[2]]
                )
                / 3.0
                * 1e3
            )
            X[i] = face_c[0]
            Y[i] = face_c[1]
            Z[i] = face_c[2]

        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
