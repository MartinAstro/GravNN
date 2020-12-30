import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.CelestialBodies.Asteroids import Bennu

from numba import jit, njit, prange



def get_poly_data(trajectory, obj_file):
    Call_r0_gm = Polyhedral(trajectory.celestial_body, obj_file, trajectory=trajectory)
    accelerations = Call_r0_gm.load()

    Clm_r0_gm = PointMass(trajectory.celestial_body, trajectory=trajectory)
    accelerations_Clm = Clm_r0_gm.load()

    x = Call_r0_gm.positions # position (N x 3)
    a = accelerations - accelerations_Clm
    u = np.array([None for _ in range(len(a))]) # potential (N x 1)

    return u, x, a


@njit(cache=True, parallel=True)
def compute_edge_dyads(vertices, faces, edges_unique, face_adjacency_edges, face_normals, face_adjacency):
    edge_dyads = np.zeros((len(edges_unique),3,3)) # In order of unique edges 
    for i in prange(len(edges_unique)):
        vertex_0_idx = int(face_adjacency_edges[i][0])
        vertex_1_idx = int(face_adjacency_edges[i][1])
        vertex_0 = vertices[vertex_0_idx]
        vertex_1 = vertices[vertex_1_idx]

        facet_A_idx = int(face_adjacency[i][0])
        facet_B_idx = int(face_adjacency[i][1])
        normal_A = face_normals[facet_A_idx]
        normal_B = face_normals[facet_B_idx]

        face_A_vertices = faces[facet_A_idx]
        face_B_vertices = faces[facet_B_idx]
        face_A_c = (vertices[face_A_vertices[0]]+ \
                    vertices[face_A_vertices[1]]+\
                    vertices[face_A_vertices[2]])/3.0
        face_B_c = (vertices[face_B_vertices[0]]+ \
                    vertices[face_B_vertices[1]]+\
                    vertices[face_B_vertices[2]])/3.0
        
        B_2_A = face_A_c - face_B_c
        A_2_B = face_B_c - face_A_c

        edge_direction = np.cross(normal_A, normal_B)
        edge_direction /= np.linalg.norm(edge_direction)

        edge_normal_A_to_B = np.cross(normal_A, edge_direction)
        edge_normal_B_to_A = np.cross(normal_B, edge_direction)

        if np.dot(A_2_B, edge_normal_A_to_B) < 0:
            edge_normal_A_to_B *= -1.0
        if np.dot(B_2_A, edge_normal_B_to_A) < 0:
            edge_normal_B_to_A *= -1.0

        dyad_A = np.outer(normal_A, edge_normal_A_to_B)
        dyad_B = np.outer(normal_B, edge_normal_B_to_A)

        edge_dyads[i] = dyad_A + dyad_B
    return edge_dyads 

@njit(cache=True, parallel=True)
def compute_facet_dyads(face_normals):
    facet_dyads = np.zeros((len(face_normals),3,3))
    for i in prange(len(face_normals)):
        facet_normal = face_normals[i]
        facet_dyads[i] = np.outer(facet_normal, facet_normal)
    return facet_dyads

@njit(cache=True)
def GetPerformanceFactor(r_scaled, vertices, faces, facet_idx):
    r0 = vertices[int(faces[facet_idx][0])]
    r1 = vertices[int(faces[facet_idx][1])]
    r2 = vertices[int(faces[facet_idx][2])]

    r0m = r0 - r_scaled
    r1m = r1 - r_scaled
    r2m = r2 - r_scaled

    R0 = np.linalg.norm(r0m)
    R1 = np.linalg.norm(r1m)
    R2 = np.linalg.norm(r2m)

    r1m_cross_r2m = np.cross(r1m, r2m)

    return 2.0*np.arctan2(np.dot(r0m,r1m_cross_r2m), R0*R1*R2 + R0*np.dot(r1m,r2m) + R1*np.dot(r0m,r2m) + R2*np.dot(r0m,r1m))

@njit(cache=True)
def GetLe(r_scaled, vertices, edges_unique, edge_idx):
    r0 = vertices[int(edges_unique[edge_idx][0])]
    r1 = vertices[int(edges_unique[edge_idx][1])]

    r0m = r0 - r_scaled
    r1m = r1 - r_scaled
    rem = r1m - r0m

    R0 = np.linalg.norm(r0m)
    R1 = np.linalg.norm(r1m)
    Re = np.linalg.norm(rem)

    return np.log((R0+R1+Re)/(R0+R1-Re))

@njit(cache=True, parallel=True)
def facet_acc_loop(point_scaled, vertices, faces, facet_dyads):
    acc = np.zeros((3,))
    for i in prange(len(faces)):
        r0 = vertices[faces[i][0]]
        r0m = r0 - point_scaled

        wf = GetPerformanceFactor(point_scaled, vertices, faces, i)
        F = facet_dyads[i]

        acc += wf*np.dot(F, r0m)
    return acc 

@njit(cache=True,parallel=True)
def edge_acc_loop(point_scaled, vertices, edges_unique, edge_dyads):
    acc = np.zeros((3,))
    for i in prange(len(edges_unique)):
        r0 = vertices[edges_unique[i][0]]
        r0m = r0 - point_scaled

        Le = GetLe(point_scaled, vertices, edges_unique, i)
        E = edge_dyads[i]
        
        acc -= Le*np.dot(E, r0m)
    return acc

class Polyhedral(GravityModelBase):
    def __init__(self, celestial_body, obj_file, trajectory=None): 
        super().__init__()
        self.obj_file = obj_file

        self.configure(trajectory)

        self.density = celestial_body.density
        self.mesh = trimesh.load_mesh(obj_file)
        self.scaleFactor = 1E3 # Assume that the mesh is given in km 

        self.facet_dyads = compute_facet_dyads(self.mesh.face_normals)
        self.edge_dyads = compute_edge_dyads(self.mesh.vertices, 
                                            self.mesh.faces, 
                                            self.mesh.edges_unique, 
                                            self.mesh.face_adjacency_edges, 
                                            self.mesh.face_normals, 
                                            self.mesh.face_adjacency)

    def generate_full_file_directory(self):
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "_" + os.path.basename(self.obj_file).split('.')[0] + "/"
        pass

    # Debug functions
    def find_vertex(self, value_1, idx_1, value_2, idx_2):
        for i in range(len(self.vertices)):
            if np.isclose(self.mesh.vertices[i][idx_1], value_1) and np.isclose(self.mesh.vertices[i][idx_2], value_2):
                return i

    def find_edge(self, vertex_1_idx, vertex_2_idx):
        return np.intersect1d(np.where(self.mesh.edges_unique == vertex_1_idx)[0],np.where(self.mesh.edges_unique == vertex_2_idx)[0])

    def plot_geometry(self, edge, edge_direction, normal_A, normal_B, edge_normal_A_to_B, edge_normal_B_to_A):
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.quiver(0,0,0, edge[0], edge[1], edge[2],color='red')
        ax.quiver(0,0,0, normal_A[0], normal_A[1], normal_A[2],color='blue')
        ax.quiver(0,0,0, normal_B[0], normal_B[1], normal_B[2],color='cyan')
        ax.quiver(0,0,0, edge_normal_A_to_B[0], edge_normal_A_to_B[1], edge_normal_A_to_B[2],color='purple')
        ax.quiver(0,0,0, edge_normal_B_to_A[0], edge_normal_B_to_A[1], edge_normal_B_to_A[2],color='orange')
        ax.quiver(0,0,0, edge_direction[0], edge_direction[1], edge_direction[2],color='yellow')
        plt.show()

    # Bulk function 
    def compute_acc(self, positions=None):
        "Compute the acceleration for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions

        self.accelerations = np.zeros(positions.shape)
        for i in range(len(self.accelerations)):
            self.accelerations[i] = self.compute_acceleration(positions[i])      
        
        return self.accelerations

    def compute_acceleration(self, position):
        G = 6.67408*1E-11 #m^3/(kg s^2)
        point_scaled = position/self.scaleFactor
        acc = np.zeros((3,))

        acc_facet = facet_acc_loop(point_scaled, self.mesh.vertices, self.mesh.faces, self.facet_dyads)
        acc_edge = edge_acc_loop(point_scaled, self.mesh.vertices, self.mesh.edges_unique, self.edge_dyads)

        acc = acc_facet + acc_edge
        acc *= G*self.density*self.scaleFactor
        return acc


def main():
    density = 1260.0 #kg/m^3 bennu https://github.com/bbercovici/SBGAT/blob/master/SbgatCore/include/SbgatCore/Constants.hpp 
    import time 
    start = time.time()
    asteroid = Bennu()
    poly_model = Polyhedral(asteroid, asteroid.obj_file)
    print(time.time() - start)

    position = np.array([[1.,1.,1.],[1.,1.,1.]])*1E3 # Must be in meters

    print(position)
    start = time.time()
    print(poly_model.compute_acc(position))
    print(time.time() - start)


if __name__ == '__main__':
    main()
