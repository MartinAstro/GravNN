import numpy as np
from numba import jit, prange, njit
from stl.mesh import Mesh as mesh_stl
import tinyobjloader
import trimesh
import matplotlib.pyplot as plt


class Polyhedral:
    def find_vertex(self, value_1, idx_1, value_2, idx_2):
        for i in range(len(self.mesh.vertices)):
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
    def __init__(self,obj_file, density): 

        self.mesh = trimesh.load_mesh(obj_file)
        self.density = density
        self.scaleFactor = 1E3    

        # v_1 = self.find_vertex(0.0, 0, 0.253214, 2)
        # v_2 = self.find_vertex(0.027957, 0, 0.263068, 2)
        # idx = self.find_edge(v_1,v_2)[0]

        # Compute Facet Dyads
        self.facet_dyads = np.zeros((len(self.mesh.faces),3,3))
        for i in range(len(self.mesh.faces)):
            facet_normal = self.mesh.face_normals[i]
            self.facet_dyads[i] = np.outer(facet_normal, facet_normal)
        
        # Compute Edge Dyads
        self.edge_dyads = np.zeros((len(self.mesh.edges_unique),3,3)) # In order of unique edges 
        for i in range(len(self.mesh.edges_unique)):
            vertex_0_idx = int(self.mesh.face_adjacency_edges[i][0])
            vertex_1_idx = int(self.mesh.face_adjacency_edges[i][1])
            vertex_0 = self.mesh.vertices[vertex_0_idx]
            vertex_1 = self.mesh.vertices[vertex_1_idx]

            facet_A_idx = int(self.mesh.face_adjacency[i][0])
            facet_B_idx = int(self.mesh.face_adjacency[i][1])
            normal_A = self.mesh.face_normals[facet_A_idx]
            normal_B = self.mesh.face_normals[facet_B_idx]

            face_A_vertices = self.mesh.faces[facet_A_idx]
            face_B_vertices = self.mesh.faces[facet_B_idx]
            face_A_c = (self.mesh.vertices[face_A_vertices[0]]+ \
                        self.mesh.vertices[face_A_vertices[1]]+\
                        self.mesh.vertices[face_A_vertices[2]])/3.0
            face_B_c = (self.mesh.vertices[face_B_vertices[0]]+ \
                        self.mesh.vertices[face_B_vertices[1]]+\
                        self.mesh.vertices[face_B_vertices[2]])/3.0
            
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

            self.edge_dyads[i] = dyad_A + dyad_B


    def GetPerformanceFactor(self, position, facet_idx):
        r0 = self.mesh.vertices[int(self.mesh.faces[facet_idx][0])]
        r1 = self.mesh.vertices[int(self.mesh.faces[facet_idx][1])]
        r2 = self.mesh.vertices[int(self.mesh.faces[facet_idx][2])]

        pos_scaled = position/self.scaleFactor
 
        r0m = r0 - pos_scaled
        r1m = r1 - pos_scaled
        r2m = r2 - pos_scaled

        R0 = np.linalg.norm(r0m)
        R1 = np.linalg.norm(r1m)
        R2 = np.linalg.norm(r2m)

        r1m_cross_r2m = np.cross(r1m, r2m)

        return 2.0*np.arctan2(np.dot(r0m,r1m_cross_r2m), R0*R1*R2 + R0*np.dot(r1m,r2m) + R1*np.dot(r0m,r2m) + R2*np.dot(r0m,r1m))

    def GetLe(self, position, edge_idx):
        r0 = self.mesh.vertices[int(self.mesh.edges_unique[edge_idx][0])]
        r1 = self.mesh.vertices[int(self.mesh.edges_unique[edge_idx][1])]

        pos_scaled = position/self.scaleFactor
        r0m = r0 - pos_scaled
        r1m = r1 - pos_scaled
        rem = r1m - r0m

        R0 = np.linalg.norm(r0m)
        R1 = np.linalg.norm(r1m)
        Re = np.linalg.norm(rem)

        return np.log((R0+R1+Re)/(R0+R1-Re))

    def compute_acceleration(self,position):
        G = 6.67408*1E-11 #m^3/(kg s^2)
        point_scaled = position/self.scaleFactor
        acc = np.zeros((3,))

        # Facet Loop
        for i in range(len(self.mesh.faces)):
            r0 = self.mesh.vertices[self.mesh.faces[i][0]]
            r0m = r0 - point_scaled

            wf = self.GetPerformanceFactor(position, i)
            F = self.facet_dyads[i]

            acc += wf*np.matmul(F, r0m)
        
        # Edge Loop
        for i in range(len(self.mesh.edges_unique)):
            r0 = self.mesh.vertices[self.mesh.edges_unique[i][0]]
            r0m = r0 - point_scaled

            Le = self.GetLe(position, i)
            E = self.edge_dyads[i]
            
            acc -= Le*np.matmul(E, r0m)
        
        acc *= G*self.density*self.scaleFactor
        return acc


def main():
    density = 1260.0 #kg/m^3 bennu https://github.com/bbercovici/SBGAT/blob/master/SbgatCore/include/SbgatCore/Constants.hpp s
    poly_model = Polyhedral("C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\ShapeModels\\Bennu-Radar.obj", density)
    position = np.array([1.,1.,1.])*1E3 # Must be in meters

    print(position)
    print(poly_model.compute_acceleration(position))
    print(np.linalg.norm(poly_model.compute_acceleration(position)))


if __name__ == '__main__':
    main()