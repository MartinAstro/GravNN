import numpy as np
from numba import jit, prange, njit
from stl.mesh import Mesh as mesh_stl
import tinyobjloader

class Polyhedral:
    def __init__(self,stl_file, density):

        '''
        faces_vertex_idx [[1,2,3][3,2,4],[...],...]
        edges = [[1,2],[2,3][3,1],...,[...]]
        faces_edge_idx [[1,3,2],[3,2,4],[5,2,1],...,[...]]
        '''
        self.mesh = mesh_stl.from_file(stl_file)


        from mpl_toolkits import mplot3d
        from matplotlib import pyplot

        # # Create a new plot
        # figure = pyplot.figure()
        # axes = mplot3d.Axes3D(figure)

        # # Load the STL files and add the vectors to the plot
        # axes.add_collection3d(mplot3d.art3d.Line3DCollection(self.mesh.vectors))
        # axes.plot3D([0,self.mesh.normals[0,0]], [0, self.mesh.normals[0,1]], [0,self.mesh.normals[0,2]], c='r')

        # # Auto scale to the mesh size
        # scale = self.mesh.points.flatten()
        # axes.auto_scale_xyz(scale, scale, scale)

        # # Show the plot to the screen
        # pyplot.show()


        self.density = density
        self.scaleFactor = 1.0#? I think this is for meters vs km scaling        


        all_vertices = np.concatenate((self.mesh.points[:,0:3], self.mesh.points[:,3:6], self.mesh.points[:,6:9]), axis=0)
        self.vertices = np.unique(all_vertices, axis=0)


        # Generate all vertex indices associated with each facet
        self.faces_vertex_idx = np.zeros((len(self.mesh.data), 3))
        for i in range(len(self.mesh.data)):
            facet = self.mesh.data[i][1]
            for j in range(0,3):
                self.faces_vertex_idx[i,j] = np.where(
                    (facet[j][0] == self.vertices[:,0])* \
                    (facet[j][1] == self.vertices[:,1])* \
                    (facet[j][2] == self.vertices[:,2]))[0][0]
            assert len(np.unique(self.faces_vertex_idx[i,:])) == 3

        # Define all unique edges by vertex indices
        all_edges = np.zeros((len(self.faces_vertex_idx)*3,2))
        for i in range(len(self.faces_vertex_idx)):
            face = self.faces_vertex_idx[i]
            for j in range(0,3):
                all_edges[3*i + j] = np.roll(face, j)[0:2]
        
        all_edges = np.sort(all_edges,axis=1)
        self.edges = np.unique(all_edges, axis=0)

        # Generate all edge indices associated with each facet
        self.faces_edge_idx = np.zeros((len(self.mesh.data), 3))
        for i in range(len(self.faces_vertex_idx)):
            vertices_idx = self.faces_vertex_idx[i]
            for j in range(0,3):
                edge = np.sort(np.roll(vertices_idx, j)[0:2])
                has_one_vertex = np.concatenate((
                    np.where(edge[0] == self.edges[:,0])[0],
                    np.where(edge[0] == self.edges[:,1])[0]))

                has_two_vertex = np.concatenate((
                    np.where(edge[1] == self.edges[:,0])[0],
                    np.where(edge[1] == self.edges[:,1])[0]))
                
                interesection = np.intersect1d(has_one_vertex, has_two_vertex)
                self.faces_edge_idx[i,j] = interesection[0]
            assert len(np.unique(self.faces_edge_idx[i,:])) == 3


        # Find the two facets connected by an edge (edge 1 spans facets [])
        self.edge_facet_idx = np.zeros((len(self.edges), 2))
        for i in range(len(self.edges)):
            facet_idx = np.concatenate((
                np.where(i == self.faces_edge_idx[:,0])[0],
                np.where(i == self.faces_edge_idx[:,1])[0],
                np.where(i == self.faces_edge_idx[:,2])[0]))
            self.edge_facet_idx[i] = facet_idx

        # Compute Edge Dyads
        self.edge_dyads = np.zeros((len(self.edges),3,3))
        for i in range(len(self.edges)):
            edge = self.edges[i]
            facet_A_idx = int(self.edge_facet_idx[i][0])
            facet_B_idx = int(self.edge_facet_idx[i][1])
            normal_A = self.mesh.normals[facet_A_idx]/np.linalg.norm(self.mesh.normals[facet_A_idx])
            normal_B = self.mesh.normals[facet_B_idx]/np.linalg.norm(self.mesh.normals[facet_B_idx])

            vertex_0_idx = int(edge[0])
            vertex_1_idx = int(edge[1])
            vertex_0 = self.vertices[vertex_0_idx]
            vertex_1 = self.vertices[vertex_1_idx]

            edge_direction = np.cross(normal_A, normal_B)
            edge_direction /= np.linalg.norm(edge_direction)

            v1_m_v0 = vertex_1 - vertex_0
            if np.dot(v1_m_v0, edge_direction) < 0:
                edge_direction *= -1.0

            edge_normal_A_to_B = -1.0*np.cross(normal_A, edge_direction)
            edge_normal_B_to_A = np.cross(normal_B, edge_direction)

            dyad_A = np.outer(normal_A, edge_normal_A_to_B)
            dyad_B = np.outer(normal_B, edge_normal_B_to_A)

            self.edge_dyads[i] = dyad_A + dyad_B

        # Compute Facet Dyads
        self.facet_dyads = np.zeros((len(self.mesh.data),3,3))
        for i in range(len(self.mesh.data)):
            facet_normal = self.mesh.normals[i]/np.linalg.norm(self.mesh.normals[i])
            self.facet_dyads[i] = np.outer(facet_normal, facet_normal)

    def GetPerformanceFactor(self, position, facet_idx):
        r0 = self.vertices[int(self.faces_vertex_idx[facet_idx][0])]
        r1 = self.vertices[int(self.faces_vertex_idx[facet_idx][1])]
        r2 = self.vertices[int(self.faces_vertex_idx[facet_idx][2])]

        pos_scaled = position/self.scaleFactor
 
        r0m = r0 - pos_scaled
        r1m = r1 - pos_scaled
        r2m = r2 - pos_scaled

        R0 = np.linalg.norm(r0m)
        R1 = np.linalg.norm(r1m)
        R2 = np.linalg.norm(r2m)

        r1m_cross_r2m = np.cross(r1m, r2m)

        return 2*np.arctan2(np.dot(r0m,r1m_cross_r2m), R0*R1*R2 + R0*np.dot(r1m,r2m) + R1*np.dot(r0m,r2m) + R2*np.dot(r0m,r1m))

    def GetLe(self, position, edge_idx):
        r0 = self.vertices[int(self.edges[edge_idx][0])]
        r1 = self.vertices[int(self.edges[edge_idx][1])]

        pos_scaled = position/self.scaleFactor
        r0m = r0 - pos_scaled
        r1m = r1 - pos_scaled
        rem = r1m - r0m

        R0 = np.linalg.norm(r0m)
        R1 = np.linalg.norm(r1m)
        Re = np.linalg.norm(rem)

        return np.log((R0+R1+Re)/(R0+R1-Re))

    def compute_acceleration(self,position):
        G = 6.67408*10E-11 #m^3/(kg s^2)
        point_scaled = position/self.scaleFactor
        acc = np.zeros((3,))

        # Facet Loop
        for i in range(len(self.mesh.data)):
            r0 = self.mesh.v0[i]
            r0m = r0 - point_scaled

            wf = self.GetPerformanceFactor(position, i)
            F = self.facet_dyads[i]

            acc += wf*np.matmul(F, r0m)
        
        # Edge Loop
        for i in range(len(self.edges)):
            #vertex0 = self.faces_vertex_idx[i][0]
            edge_vertex_idx = int(self.edges[i][0])
            r0 = self.vertices[edge_vertex_idx]
            r0m = r0 - point_scaled

            Le = self.GetLe(position, i)
            E = self.edge_dyads[i]
            
            acc -= Le*np.matmul(E, r0m)
        
        acc *= G*self.density*self.scaleFactor
        return acc


def main():
    density = 1260.0 #kg/m^3 bennu https://github.com/bbercovici/SBGAT/blob/master/SbgatCore/include/SbgatCore/Constants.hpp s
    poly_model = Polyhedral('C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\ShapeModels\\Radar-Bennu-Model.stl', density)
    position = np.array([10000,10000,10000]) # Must be in meters
    print(poly_model.compute_acceleration(np.array([10000,10000,10000])))
    print(poly_model.compute_acceleration(np.array([1000000,1000000,1000000])*10E6))

    print(np.linalg.norm(poly_model.compute_acceleration(np.array([10000,10000,10000]))))
    print(np.linalg.norm(poly_model.compute_acceleration(np.array([1000000,1000000,1000000]))))

    # print(poly_model.compute_acceleration(np.array([-10000,10000,10000])))
    # print(poly_model.compute_acceleration(np.array([10000,-10000,10000])))
    # print(poly_model.compute_acceleration(np.array([10000,10000,-10000])))

    # print(np.linalg.norm(poly_model.compute_acceleration(np.array([10000,10000,10000]))))
    # print(np.linalg.norm(poly_model.compute_acceleration(np.array([-10000,10000,10000]))))
    # print(np.linalg.norm(poly_model.compute_acceleration(np.array([10000,-10000,10000]))))
    # print(np.linalg.norm(poly_model.compute_acceleration(np.array([10000,10000,-10000]))))

if __name__ == '__main__':
    main()