import copy
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from numba import njit

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.GravityModels.PointMass import PointMass
from GravNN.Support.PathTransformations import make_windows_path_posix


def get_poly_data(trajectory, obj_mesh_file, **kwargs):
    override = bool(kwargs.get("override", [False])[0])
    remove_point_mass = bool(kwargs.get("remove_point_mass", [False])[0])

    obj_mesh_file = make_windows_path_posix(obj_mesh_file)

    poly_r0_gm = Polyhedral_2(
        trajectory.celestial_body,
        obj_mesh_file,
        trajectory=trajectory,
    )
    poly_r0_gm.load(override=override)

    x = poly_r0_gm.positions  # position (N x 3)
    a = poly_r0_gm.accelerations
    u = poly_r0_gm.potentials  # potential (N,)

    # TODO: Determine if this is valuable -- how do dynamics and representation change
    # inside brillouin sphere
    if remove_point_mass:
        point_mass_r0_gm = PointMass(trajectory.celestial_body, trajectory=trajectory)
        point_mass_r0_gm.load(override=override)
        a_pm = point_mass_r0_gm.accelerations
        u_pm = point_mass_r0_gm.potentials

        a = a - a_pm
        u = u - u_pm

    return x, a, u


@njit(cache=True, parallel=False)
def get_values(faces, vertices, point_scaled):
    U = np.float64(0.0)
    acc = np.zeros((3,), dtype=np.float64)
    for face_idx, face in enumerate(faces):
        i, j, k = face[0:3]  # vertex index
        # vertex locations
        vertices = vertices
        r_i, r_j, r_k = vertices[face[0:3], :] - point_scaled

        # Edges
        e_1 = r_j - r_i
        e_2 = r_k - r_j

        # print("Max length between edges:", np.max(e_1)*1e3, np.max(e_2)*1e3)

        # Facet normal
        n_f = np.cross(e_1, e_2)
        n_f /= np.linalg.norm(n_f)

        for edge_idx in range(3):
            if edge_idx == 0:
                idx_min = i if i < j else j
                r1, r2 = r_i, r_j
            if edge_idx == 1:
                idx_min = j if j < k else k
                r1, r2 = r_j, r_k
            if edge_idx == 2:
                idx_min = i if i < k else k
                r1, r2 = r_k, r_i

            a = np.linalg.norm(r1)
            b = np.linalg.norm(r2)
            r_e = vertices[idx_min] - point_scaled

            # compute the edge normal
            r21 = r2 - r1
            e = np.linalg.norm(r21)
            n21 = np.cross(r21, n_f)
            n21 /= np.linalg.norm(n21)

            # compute the edge dyad
            Ee = np.outer(n_f, n21)

            # Compute Edge Performance Factor
            # Le = np.log((a + b + e) / (a + b - e))
            Le = np.log(a + b + e) - np.log(a + b - e)

            # Add to edge acceleration and potential
            acc -= Le * Ee @ r_e
            U += Le * r_e @ Ee @ r_e

        # compute the solid angle for the facet
        R1 = np.linalg.norm(r_i)
        R2 = np.linalg.norm(r_j)
        R3 = np.linalg.norm(r_k)

        wy = r_i @ np.cross(r_j, r_k)
        wx = (
            R1 * R2 * R3
            + R1 * np.dot(r_j, r_k)
            + R2 * np.dot(r_k, r_i)
            + R3 * np.dot(r_i, r_j)
        )
        wf = 2.0 * np.arctan2(wy, wx)

        # pot -= wf * np.dot(r_f, np.dot(F, r_f))
        # acc += wf * np.dot(F, r_f)

        F = np.outer(n_f, n_f)
        U -= r_i @ F @ r_i * wf
        acc += F @ r_i * wf
        # U -= r_i @ n_f @ np.outer(n_f, r_i)* wf
        # a += n_f @ np.outer(n_f, r_i) * wf

    return U, acc


class Mesh:
    def __init__(self, trimesh):
        self.vertices = copy.deepcopy(np.array(trimesh.vertices, dtype=np.float64))
        self.faces = copy.deepcopy(np.array(trimesh.faces, dtype=np.int32))
        self.edges_unique = copy.deepcopy(
            np.array(trimesh.edges_unique, dtype=np.int32),
        )


class Polyhedral_2(GravityModelBase):
    def __init__(self, celestial_body, obj_file, trajectory=None):
        """Polyhedral gravity model based on work from Werner and Scheeres
        (https://link.springer.com/article/10.1007/BF00053511)
        The model computes the accelerations from a constant density polyhedral shape
        model which is often a reasonable approximation for small bodies.

        Args:
            celestial_body (CelestialBody): Body for gravity calc
            obj_file (str): path to shape model of the body
            trajectory (TrajectoryBase, optional): Trajectory / distribution for which
                the gravity measurements should be computed. Defaults to None.
        """
        super().__init__(celestial_body, obj_file, trajectory=trajectory)
        self.obj_file = obj_file

        self.configure(trajectory)

        self.planet = celestial_body
        obj_file = make_windows_path_posix(obj_file)
        filename, file_extension = os.path.splitext(obj_file)
        self.mesh = trimesh.load_mesh(
            obj_file,
            file_type=file_extension[1:],
            dtype=np.float64,
        )
        # self.mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:], process=False)
        self.scaleFactor = 1e3  # Assume that the mesh is given in km
        self.volume = self.compute_volume()
        self.density = self.compute_density()

        self.reduce_mesh_memory()
        self.get_available_cores()

    def get_available_cores(self):
        try:
            int(os.environ["SLURM_JOB_NUM_NODES"])
            cores_per_nodes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split("(")[0])
            processes = cores_per_nodes
        except Exception:
            processes = mp.cpu_count()
        self.processes = processes

    def reduce_mesh_memory(self):
        smaller_mesh = Mesh(self.mesh)
        self.mesh = smaller_mesh

    def generate_full_file_directory(self):
        self.file_directory += (
            os.path.splitext(os.path.basename(__file__))[0]
            + "_"
            + os.path.basename(self.obj_file).split(".")[0]
            + "/"
        )
        pass

    def compute_volume(self):
        V = 0
        for face in self.mesh.faces:
            r1, r2, r3 = self.mesh.vertices[face[0:3], :] * self.scaleFactor
            V += np.abs(np.cross(r1, r2) @ r3) / 6.0
        return float(V)

    def compute_density(self):
        volume = self.mesh.volume * 1e9  # m^3
        volume = self.volume
        mu = self.planet.mu
        G = 6.67430 * 10**-11
        density = mu / (G * volume)
        print("Original Density: ", self.planet.density, "kg/m^3")
        print("Computed Density: ", density, "kg/m^3")
        return density

    # Debug functions
    def find_vertex(self, value_1, idx_1, value_2, idx_2):
        for i in range(len(self.vertices)):
            if np.isclose(self.mesh.vertices[i][idx_1], value_1) and np.isclose(
                self.mesh.vertices[i][idx_2],
                value_2,
            ):
                return i

    def find_edge(self, vertex_1_idx, vertex_2_idx):
        return np.intersect1d(
            np.where(self.mesh.edges_unique == vertex_1_idx)[0],
            np.where(self.mesh.edges_unique == vertex_2_idx)[0],
        )

    def plot_geometry(
        self,
        edge,
        edge_direction,
        normal_A,
        normal_B,
        edge_normal_A_to_B,
        edge_normal_B_to_A,
    ):
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.quiver(0, 0, 0, edge[0], edge[1], edge[2], color="red")
        ax.quiver(0, 0, 0, normal_A[0], normal_A[1], normal_A[2], color="blue")
        ax.quiver(0, 0, 0, normal_B[0], normal_B[1], normal_B[2], color="cyan")
        ax.quiver(
            0,
            0,
            0,
            edge_normal_A_to_B[0],
            edge_normal_A_to_B[1],
            edge_normal_A_to_B[2],
            color="purple",
        )
        ax.quiver(
            0,
            0,
            0,
            edge_normal_B_to_A[0],
            edge_normal_B_to_A[1],
            edge_normal_B_to_A[2],
            color="orange",
        )
        ax.quiver(
            0,
            0,
            0,
            edge_direction[0],
            edge_direction[1],
            edge_direction[2],
            color="yellow",
        )
        plt.show()

    # Bulk function
    def compute_acceleration(self, positions=None, pbar=True):
        "Compute the acceleration for an existing trajectory or provided positions"
        if positions is None:
            positions = self.trajectory.positions

        # bar = ProgressBar(positions.shape[0], enable=pbar)
        self.accelerations = np.zeros(positions.shape)
        self.potentials = np.zeros(len(positions))

        if len(positions) == 1:
            results = map(self.compute_values, positions)
        else:
            with mp.Pool(processes=self.processes) as pool:
                results = pool.map(self.compute_values, positions)

        for i, result in enumerate(results):
            self.accelerations[i] = result[0]
            self.potentials[i] = result[1]

        return self.accelerations

    def compute_potential(self, positions=None):
        "Compute the potential for an existing trajectory or provided positions"
        if positions is None:
            positions = self.trajectory.positions

        self.accelerations = np.zeros(positions.shape)
        self.potentials = np.zeros(len(positions))

        if len(positions) == 1:
            results = map(self.compute_values, positions)
        else:
            with mp.Pool(processes=self.processes) as pool:
                results = pool.map(self.compute_values, positions)

        for i, result in enumerate(results):
            self.accelerations[i] = result[0]
            self.potentials[i] = result[1]

        return self.potentials

    def compute_values(self, position):
        # G = 6.67408 * 1e-11  # m^3/(kg s^2)
        G = 6.67430 * 10**-11

        acc = np.zeros((3,))
        pot = 0.0

        # point_scaled = position
        # pot, acc = get_values(self.mesh.faces, self.mesh.vertices*self.scaleFactor, point_scaled)
        point_scaled = position / self.scaleFactor
        pot, acc = get_values(self.mesh.faces, self.mesh.vertices, point_scaled)

        pot *= 1.0 / 2.0 * G * self.density * self.scaleFactor**2
        acc *= G * self.density * self.scaleFactor

        # the paper gives delta U, not a.
        # Given that a is already standard, we are going to negate U
        return acc, -pot


def main():
    import time

    start = time.time()
    asteroid = Eros()
    poly_model = Polyhedral_2(asteroid, asteroid.obj_200k)
    poly_model = Polyhedral_2(asteroid, asteroid.obj_200k)
    print(time.time() - start)

    timeList = []
    position = np.ones((64, 3)) * 1e4  # Must be in meters
    start = time.time()
    print(poly_model.compute_acceleration(position))
    stop = time.time() - start
    timeList.append(stop)
    print(stop)

    print(timeList)


if __name__ == "__main__":
    main()
