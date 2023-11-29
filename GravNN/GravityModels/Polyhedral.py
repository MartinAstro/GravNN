import copy
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from numba import njit  # , range

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.GravityModels.PointMass import PointMass
from GravNN.Support.PathTransformations import make_windows_path_posix


def get_poly_data(trajectory, obj_mesh_file, **kwargs):
    override = bool(kwargs.get("override", [False])[0])
    remove_point_mass = bool(kwargs.get("remove_point_mass", [False])[0])

    obj_mesh_file = make_windows_path_posix(obj_mesh_file)

    poly_r0_gm = Polyhedral(
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
def compute_edge_dyads(
    vertices,
    faces,
    edges_unique,
    face_adjacency_edges,
    face_normals,
    face_adjacency,
):
    edge_dyads = np.zeros((len(edges_unique), 3, 3))  # In order of unique edges
    for i in range(len(edges_unique)):
        P1 = vertices[edges_unique[i][0]]
        P2 = vertices[edges_unique[i][1]]

        facet_A_idx = int(face_adjacency[i][0])
        facet_B_idx = int(face_adjacency[i][1])
        normal_A = face_normals[facet_A_idx]
        normal_B = face_normals[facet_B_idx]

        face_A_vertices = faces[facet_A_idx]
        face_B_vertices = faces[facet_B_idx]
        face_A_c = (
            vertices[face_A_vertices[0]]
            + vertices[face_A_vertices[1]]
            + vertices[face_A_vertices[2]]
        ) / 3.0
        face_B_c = (
            vertices[face_B_vertices[0]]
            + vertices[face_B_vertices[1]]
            + vertices[face_B_vertices[2]]
        ) / 3.0

        B_2_A = face_A_c - face_B_c
        A_2_B = face_B_c - face_A_c

        edge_direction = P1 - P2
        edge_direction /= np.linalg.norm(edge_direction)

        # edge_direction = np.cross(normal_A, normal_B)
        # if np.any(np.isnan(edge_direction)):
        #     print("same direction")
        #     edge_direction = np.array([0.0, 0.0, 0.0])

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


@njit(cache=True, parallel=False)
def compute_facet_dyads(face_normals):
    facet_dyads = np.zeros((len(face_normals), 3, 3))
    for i in range(len(face_normals)):
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

    return 2.0 * np.arctan2(
        np.dot(r0m, r1m_cross_r2m),
        R0 * R1 * R2
        + R0 * np.dot(r1m, r2m)
        + R1 * np.dot(r0m, r2m)
        + R2 * np.dot(r0m, r1m),
    )


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

    return np.log((R0 + R1 + Re) / (R0 + R1 - Re))


@njit(cache=True, parallel=False)
def facet_acc_loop(point_scaled, vertices, faces, facet_dyads):
    acc = np.zeros((3,))
    pot = 0.0
    for i in range(len(faces)):
        r0 = vertices[faces[i][0]]
        vertices[faces[i][1]]
        vertices[faces[i][2]]

        r_e = r0 - point_scaled
        r_f = r_e  # Page 11

        wf = GetPerformanceFactor(point_scaled, vertices, faces, i)
        F = facet_dyads[i]

        acc += wf * np.dot(F, r_f)
        pot -= wf * np.dot(r_f, np.dot(F, r_f))
    return acc, pot


@njit(cache=True, parallel=False)
def edge_acc_loop(point_scaled, vertices, edges_unique, edge_dyads):
    acc = np.zeros((3,))
    pot = 0.0
    for i in range(len(edges_unique)):
        r0 = vertices[edges_unique[i][0]]
        r1 = vertices[edges_unique[i][1]]

        # Page 12 implies that r_e can be any point on edge e or its infinite extension
        r_middle = (r0 + r1) / 2

        r_e = r_middle - point_scaled

        Le = GetLe(point_scaled, vertices, edges_unique, i)
        E = edge_dyads[i]

        acc -= Le * np.dot(E, r_e)
        pot += Le * np.dot(r_e, np.dot(E, r_e))

    return acc, pot


class Mesh:
    def __init__(self, trimesh):
        self.vertices = copy.deepcopy(np.array(trimesh.vertices))
        self.faces = copy.deepcopy(np.array(trimesh.faces, dtype=np.int32))
        self.edges_unique = copy.deepcopy(
            np.array(trimesh.edges_unique, dtype=np.int32),
        )


class Polyhedral(GravityModelBase):
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
        self.mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
        self.scaleFactor = 1e3  # Assume that the mesh is given in km
        self.density = self.compute_density()

        self.facet_dyads = compute_facet_dyads(self.mesh.face_normals)
        self.edge_dyads = compute_edge_dyads(
            self.mesh.vertices,
            self.mesh.faces,
            self.mesh.edges_unique,
            self.mesh.face_adjacency_edges,
            self.mesh.face_normals,
            self.mesh.face_adjacency,
        )
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

    def compute_density(self):
        volume = self.mesh.volume * 1e9  # m^3
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
        G = 6.67430 * 10**-11  # m^3/(kg s^2)

        point_scaled = position / self.scaleFactor
        acc = np.zeros((3,))
        pot = 0.0

        acc_facet, pot_facet = facet_acc_loop(
            point_scaled,
            self.mesh.vertices,
            self.mesh.faces,
            self.facet_dyads,
        )
        acc_edge, pot_edge = edge_acc_loop(
            point_scaled,
            self.mesh.vertices,
            self.mesh.edges_unique,
            self.edge_dyads,
        )

        acc = acc_facet + acc_edge  # signs taken care of in loop
        acc *= G * self.density * self.scaleFactor

        pot = pot_edge + pot_facet  # signs taken care of in loop
        pot *= (
            1.0 / 2.0 * G * self.density * self.scaleFactor**2
        )  # [km^2/s^2] - > [m^2/s^2]

        # the paper gives delta U, not a.
        # Given that a is already standard, we are going to negate U
        return acc, -pot


def main():
    import time

    start = time.time()
    asteroid = Eros()
    poly_model = Polyhedral(asteroid, asteroid.obj_200k)
    poly_model = Polyhedral(asteroid, asteroid.obj_200k)
    print(time.time() - start)

    timeList = []
    position = np.ones((64, 3)) * 1e4  # Must be in meters
    start = time.time()
    print(poly_model.compute_acceleration(position))
    stop = time.time() - start
    timeList.append(stop)
    print(stop)

    position = np.ones((128, 3)) * 1e4  # Must be in meters
    start = time.time()
    poly_model.compute_acceleration(position)
    stop = time.time() - start
    timeList.append(stop)
    print(stop)

    position = np.ones((256, 3)) * 1e4  # Must be in meters
    start = time.time()
    poly_model.compute_acceleration(position)
    stop = time.time() - start
    timeList.append(stop)
    print(stop)

    position = np.ones((1024, 3)) * 1e4  # Must be in meters
    start = time.time()
    poly_model.compute_acceleration(position)
    stop = time.time() - start
    timeList.append(stop)
    print(stop)

    # position = np.ones((512,3))*1E4# Must be in meters
    # start = time.time()
    # poly_model.compute_acceleration(position)
    # stop = time.time() - start
    # timeList.append(stop)
    # print(stop)

    # position = np.ones((1024,3))*1E4# Must be in meters
    # start = time.time()
    # poly_model.compute_acceleration(position)
    # stop = time.time() - start
    # timeList.append(stop)
    # print(stop)

    print(timeList)


def test_energy_conservation():
    from scipy.integrate import solve_ivp

    asteroid = Eros()
    poly_model = Polyhedral(asteroid, asteroid.obj_8k)

    def fun(x, y, IC=None):
        "Return the first-order system"
        # print(x)
        R, V = y[0:3], y[3:6]
        np.linalg.norm(R)
        a = poly_model.compute_acceleration(R.reshape((1, 3)), pbar=False)
        dxdt = np.hstack((V, a.reshape((3,))))
        return dxdt.reshape((6,))

    T = 1000
    state = np.array(
        [
            -6.36256532e02,
            -4.58656092e04,
            1.31640352e04,
            3.17126984e-01,
            -1.12030801e00,
            -3.38751010e00,
        ],
    )

    sol = solve_ivp(
        fun,
        [0, T],
        state.reshape((-1,)),
        t_eval=None,
        events=None,
        atol=1e-12,
        rtol=1e-12,
    )

    from OrbitalElements.orbitalPlotting import plot_orbit_3d

    plot_orbit_3d(sol.y, save=False)
    # sol = solve_ivp(
    #     fun,
    #     [0, T],
    #     state.reshape((-1,)),
    #     t_eval=None,
    #     events=None,
    #     atol=1e-14,
    #     rtol=1e-14,
    # )
    rVec = sol.y[0:3, :]
    vVec = sol.y[3:6, :]
    hVec = np.cross(rVec.T, vVec.T)
    h_norm = np.linalg.norm(hVec, axis=1)

    KE = 1.0 / 2.0 * 1 * np.linalg.norm(vVec, axis=0) ** 2
    U = poly_model.compute_potential(rVec.T)

    energy = KE + U

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, T, len(h_norm)), h_norm, label="orbit")
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, T, len(h_norm)), energy, label="orbit")
    plt.show()


if __name__ == "__main__":
    main()
    # test_energy_conservation()
