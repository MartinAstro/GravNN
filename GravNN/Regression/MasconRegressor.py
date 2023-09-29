import os

import numpy as np
import trimesh
from numba import njit
from scipy.optimize import nnls

import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories.RandomDist import RandomDist

np.random.seed(10)


@njit(cache=True)
def populate_M(r_vec, r_masses, a_vec):
    N_masses = len(r_masses)
    N_meas = len(a_vec) * 3
    M = np.zeros((N_meas, N_masses))

    for i, r_i in enumerate(r_vec):
        A_i = np.zeros((3, N_masses))
        for j in range(3):
            for k in range(N_masses):
                dr = r_i - r_masses[k]
                dr_mag = np.linalg.norm(dr)
                A_i[j, k] = -dr[j] / dr_mag**3
        M[i * 3 : (i + 1) * 3, :] = A_i
    return M


def iterate_lstsq(M, aVec, iterations):
    results = np.linalg.lstsq(M, aVec)[0]
    delta_a = aVec - np.dot(M, results)
    for i in range(iterations):
        delta_coef = np.linalg.lstsq(M, delta_a)[0]
        results -= delta_coef
        delta_a = aVec - np.dot(M, results)
    return results


class MasconRegressor:
    def __init__(self, planet, obj_file, N_masses):
        self.planet = planet
        self.radius = planet.radius
        self.mu = planet.mu
        self.obj_file = obj_file
        self.N_masses = N_masses

        self.filename = os.path.basename(self.obj_file)
        _, file_extension = os.path.splitext(self.obj_file)
        self.obj_mesh = trimesh.load_mesh(
            self.obj_file,
            file_type=file_extension[1:],
        )

        self.r_masses = self.initializes_mass_positions()

    def initializes_mass_positions(self):
        positions = self.sample_volume(self.N_masses)
        positions = self.recursively_remove_exterior_points(positions)
        return positions

    def sample_volume(self, points):
        X = []
        Y = []
        Z = []
        X.extend(np.zeros((points,)).tolist())
        Y.extend(np.zeros((points,)).tolist())
        Z.extend(np.zeros((points,)).tolist())

        theta = np.random.uniform(0, 2 * np.pi, size=(points,))
        cosphi = np.random.uniform(-1, 1, size=(points,))
        R_min = 0
        R_max = self.radius

        # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
        u_min = (R_min / R_max) ** 3
        u_max = 1.0

        # want distribution to be uniform across volume the sphere
        u = np.random.uniform(u_min, u_max, size=(points,))

        # convert the uniform volume length into physical radius
        r = R_max * u ** (1.0 / 3.0)

        phi = np.arccos(cosphi)

        X = r * np.sin(phi) * np.cos(theta)
        Y = r * np.sin(phi) * np.sin(theta)
        Z = r * np.cos(phi)

        return np.transpose(np.array([X, Y, Z]))  # [N x 3]

    def identify_exterior_points(self, positions):
        # Necessary to cap memory footprint
        N = len(positions)
        step = 500
        mask = np.full((N,), False)
        rayObject = trimesh.ray.ray_triangle.RayMeshIntersector(self.obj_mesh)
        for i in range(0, N, step):
            end_idx = (i // step + 1) * step
            position_subset = positions[i:end_idx] / 1e3
            mask[i:end_idx] = ~rayObject.contains_points(position_subset)
            print(i / N)
        return mask

    def recursively_remove_exterior_points(self, positions):
        mask = self.identify_exterior_points(positions)
        exterior_points = np.sum(mask)
        print(f"Remaining Points: {exterior_points}")
        if exterior_points > 0:
            new_positions = self.sample_volume(exterior_points)
            positions[mask] = self.recursively_remove_exterior_points(new_positions)
        return positions

    def update(self, r_vec, a_vec, iterations=5):
        M = populate_M(r_vec, self.r_masses, a_vec)
        a_vec_1D = a_vec.reshape((-1,))
        mu_vec, rnorm = nnls(M, a_vec_1D)
        self.mu_vec = mu_vec
        # results = iterate_lstsq(M, a_vec_1D, iterations)
        return mu_vec

    def save(self, name):
        save_data = np.append(self.mu_vec.reshape((-1, 1)), self.r_masses, axis=1)
        gravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__))
        os.makedirs(
            f"{gravNN_dir}/Files/GravityModels/Regressed/Mascons/",
            exist_ok=True,
        )
        np.savetxt(
            f"{gravNN_dir}/Files/GravityModels/Regressed/Mascons/{name}",
            save_data,
            delimiter=",",
        )


def main():
    import time

    planet = Eros()
    obj_file = planet.obj_8k
    N_masses = 1000

    poly_gm = Polyhedral(planet, obj_file)
    traj = RandomDist(
        planet,
        [planet.radius, planet.radius * 2],
        5000,
        obj_file=obj_file,
    )

    x = traj.positions
    a = poly_gm.compute_acceleration(x)

    filename = "Mascon_Eros_10_test.csv"
    regressor = MasconRegressor(planet, obj_file, N_masses)
    start = time.time()
    mu_vec = regressor.update(x, a)
    regressor.save(filename)

    print(regressor.r_masses)
    print(mu_vec)
    print(time.time() - start)

    from GravNN.GravityModels.Mascons import Mascons

    mascons = Mascons(planet, filename)
    a_mascons = mascons.compute_acceleration(x)

    da = a - a_mascons
    percent_error = np.linalg.norm(da, axis=1) / np.linalg.norm(a, axis=1) * 100
    print(np.average(percent_error))


if __name__ == "__main__":
    main()
