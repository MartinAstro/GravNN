import os
import tempfile

import numpy as np
import trimesh

import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Mascons import Mascons
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.ProgressBar import ProgressBar
from GravNN.Trajectories.RandomDist import RandomDist

np.random.seed(10)


# @njit(cache=True)
def populate_M(r_points, r_masses):
    N_masses = len(r_masses)
    N_meas = len(r_points)
    M = np.zeros((3 * N_meas, N_masses))

    for i, r_i in enumerate(r_points):
        dr = r_i - r_masses
        dr_mag = np.linalg.norm(dr, axis=1, keepdims=True)

        M[3 * i : 3 * (i + 1), :] = np.transpose(-dr / dr_mag**3)

    return M


def iterate_lstsq(M, aVec, iterations):
    results = np.linalg.lstsq(M, aVec)[0]
    delta_a = aVec - np.dot(M, results)
    for i in range(iterations):
        delta_coef = np.linalg.lstsq(M, delta_a)[0]
        results -= delta_coef
        delta_a = aVec - np.dot(M, results)
    return results


class MasconRegressorSequential:
    def __init__(self, planet, obj_file, N_masses):
        self.planet = planet
        self.radius = planet.radius
        self.mu = planet.mu
        self.obj_file = obj_file
        self.N_masses = N_masses

        self.filename = os.path.basename(self.obj_file)

    def remove_current_model(self, x, a, mu_list, r_masses):
        with tempfile.NamedTemporaryFile() as tmpfile:
            save_data = np.append(mu_list.reshape((-1, 1)), r_masses, axis=1)
            np.savetxt(
                tmpfile.name,
                save_data,
                delimiter=",",
            )
            regressed_model = Mascons(self.planet, tmpfile.name)
            accelerations = regressed_model.compute_acceleration(x)
            da = a - accelerations
            da_percent = np.linalg.norm(da, axis=1) / np.linalg.norm(a, axis=1)
            da_percent_avg = np.mean(da_percent)
            brill_mask = np.linalg.norm(x, axis=1) > self.planet.radius
            print(f"Current model error: {da_percent_avg*100}% \t {len(mu_list)}")
            print(f"Outside Brillouin Sphere: {np.mean(da_percent[brill_mask]) * 100}")
        return da

    def batches(self, batch_size):
        # divide N_masses into batches of mass_batch_size
        full_batches = self.N_masses // batch_size
        batches = np.ones((full_batches,), dtype=int) * batch_size
        if self.N_masses % batch_size != 0:
            batches = np.append(batches, self.N_masses % batch_size)
        return batches

    def update(self, r_vec, a_vec, mass_batch_size=1000):
        batches = self.batches(mass_batch_size)

        mu_list = None
        r_masses = None

        da = a_vec.copy()

        # Iterate over the batches and continuously update the model
        pbar = ProgressBar(len(batches), enable=True)
        for i, mass_batch in enumerate(batches):
            regressor = MasconRegressorFancy(self.planet, self.obj_file, mass_batch)
            mu_vec = regressor.update(r_vec, da, mass_batch / self.N_masses)
            # regressor = MasconRegressor(self.planet, self.obj_file, mass_batch)
            # mu_vec = regressor.update(r_vec, da)

            # if i != len(batches) - 1:
            #     mu_vec *= mass_batch / self.N_masses

            # save off the masses
            if mu_list is None:
                mu_list = mu_vec
                r_masses = regressor.r_masses
            else:
                mu_list = np.concatenate((mu_list, mu_vec))
                r_masses = np.concatenate((r_masses, regressor.r_masses))

            # remove the current model from the acceleration
            da = self.remove_current_model(r_vec, a_vec, mu_list, r_masses)
            pbar.update(i)

        # save values for saving
        self.mu_vec = mu_list
        self.r_masses = r_masses

        # count number of zeros in mu
        print("Number of Zero Mascons: ", np.sum(mu_list == 0))

    def save(self, name):
        save_data = np.append(self.mu_vec.reshape((-1, 1)), self.r_masses, axis=1)

        # if the filename is absolute, just save it there
        if os.path.isabs(name):
            np.savetxt(name, save_data, delimiter=",")
            return

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

    def update(self, r_vec, a_vec):
        M = populate_M(r_vec, self.r_masses)
        a_vec_1D = a_vec.reshape((-1,))
        # mu_vec, rnorm = nnls(M, a_vec_1D)
        mu_vec = iterate_lstsq(M, a_vec_1D, 1)
        self.mu_vec = mu_vec
        return mu_vec

    def save(self, name):
        save_data = np.append(self.mu_vec.reshape((-1, 1)), self.r_masses, axis=1)

        # if the filename is absolute, just save it there
        if os.path.isabs(name):
            np.savetxt(name, save_data, delimiter=",")
            return

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


class MasconRegressorFancy(MasconRegressor):
    def __init__(self, planet, obj_file, N_masses):
        super().__init__(planet, obj_file, N_masses)

    def update(self, r_vec, a_vec, mu_frac):
        M = populate_M(r_vec, self.r_masses)
        a_vec_1D = a_vec.reshape((-1,))
        # mu_vec, rnorm = nnls(M, a_vec_1D)
        mu_vec = iterate_lstsq(M, a_vec_1D, 1)
        self.mu_vec = mu_vec
        return mu_vec


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

    print("Mass Positions:", regressor.r_masses)
    print("Mu Vec:", mu_vec)
    print("Elapsed Time:", time.time() - start)

    from GravNN.GravityModels.Mascons import Mascons

    mascons = Mascons(planet, filename)
    a_mascons = mascons.compute_acceleration(x)

    da = a - a_mascons
    percent_error = np.linalg.norm(da, axis=1) / np.linalg.norm(a, axis=1) * 100
    print(np.average(percent_error))


def test_sequential(N_masses, N_batch):
    planet = Eros()
    obj_file = planet.obj_8k

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
    regressor = MasconRegressorSequential(planet, obj_file, N_masses)
    regressor.update(x, a, N_batch)
    regressor.save(filename)

    # print("Mass Positions:", regressor.r_masses)
    # print("Mu Vec:", mu_vec)
    # print("Elapsed Time:", time.time() - start)

    from GravNN.GravityModels.Mascons import Mascons

    mascons = Mascons(planet, filename)

    # validation Data
    traj = RandomDist(
        planet,
        [planet.radius, planet.radius * 2],
        1000,
        obj_file=obj_file,
    )
    x_val = traj.positions
    a_val = poly_gm.compute_acceleration(x_val)
    a_mascons = mascons.compute_acceleration(x_val)
    da = a_val - a_mascons
    percent_error = np.linalg.norm(da, axis=1) / np.linalg.norm(a_val, axis=1) * 100
    print("Masses:", N_masses, "\t Batch Size:", N_batch)
    print("Error:", np.average(percent_error))


if __name__ == "__main__":
    # main()

    # test_sequential(
    #     N_masses=500,
    #     N_batch=100,
    # )
    # test_sequential(
    #     N_masses=1000,
    #     N_batch=1000,
    # )
    # test_sequential(
    #     N_masses=1000,
    #     N_batch=500,
    # )
    # test_sequential(
    #     N_masses=1000,
    #     N_batch=100,
    # )
    test_sequential(
        N_masses=1000,
        N_batch=100,
    )
