import csv
import json
import os

import numpy as np

from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.GravityModels.PinesAlgorithm import *
from GravNN.Regression.utils import RegressSolution


def make_2D_array(lis):
    """Funciton to get 2D array from a list of lists"""
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len))

    for i in range(n):
        arr[i, : lengths[i]] = lis[i]
    return arr


def get_normalization(l, m):  # noqa: E741
    N = np.zeros((l + 1, l + 1))
    for i in range(0, l + 1):
        for j in range(0, i + 1):
            if j == 0:
                N[i, j] = np.sqrt(2.0 * i + 1)
            else:
                N[i, j] = np.sqrt(
                    (2 * (2 * i + 1) * np.math.factorial(i - j))
                    / (np.math.factorial(i + j)),
                )
    return N


def get_sh_data(trajectory, gravity_file, **kwargs):
    override = bool(kwargs.get("override", [False])[0])
    parallel = kwargs.get("parallel", False)
    try:
        max_deg = int(kwargs["max_deg"][0])
        deg_removed = int(kwargs["deg_removed"][0])
    except Exception:
        max_deg = int(kwargs["max_deg"])
        deg_removed = int(kwargs["deg_removed"])

    Call_r0_gm = SphericalHarmonics(
        gravity_file,
        degree=max_deg,
        trajectory=trajectory,
        parallel=parallel,
    )
    accelerations = Call_r0_gm.load(override=override).accelerations
    potentials = Call_r0_gm.potentials

    Clm_r0_gm = SphericalHarmonics(
        gravity_file,
        degree=deg_removed,
        trajectory=trajectory,
        parallel=parallel,
    )
    accelerations_Clm = Clm_r0_gm.load(override=override).accelerations
    potentials_Clm = Clm_r0_gm.potentials

    x = Call_r0_gm.positions  # position (N x 3)
    a = accelerations - accelerations_Clm
    u = potentials - potentials_Clm  # (N,)

    return x, a, u


class SphericalHarmonicsDegRemoved(GravityModelBase):
    def __init__(self, sh_info, degree, remove_deg, trajectory=None, parallel=False):
        self.sh_hf = SphericalHarmonics(sh_info, degree, trajectory)
        self.sh_lf = SphericalHarmonics(sh_info, remove_deg, trajectory)
        super().__init__()
        self.configure(trajectory)
        self.deg_removed = degree

    def compute_potential(self, positions=None):
        if positions is None:
            positions = self.trajectory.positions
        self.sh_hf.compute_potential(positions)
        self.sh_lf.compute_potential(positions)
        self.potentials = self.sh_hf.potentials - self.sh_lf.potentials
        return self.potentials

    def compute_acceleration(self, positions=None):
        if positions is None:
            positions = self.trajectory.positions
        self.sh_hf.compute_acceleration(positions)
        self.sh_lf.compute_acceleration(positions)
        self.accelerations = self.sh_hf.accelerations - self.sh_lf.accelerations
        return self.accelerations

    def generate_full_file_directory(self):
        class_name = self.__class__.__name__
        obj_file = os.path.basename(self.sh_hf.file).split(".csv")[0].split(".txt")[0]
        hf_deg = str(self.sh_hf.degree)
        lf_deg = str(self.sh_lf.degree)
        self.file_directory += f"{class_name}_{obj_file}_{hf_deg}_{lf_deg}/"


class SphericalHarmonics(GravityModelBase):
    def __init__(self, sh_info, degree, trajectory=None, parallel=False):
        """Spherical Harmonic Gravity Model. Takes in a set of Stokes coefficients and
        computes acceleration and potentials using a non-singular representation
        (Pines Algorithm).

        Args:
            sh_info (str): path to spherical harmonic coefficients (Stokes coefficients)
            degree (int): maximum degree of the spherical harmonic expansions
            trajectory (TrajectoryBase, optional): Trajectory / distribution for which
            the gravity measurements should be produced. Defaults to None.
        """
        super().__init__(sh_info, degree, trajectory=trajectory, parallel=parallel)

        self.degree = degree

        self.mu = None
        self.radEquator = None
        self.C_lm = None
        self.S_lm = None

        if isinstance(sh_info, RegressSolution):
            self.file = "./"
        else:
            self.file = sh_info

        self.configure(trajectory)

        if isinstance(sh_info, RegressSolution):
            # If the harmonics should be loaded from a RegressSolution
            self.load_regression(sh_info)
        else:
            self.loadSH()

        self.n1, self.n2, self.n1q, self.n2q = compute_n_matrices(self.degree)

        # if parallel:
        #     self.compute_fcn = compute_acc_parallel
        # else:
        #     self.compute_fcn = compute_acc_jit
        # pass

    def generate_full_file_directory(self):
        self.file_directory += (
            os.path.splitext(os.path.basename(__file__))[0]
            + "_"
            + os.path.basename(self.file).split(".csv")[0].split(".txt")[0]
            + "_"
            + str(self.degree)
            + "/"
        )
        pass

    def loadSH_json(self):
        data = json.load(open(self.file, "r"))
        clm = np.zeros((40, 40)).tolist()
        slm = np.zeros((40, 40)).tolist()
        for i in range(len(data["Cnm_coefs"])):
            n = data["Cnm_coefs"][i]["n"]
            m = data["Cnm_coefs"][i]["m"]

            clm[n][m] = data["Cnm_coefs"][i]["value"]

            n = data["Snm_coefs"][i]["n"]
            m = data["Snm_coefs"][i]["m"]

            slm[n][m] = data["Snm_coefs"][i]["value"]

        for i in range(len(clm)):
            mask = np.ones(len(clm[i]), dtype=bool)
            mask[i + 1 :] = False
            clm_row = np.array(clm[i])[mask]
            slm_row = np.array(slm[i])[mask]

            clm[i] = clm_row.tolist()
            slm[i] = slm_row.tolist()

        self.mu = data["totalMass"]["value"] * 6.67408 * 1e-11
        self.radEquator = data["referenceRadius"]["value"]
        self.C_lm = make_2D_array(clm)
        self.S_lm = make_2D_array(slm)
        return

    def loadSH_csv(self):
        need_whitespace_fix = False
        with open(self.file, "r") as csvfile:
            gravReader = csv.reader(csvfile, delimiter=",")
            firstRow = next(gravReader)
            clmList = []
            slmList = []
            # Currently do not take the mu and radius values
            # provided by the gravity file.
            try:
                int(firstRow[0])
            except ValueError:
                try:
                    self.mu = float(firstRow[1])
                    self.radEquator = float(firstRow[0])
                except IndexError:
                    need_whitespace_fix = True
                    split_first_row = firstRow[0].split()
                    self.mu = float(split_first_row[1])
                    self.radEquator = float(split_first_row[0])

            clmRow = []
            slmRow = []
            currDeg = 0
            for gravRow in gravReader:
                if need_whitespace_fix:
                    gravRow = gravRow[0].split()

                # TODO: Check if this results in correct behavior
                if self.degree is not None:
                    # if loading coefficients beyond the maximum desired
                    # (+2 for purposes of the algorithm)
                    if int(float(gravRow[0])) > self.degree + 2:
                        break
                while int(float(gravRow[0])) > currDeg:
                    if len(clmRow) < currDeg + 1:
                        clmRow.extend([0.0] * (currDeg + 1 - len(clmRow)))
                        slmRow.extend([0.0] * (currDeg + 1 - len(slmRow)))
                    clmList.append(clmRow)
                    slmList.append(slmRow)
                    clmRow = []
                    slmRow = []
                    currDeg += 1
                clmRow.append(float(gravRow[2]))
                slmRow.append(float(gravRow[3]))

            clmList.append(clmRow)
            slmList.append(slmRow)
            if self.degree is None:
                self.degree = currDeg - 2

            self.C_lm = make_2D_array(clmList)
            self.S_lm = make_2D_array(slmList)
            return

    def loadSH(self):
        if ".json" in self.file:
            self.loadSH_json()
        else:
            self.loadSH_csv()

    def load_regression(self, reg_solution):
        self.file_directory += "_Regress"
        self.mu = reg_solution.planet.mu
        self.radEquator = reg_solution.planet.radius
        self.C_lm = reg_solution.C_lm
        self.S_lm = reg_solution.S_lm

        return

    def compute_potential(self, positions=None):
        "Compute the potential for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions

        positions = np.reshape(positions, (len(positions) * 3))

        accelerations, potentials = compute_acc(
            positions,
            self.degree,
            self.mu,
            self.radEquator,
            self.n1,
            self.n2,
            self.n1q,
            self.n2q,
            self.C_lm,
            self.S_lm,
        )

        self.accelerations = np.reshape(
            np.array(accelerations),
            (int(len(np.array(accelerations)) / 3), 3),
        )
        self.potentials = potentials

        return self.potentials

    def compute_acceleration(self, positions=None):
        "Compute the acceleration for an existing trajectory or set of positions"
        if positions is None:
            positions = self.trajectory.positions

        positions = np.reshape(positions, (len(positions) * 3))

        accelerations, potentials = compute_acc(
            positions,
            self.degree,
            self.mu,
            self.radEquator,
            self.n1,
            self.n2,
            self.n1q,
            self.n2q,
            self.C_lm,
            self.S_lm,
        )

        self.accelerations = np.reshape(
            np.array(accelerations),
            (int(len(np.array(accelerations)) / 3), 3),
        )
        self.potentials = potentials
        return self.accelerations


if __name__ == "__main__":
    import time

    from GravNN.CelestialBodies.Planets import Earth

    planet = Earth()
    # traj = FibonacciDist(planet, planet.radius, 1000)
    # grav_model = SphericalHarmonics(planet.sh_file, 1000, traj)
    # grav_model.load(override=True)
    # acc = grav_model.accelerations
    # pot = grav_model.potentials

    # grav_model = SphericalHarmonics(planet.sh_file, 13)
    # print(grav_model.compute_acceleration([[Earth().radius, 0, 0]]))
    # grav_model.load(override=True)
    # acc = grav_model.accelerations
    # pot = grav_model.potentials

    N = 10000
    x = planet.radius * np.random.normal(2, 0.05, size=(N, 3))
    grav_model = SphericalHarmonics(planet.sh_file, 1000, parallel=False)
    start = time.time()
    grav_model.compute_acceleration(x)
    print((time.time() - start) / N)

    grav_model = SphericalHarmonics(planet.sh_file, 1000, parallel=True)
    start = time.time()
    grav_model.compute_acceleration(x)
    print((time.time() - start) / N)
