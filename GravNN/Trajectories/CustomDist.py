import os
import re
from copy import deepcopy

import numpy as np

from GravNN.Trajectories.TrajectoryBase import TrajectoryBase


def read_ephemeris_head(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    object = re.split(r" {2,}", lines[1])[1]
    for i in range(len(lines)):
        if "Start time" in lines[i]:
            start_time = lines[i].split(":")[1].replace(" ", "-")
        if "Stop  time" in lines[i]:
            stop_time = lines[i].split(":")[1].replace(" ", "-")

    return object, start_time, stop_time


def read_ephemeris_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if "$$SOE" in lines[i]:
            start = i + 1
            break

    for i in range(len(lines) - 1, 0, -1):
        if "$$EOE" in lines[i]:
            end = i - 1

    x_vec = np.zeros((end - start, 3))
    t_vec = np.zeros((end - start, 1))

    for k in range(start, end):
        line = lines[k]
        entries = line.split(",")
        t_vec[k - start] = float(entries[0])
        x_vec[k - start][0] = float(entries[2])
        x_vec[k - start][1] = float(entries[3])
        x_vec[k - start][2] = float(entries[4])

    return t_vec, x_vec


class CustomDist(TrajectoryBase):
    def __init__(self, ephemeris_file, max_data_entries=None, **kwargs):
        """Samples taken from a distribution defined in an external text file. Currently only for
        ephemeris files produced by JPL Horizons. WIP.

        Args:
            ephemeris_file (str): path to external file
            max_data_entries (int, optional): truncate the number of samples in file. Defaults to None.
        """
        self.ephemeris_file = ephemeris_file
        self.file_name = os.path.basename(ephemeris_file).split(".")[0]
        self.object, self.start_time, self.stop_time = read_ephemeris_head(
            ephemeris_file,
        )
        self.max_data_entries = max_data_entries
        super().__init__(**kwargs)

        pass

    def generate_full_file_directory(self):
        self.data_name = (
            self.file_name
            + "_start_"
            + str(self.start_time)
            + "_stop_"
            + str(self.stop_time)
            + "_N_"
            + str(self.max_data_entries)
        )
        self.trajectory_name = (
            os.path.splitext(os.path.basename(__file__))[0] + "/" + self.data_name
        )

        self.file_directory += self.trajectory_name + "/"
        pass

    def generate(self):
        """Read in the times and positions provided in the ephemeris text file

        Returns:
            np.array: cartesian position vectors
        """
        self.times, self.positions = read_ephemeris_data(self.ephemeris_file)
        self.dt = self.times[1] - self.times[0]

        if self.max_data_entries is not None:
            self.times = self.times[: self.max_data_entries]
            self.positions = self.positions[: self.max_data_entries]

        self.points = len(self.times)
        return self.positions

    def __sub__(self, other):
        """Subtract different custom distribution (i.e. near w.r.t. sun minus eros w.r.t. sun)

        Args:
            other (CustomDist): the distribution to be subtracted off of self

        Returns:
            CustomDist: new CustomDist
        """
        # Difference the positions of the two ephermii

        # Ensure that the trajectories share the same timestamps
        assert np.all(other.times == self.times)

        newEphemeris = deepcopy(self)
        newEphemeris.positions -= other.positions
        newEphemeris.file_directory += other.data_name + "/"

        return newEphemeris


def main():
    # Meant for testing
    import matplotlib.pyplot as plt

    from GravNN.Visualization.VisualizationBase import VisualizationBase

    near_traj = EphemerisDist(
        "GravNN/Files/Ephmerides/near_positions.txt",
        override=False,
    )
    eros_traj = EphemerisDist(
        "GravNN/Files/Ephmerides/eros_positions.txt",
        override=True,
    )

    near_m_eros_traj = near_traj - eros_traj
    X = near_m_eros_traj.positions[1000:, 0]
    Y = near_m_eros_traj.positions[1000:, 1]
    Z = near_m_eros_traj.positions[1000:, 2]
    vis = VisualizationBase()
    fig, ax = vis.new3DFig()
    ax.plot3D(X, Y, Z, "gray")
    # plt.scatter(X, Y, Z)
    plt.show()


if __name__ == "__main__":
    main()
