from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.RandomDist import RandomDist


def main():
    traj = RandomDist(Earth(), [0, Earth().radius * 2], 10)
    file_name_int = traj.file_directory

    traj = RandomDist(Earth(), [0.0, Earth().radius * 2], 10)
    file_name_float = traj.file_directory

    assert file_name_float == file_name_int


if __name__ == "__main__":
    main()
