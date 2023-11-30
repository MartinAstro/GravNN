from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories.PlanesDist import PlanesDist
from GravNN.Trajectories.RandomDist import RandomDist


def Random():
    traj = RandomDist(Earth(), [0, Earth().radius * 2], 13)
    file_name_int = traj.file_directory

    traj = RandomDist(Earth(), [0.0, Earth().radius * 2], 13)
    file_name_float = traj.file_directory

    assert file_name_float == file_name_int

    sh = SphericalHarmonics(Earth().EGM96, degree=2, trajectory=traj, parallel=False)
    sh.load(override=True)

    sh = SphericalHarmonics(Earth().EGM96, degree=2, trajectory=traj, parallel=False)
    sh.load(override=False)

    # delete the directory and its contents
    import shutil

    shutil.rmtree(file_name_int)


def Planes():
    traj = PlanesDist(Eros(), [-Eros().radius * 2, Eros().radius * 2], 5)
    file_name_int = traj.file_directory

    traj = PlanesDist(Eros(), [-Eros().radius * 2, Eros().radius * 2], 5)
    file_name_float = traj.file_directory

    assert file_name_float == file_name_int

    poly = Polyhedral(Eros(), Eros().obj_77, trajectory=traj)
    poly.load(override=True)

    poly = Polyhedral(Eros(), Eros().obj_77, trajectory=traj)
    poly.load(override=False)

    # delete the directory and its contents
    import shutil

    shutil.rmtree(file_name_int)


if __name__ == "__main__":
    Random()
    Planes()
