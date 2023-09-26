import time

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.slurm_utils import print_slurm_info
from GravNN.Trajectories import SurfaceDist


def main():
    print_slurm_info()
    planet = Eros()
    obj_file = planet.obj_200k

    trajectory = SurfaceDist(
        planet,
        obj_file=obj_file,
    )
    start_time = time.time()
    Polyhedral(planet, obj_file, trajectory=trajectory).load()
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
