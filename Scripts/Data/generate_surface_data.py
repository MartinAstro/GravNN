import time

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.slurm_utils import print_slurm_info
from GravNN.Trajectories import SurfaceDist


def main():
    print_slurm_info()
    planet = Eros()
    model_file = planet.obj_200k

    trajectory = SurfaceDist(
        planet,
        obj_file=model_file,
    )
    start_time = time.time()
    Polyhedral(planet, model_file, trajectory=trajectory).load()
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
