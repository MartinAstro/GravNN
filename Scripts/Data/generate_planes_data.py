import time

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.slurm_utils import print_slurm_info
from GravNN.Trajectories import PlanesDist


def main():
    print_slurm_info()
    planet = Eros()
    obj_file = planet.obj_200k

    # # Training data
    # trajectory = PlanesDist(
    #     planet,
    #     [-10*planet.radius, planet.radius * 10],
    #     samples_1d=200,
    # )
    # start_time = time.time()
    # Polyhedral(planet, obj_file, trajectory=trajectory).load()
    # print(f"Total time: {time.time() - start_time}")

    trajectory = PlanesDist(
        planet,
        [-3 * planet.radius, planet.radius * 3],
        samples_1d=200,
    )
    start_time = time.time()
    Polyhedral(planet, obj_file, trajectory=trajectory).load()
    print(f"Total time: {time.time() - start_time}")

    trajectory = PlanesDist(
        planet,
        [-planet.radius, planet.radius],
        samples_1d=200,
    )
    start_time = time.time()
    Polyhedral(planet, obj_file, trajectory=trajectory).load()
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
