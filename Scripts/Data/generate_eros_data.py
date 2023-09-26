import time

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    generate_heterogeneous_sym_model,
)
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.slurm_utils import print_slurm_info
from GravNN.Trajectories import PlanesDist, RandomDist, SurfaceDist


def main():
    print_slurm_info()

    planet = Eros()
    obj_file = planet.obj_8k

    # # Training data
    # trajectory = RandomDist(
    #     planet,
    #     [0, planet.radius * 10],
    #     points=1000,
    #     obj_file=obj_file,
    # )
    # start_time = time.time()
    # # Polyhedral(planet, obj_file, trajectory=trajectory).load(override=True)
    # print(f"Total time: {time.time() - start_time}")

    trajectory = RandomDist(
        planet,
        [0, planet.radius * 3],
        points=60000,
        obj_file=obj_file,
    )
    generate_heterogeneous_sym_model(planet, obj_file, trajectory=trajectory).load()

    trajectory = SurfaceDist(
        planet,
        obj_file=obj_file,
    )
    start_time = time.time()
    Polyhedral(planet, obj_file, trajectory=trajectory).load()
    print(f"Total time: {time.time() - start_time}")

    trajectory = PlanesDist(
        planet,
        [-3 * planet.radius, planet.radius * 3],
        samples_1d=200,
    )


if __name__ == "__main__":
    main()
