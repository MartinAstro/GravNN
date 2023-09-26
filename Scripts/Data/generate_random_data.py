import os
import time

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import RandomDist


def main():
    num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    cores_per_nodes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split("(")[0])
    num_threads = os.cpu_count()

    print(f"Cores Per Node:{os.environ['SLURM_JOB_CPUS_PER_NODE']}")
    print(f"Num Nodes:{os.environ['SLURM_JOB_NUM_NODES']}")
    print(f"Available Cores:{num_nodes * cores_per_nodes}")
    print(f"Threads per core:{num_threads / (cores_per_nodes * num_nodes)}")

    # print(f"Num Threads:{mp.cpu_count()}")
    planet = Eros()
    model_file = planet.obj_200k

    # Training data
    trajectory = RandomDist(
        planet,
        [0, planet.radius * 10],
        points=1000000,
        shape_model=model_file,
    )
    start_time = time.time()
    Polyhedral(planet, model_file, trajectory=trajectory).load()
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
