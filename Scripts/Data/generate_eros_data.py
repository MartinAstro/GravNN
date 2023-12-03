import time

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Support.slurm_utils import print_slurm_info
from GravNN.Trajectories import PlanesDist, RandomDist, SurfaceDist
from GravNN.Trajectories.utils import (
    generate_near_hopper_trajectories,
    generate_near_orbit_trajectories,
)

RANDOM_N_LIST = [500, 5000, 50000, 500000]
RANDOM_MAX_RADIUS_LIST = [3, 10]
RANDOM_UNIFORM_LIST = [True, False]

PLANES_N_DENSITY_LIST = [10, 30, 50, 200]
PLANES_MAX_RADIUS_LIST = [3, 10]

NEAR_SAMPLING_INTERVAL = 60 * 10


def generate_random_data(planet, obj_file):
    # Characteristic Random Trajectories

    for N_random in RANDOM_N_LIST:
        for max_radius in RANDOM_MAX_RADIUS_LIST:
            for uniform in RANDOM_UNIFORM_LIST:
                # Training data
                trajectory = RandomDist(
                    planet,
                    [0, planet.radius * max_radius],
                    points=N_random,
                    obj_file=obj_file,
                    uniform_volume=uniform,
                )
                start_time = time.time()
                Polyhedral(planet, obj_file, trajectory=trajectory).load(override=False)
                print(f"Random Dist Params: {N_random}, {max_radius}, {uniform}")
                print(f"Total time: {time.time() - start_time}")

    # Surface Trajectory


def generate_surface_data(planet, obj_file):
    trajectory = SurfaceDist(
        planet,
        obj_file=obj_file,
    )
    start_time = time.time()
    Polyhedral(planet, obj_file, trajectory=trajectory).load()
    print(f"Total time: {time.time() - start_time}")


def generate_planes_data(planet, obj_file):
    for N_density in PLANES_N_DENSITY_LIST:
        for max_radius in RANDOM_MAX_RADIUS_LIST:
            start_time = time.time()
            trajectory = PlanesDist(
                planet,
                [-max_radius * planet.radius, planet.radius * max_radius],
                samples_1d=N_density,
            )
            Polyhedral(planet, obj_file, trajectory=trajectory).load(override=False)
            print(f"Planes Dist Params: {N_density}, {max_radius}")
            print(f"Total time: {time.time() - start_time}")


def generate_near_data(planet, obj_file):
    # spacecraft
    trajectories = generate_near_orbit_trajectories(
        sampling_inteval=NEAR_SAMPLING_INTERVAL,
    )
    for trajectory in trajectories:
        Polyhedral(
            planet,
            obj_file,
            trajectory=trajectory,
        ).load()
    # hoppers
    trajectories = generate_near_hopper_trajectories(
        sampling_inteval=NEAR_SAMPLING_INTERVAL,
    )
    for trajectory in trajectories:
        Polyhedral(
            planet,
            obj_file,
            trajectory=trajectory,
        ).load()


if __name__ == "__main__":
    print_slurm_info()
    planet = Eros()
    obj_file = planet.obj_66
    generate_random_data(planet, obj_file)
    generate_planes_data(planet, obj_file)
    generate_surface_data(planet, obj_file)

    obj_file = planet.obj_8k
    generate_random_data(planet, obj_file)
    generate_planes_data(planet, obj_file)
    generate_surface_data(planet, obj_file)
    # generate_near_data(planet, obj_file)

    obj_file = planet.obj_10k
    generate_random_data(planet, obj_file)
    generate_planes_data(planet, obj_file)
    generate_surface_data(planet, obj_file)

    obj_file = planet.obj_200k
    generate_random_data(planet, obj_file)
    generate_planes_data(planet, obj_file)
    generate_surface_data(planet, obj_file)
    # generate_near_data(planet, obj_file)
