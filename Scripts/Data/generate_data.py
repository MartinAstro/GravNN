import time

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.Support.slurm_utils import print_slurm_info
from GravNN.Trajectories import PlanesDist, RandomDist, SurfaceDist


def generate_random_data(planet, obj_file):
    # Characteristic Random Trajectories

    # Training data
    random_trajectory = RandomDist(
        planet,
        [0, planet.radius * 10],
        points=100000,
        obj_file=obj_file,
    )
    start_time = time.time()
    model = generate_heterogeneous_model(planet, obj_file, trajectory=random_trajectory)
    model.load()
    dt = time.time() - start_time
    print(f"Random Finished: {dt} [s]")

    R = planet.radius
    start_time = time.time()
    planes_trajectory = PlanesDist(
        planet,
        [-3 * R, 3 * R],
        samples_1d=200,
    )
    model = generate_heterogeneous_model(planet, obj_file, trajectory=planes_trajectory)
    model.load()
    dt = time.time() - start_time
    print(f"Planes Finished: {dt} [s]")

    surface_trajectory = SurfaceDist(
        planet,
        obj_file=obj_file,
    )
    model = generate_heterogeneous_model(
        planet,
        obj_file,
        trajectory=surface_trajectory,
    )
    model.load()
    print("Surface Finished")

    extrapolation_traj = RandomDist(
        planet,
        [0, 10 * R],
        points=4500,
        obj_file=obj_file,
    )
    model = generate_heterogeneous_model(
        planet,
        obj_file,
        trajectory=extrapolation_traj,
    )
    model.load()
    print("Extrapolation Finished 1")

    dr = 100 * R - 10 * R
    point_density = 4500 / (10 * R)
    extrap_points = int(dr * point_density)
    extrapolation_traj = RandomDist(
        planet,
        [10 * R, 100 * R],
        points=extrap_points,
        obj_file=obj_file,
    )
    model = generate_heterogeneous_model(
        planet,
        obj_file,
        trajectory=extrapolation_traj,
    )
    model.load()
    print("Extrapolation Finished 2")


if __name__ == "__main__":
    print_slurm_info()
    planet = Eros()
    obj_file = planet.obj_200k
    generate_random_data(planet, obj_file)
