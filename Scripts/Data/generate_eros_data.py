import time

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import RandomDist, SurfaceDist


def main():
    planet = Eros()
    model_file = planet.obj_8k

    # Training data
    trajectory = RandomDist(
        planet,
        [0, planet.radius * 3],
        points=200,
        shape_model=model_file,
    )
    Polyhedral(planet, model_file, trajectory=trajectory).load(override=True)

    trajectory = SurfaceDist(
        planet,
        obj_file=model_file,
    )

    start_time = time.time()
    Polyhedral(planet, model_file, trajectory=trajectory).load(override=True)
    print(f"Total time: {time.time() - start_time}")

    # # Analysis Data
    # trajectory = SurfaceDist(planet, obj_file=model_file)
    # Polyhedral(planet, model_file, trajectory=trajectory).load()

    # trajectory = RandomAsteroidDist(
    #     planet,
    #     [0, planet.radius],
    #     points=20000,
    #     model_file=model_file,
    # )
    # Polyhedral(planet, model_file, trajectory=trajectory).load()

    # trajectory = RandomAsteroidDist(
    #     planet,
    #     [planet.radius, planet.radius * 3],
    #     points=20000,
    #     model_file=model_file,
    # )
    # Polyhedral(planet, model_file, trajectory=trajectory).load()

    # trajectories = generate_near_orbit_trajectories(sampling_inteval=60 * 10)
    # for trajectory in trajectories:
    #     Polyhedral(planet, model_file, trajectory=trajectory).load()


if __name__ == "__main__":
    main()
