from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.utils import generate_orex_orbit_trajectories


def main():
    planet = Bennu()
    obj_file = planet.stl_200k

    # Training data
    trajectories = generate_orex_orbit_trajectories(sampling_inteval=60 * 10)
    for trajectory in trajectories:
        Polyhedral(planet, obj_file, trajectory=trajectory).load()

    # Entire distribution
    trajectory = RandomDist(
        planet,
        [0, planet.radius * 3],
        points=20000,
        obj_file=obj_file,
    )
    Polyhedral(planet, obj_file, trajectory=trajectory).load()

    # trajectory = RandomDist(planet, [0, planet.radius], points=20000,
    #  obj_file=obj_file)
    # gravity_model = Polyhedral(planet, obj_file, trajectory=trajectory).load()

    # trajectory = RandomDist(planet, [planet.radius, planet.radius*3],
    #  points=20000,
    # obj_file=obj_file)
    # gravity_model = Polyhedral(planet, obj_file, trajectory=trajectory).load()

    # trajectory = SurfaceDist(planet, obj_file=obj_file)
    # gravity_model = Polyhedral(planet, obj_file, trajectory=trajectory).load()


if __name__ == "__main__":
    main()
