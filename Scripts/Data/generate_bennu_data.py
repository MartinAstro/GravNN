from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories.RandomAsteroidDist import RandomAsteroidDist
from GravNN.Trajectories.SurfaceDist import SurfaceDist
from GravNN.Trajectories.utils import generate_orex_orbit_trajectories
def main():

    planet = Bennu()
    model_file = planet.stl_200k

    # Training data
    trajectories = generate_orex_orbit_trajectories(sampling_inteval=60*10)
    for trajectory in trajectories:
        gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()

    # Entire distribution
    trajectory = RandomAsteroidDist(planet, [0, planet.radius*3], points=20000, model_file=model_file)
    gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()

    # trajectory = RandomAsteroidDist(planet, [0, planet.radius], points=20000, model_file=model_file)
    # gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()

    # trajectory = RandomAsteroidDist(planet, [planet.radius, planet.radius*3], points=20000, model_file=model_file)
    # gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()

    # trajectory = SurfaceDist(planet, obj_file=model_file)
    # gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()



if __name__ == "__main__":
    main()