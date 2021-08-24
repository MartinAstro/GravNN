from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import RandomAsteroidDist, SurfaceDist

def main():
    planet = Eros()
    model_file = planet.obj_200k

    # Training data
    trajectory = RandomAsteroidDist(planet, [0, planet.radius*3], points=20000, model_file=model_file)
    #gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()

    # Analysis Data
    trajectory = SurfaceDist(planet, obj_file=model_file)
    #gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()
    
    trajectory = RandomAsteroidDist(planet, [0, planet.radius], points=20000, model_file=model_file)
    #gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()

    trajectory = RandomAsteroidDist(planet, [planet.radius, planet.radius*3], points=20000, model_file=model_file)
    #gravity_model = Polyhedral(planet, model_file, trajectory=trajectory).load()

if __name__ == "__main__":
    main()