from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories import RandomDist


def main():
    planet = Earth()
    sh_file = planet.sh_file
    bounds = [planet.radius, planet.radius + 420000.0]
    trajectory = RandomDist(planet, bounds, points=5000000)
    gravity_model = SphericalHarmonics(
        sh_file,
        degree=1000,
        trajectory=trajectory,
        parallel=False,
    )
    gravity_model.load()


if __name__ == "__main__":
    main()
