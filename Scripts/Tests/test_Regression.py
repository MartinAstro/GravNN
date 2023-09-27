from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Trajectories import DHGridDist


def main():
    planet = Earth()
    trajectory = DHGridDist(planet, planet.radius, 45)
    x, a, u = get_sh_data(trajectory, planet.sh_file, 180, -1)

    regressor = Regression(5, planet, x, a)
    coefficients = regressor.perform_regression()
    regressor.save(
        "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\Regressed\\test.csv",
    )
    print(coefficients)


if __name__ == "__main__":
    main()
