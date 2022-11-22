from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Trajectories import SurfaceDist, RandomAsteroidDist
import numpy as np

def generate_acceleration(trajectory):
    x, a, u = get_sh_data(trajectory, Eros().sh_file, max_deg=4, deg_removed=-1,override=[True])
    return a

def compute_stats(trajectory, prefix, model_file):
        x, a, u = get_poly_data(trajectory, model_file)

        a_pred = generate_acceleration(trajectory)

        diff = a - a_pred
        rse = np.linalg.norm(diff, axis=1) ** 2
        
        # Percent error as a function of difference in acceleration magnitude
        percent = np.linalg.norm(a - a_pred, axis=1)/np.linalg.norm(a, axis=1)*100


        stats = {
            prefix + "_rse_mean": np.mean(rse),
            prefix + "_rse_std": np.std(rse),
            prefix + "_rse_max": np.max(rse),
            prefix + "_percent_mean": np.mean(percent),  # error
            prefix + "_percent_std": np.std(percent),  # error
            prefix + "_percent_max": np.max(percent),  # error
        }
        return stats

def main():
    planet = Eros()
    interior_bound = planet.radius
    exterior_bound = planet.radius*3
    model_file = planet.obj_200k
    trajectory = SurfaceDist(planet, model_file)
    stats = {}
    stats.update(compute_stats(trajectory, "surface", model_file))

    trajectory = RandomAsteroidDist(
        planet, [0, interior_bound], 20000, model_file
    )
    stats.update(compute_stats(trajectory, "interior", model_file))

    trajectory = RandomAsteroidDist(
        planet,
        [interior_bound, exterior_bound],
        20000,
        model_file,
    )
    stats.update(compute_stats(trajectory, "exterior", model_file))
    print(stats)

if __name__ == "__main__":
    main()