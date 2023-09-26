import numpy as np

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Trajectories import RandomDist, SurfaceDist


def compute_acceleration(trajectory):
    x, a, u = get_sh_data(
        trajectory,
        Eros().sh_file,
        max_deg=4,
        deg_removed=-1,
        override=[True],
    )
    return a


def compute_stats(trajectory, prefix, obj_file):
    x, a, u = get_poly_data(trajectory, obj_file)

    a_pred = compute_acceleration(trajectory)

    diff = a - a_pred
    rse = np.linalg.norm(diff, axis=1) ** 2

    # Percent error as a function of difference in acceleration magnitude
    percent = np.linalg.norm(a - a_pred, axis=1) / np.linalg.norm(a, axis=1) * 100

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
    exterior_bound = planet.radius * 3
    obj_file = planet.obj_200k
    trajectory = SurfaceDist(planet, obj_file)
    stats = {}
    stats.update(compute_stats(trajectory, "surface", obj_file))

    trajectory = RandomDist(
        planet,
        [0, interior_bound],
        20000,
        obj_file,
    )
    stats.update(compute_stats(trajectory, "interior", obj_file))

    trajectory = RandomDist(
        planet,
        [interior_bound, exterior_bound],
        20000,
        obj_file,
    )
    stats.update(compute_stats(trajectory, "exterior", obj_file))
    print(stats)


if __name__ == "__main__":
    main()
