import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Support.transformations import cart2sph
from GravNN.Trajectories.utils import generate_near_orbit_trajectories


def main():
    # This has the true SH coef of Eros -- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.998.2986&rep=rep1&type=pdf
    planet = Eros()
    model_file = planet.obj_200k
    trajectories = generate_near_orbit_trajectories(sampling_inteval=10 * 60)

    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        r, a, u = get_poly_data(
            trajectory,
            model_file,
            remove_point_mass=[False],
            override=[False],
        )

        try:
            x = np.concatenate((x, r), axis=0)
        except:
            x = r

    r = cart2sph(np.array(x))
    plt.figure()
    plt.hist(r[:, 0], bins=100)
    plt.show()


if __name__ == "__main__":
    main()
