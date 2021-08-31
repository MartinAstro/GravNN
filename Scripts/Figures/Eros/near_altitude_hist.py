from numpy.core.fromnumeric import ndim
from GravNN.Support.transformations import cart2sph
import numpy as np
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import format_coefficients, populate_removed_degrees, save
from GravNN.Trajectories import RandomAsteroidDist, DHGridDist, EphemerisDist
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Trajectories.utils import generate_near_orbit_trajectories
from GravNN.Support.ProgressBar import ProgressBar
import matplotlib.pyplot as plt


def main():
    # This has the true SH coef of Eros -- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.998.2986&rep=rep1&type=pdf
    planet = Eros()
    model_file = planet.obj_200k
    directory = "/Users/johnmartin/Documents/GraduateSchool/Research/ML_Gravity/GravNN/Files/GravityModels/Regressed/"
    trajectories = generate_near_orbit_trajectories(sampling_inteval=10*60)

    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        r, a, u = get_poly_data(
            trajectory,
            model_file,
            remove_point_mass=[False],
            override=[False],
        )

        try: 
            x = np.concatenate((x,r),axis=0)
        except:
            x = r

    r = cart2sph(np.array(x)) 
    plt.figure()
    plt.hist(r[:,0],bins=100)
    plt.show()

if __name__ == "__main__":
    main()
