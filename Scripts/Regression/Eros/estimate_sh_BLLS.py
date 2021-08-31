import numpy as np
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import format_coefficients, populate_removed_degrees, save
from GravNN.Trajectories import RandomAsteroidDist, DHGridDist, EphemerisDist
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Trajectories.utils import generate_near_orbit_trajectories
from GravNN.Support.ProgressBar import ProgressBar
import matplotlib.pyplot as plt

def plot_coef_history(t_hist, x_hat_hist, remove_deg, start_idx=0):
    x_hat_hist = np.array(x_hat_hist)

    l = remove_deg + 1
    m = 0

    for i in range(len(x_hat_hist[0])):
        plt.figure()
        try:
            plt.plot(t_hist, x_hat_hist[start_idx:, i], c="b")
        except:
            plt.plot(x_hat_hist[start_idx:, i], c="b")
        letter = "C" if i % 2 == 0 else "S"
        plt.suptitle(letter + str(l) + str(m))

        if i % 2 != 0:
            if m < l:
                m += 1
            else:
                l += 1
                m = 0


def BLLS_SH(regress_deg, remove_deg, sampling_interval):
    # This has the true SH coef of Eros -- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.998.2986&rep=rep1&type=pdf
    planet = Eros()
    model_file = planet.model_data
    model_file = planet.model_7790
    directory = "/Users/johnmartin/Documents/GraduateSchool/Research/ML_Gravity/GravNN/Files/GravityModels/RegressedRealModel/"
    
    # planet.radius = planet.physical_radius
    model_file = planet.obj_200k
    directory = "/Users/johnmartin/Documents/GraduateSchool/Research/ML_Gravity/GravNN/Files/GravityModels/Regressed/"

    N = regress_deg  
    M = remove_deg 

    trajectories = generate_near_orbit_trajectories(sampling_inteval=sampling_interval)

    # Initialize the regressor
    regressor = BLLS(N, planet, M)

    # Record time history of regressor
    x_hat_hist = []
    t_hist = []

    r_batch = np.array([]).reshape(0,3)
    a_batch = np.array([]).reshape(0,3)

    remove_point_mass = False if M == -1 else True
    pbar = ProgressBar(len(trajectories), enable=True)

    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        r, a, u = get_poly_data(
            trajectory,
            model_file,
            remove_point_mass=[remove_point_mass],
            override=[False],
        )

        r_batch = np.vstack((r_batch, r))
        a_batch = np.vstack((a_batch, a))
        
        x_hat = regressor.update(r_batch, a_batch)
        x_hat_hist.append(x_hat)
        t_hist.append(trajectory.times)
        time = trajectory.times[0]
  
        pbar.update(k)
        C_lm, S_lm = format_coefficients(x_hat, N, M)
        file_name = "%s/%s/BLLS_%d_%d_%d_%d_%d.csv" % (
            planet.__class__.__name__,
            trajectory.__class__.__name__,
            N,
            M,
            len(r_batch),
            time,
            sampling_interval
        )
        save(directory + file_name, planet, C_lm, S_lm)
    plot_coef_history(t_hist, x_hat_hist, M, start_idx=0)
    
    #plt.show()


def main():
    # 10 minute sample interval
    BLLS_SH(4, 0, 10*60)
    BLLS_SH(8, 0, 10*60)
    BLLS_SH(16, 0, 10*60)

    # 1 minute sample interval
    # BLLS_SH(4, 0, 1*60)
    # BLLS_SH(8, 0, 1*60)
    # BLLS_SH(16, 0, 1*60)

if __name__ == "__main__":
    main()
