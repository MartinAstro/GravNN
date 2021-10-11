import numpy as np
import os 
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import format_coefficients, populate_removed_degrees, save
from GravNN.Trajectories import RandomAsteroidDist, DHGridDist, EphemerisDist
from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Trajectories.utils import generate_orex_hopper_trajectories, generate_near_orbit_trajectories, generate_orex_orbit_trajectories
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


def preprocess_data(x_dumb, a_dumb, acc_noise):

    x_dumb = np.array(x_dumb)
    a_dumb = np.array(a_dumb)

    # (Optionally) Add noise
    a_mag = np.linalg.norm(a_dumb, axis=1).reshape(len(a_dumb), 1)
    a_unit = np.random.uniform(-1, 1, size=np.shape(a_dumb))
    a_unit = a_unit / np.linalg.norm(a_unit, axis=1).reshape(len(a_unit), 1)
    a_error = acc_noise * a_mag * a_unit  # 10% of the true magnitude
    a_dumb = a_dumb + a_error

    return x_dumb, a_dumb

def BLLS_SH(regress_deg, remove_deg, sampling_interval,include_hoppers=False):
    # This has the true SH coef of Bennu -- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.998.2986&rep=rep1&type=pdf
    planet = Bennu()
    model_file = planet.stl_200k

    N = regress_deg  
    M = remove_deg 

    trajectories = generate_orex_orbit_trajectories(sampling_inteval=sampling_interval)
    
    hopper_trajectories = generate_orex_hopper_trajectories(sampling_inteval=sampling_interval)
    
    # Initialize the regressor
    regressor = BLLS(N, planet, M)

    # Record time history of regressor
    x_hat_hist = []
    t_hist = []

    x_train = []
    y_train = []

    total_samples = 0
    hopper_samples = 0
    remove_point_mass = False if M == -1 else True
    pbar = ProgressBar(len(trajectories), enable=True)

    for k in range(len(trajectories)):
        trajectory = trajectories[k]
        x, a, u = get_poly_data(
            trajectory,
            model_file,
            remove_point_mass=[remove_point_mass],
            override=[False],
        )
        try:
            for i in range(len(x)):
                x_train.append(x[i])
                y_train.append(a[i])
        except:
            x_train = np.concatenate((x_train, x))
            y_train = np.concatenate((y_train, a))


        # Don't include the hoppers in the sample count because those samples are used to compute the times
        # in the plotting routines.
        if include_hoppers:
            hop_trajectory = hopper_trajectories[k]
            x_hop, a_hop, u_hop = get_poly_data(
                hop_trajectory, planet.stl_200k, remove_point_mass=[False]
            )
            hopper_samples += len(x_hop)
            x_train = np.concatenate((x_train, x_hop))
            y_train = np.concatenate((y_train, a_hop))
       

        total_samples = len(x_train) - hopper_samples

        x_train_sample, y_train_sample = preprocess_data(x_train, y_train, acc_noise=0.1)

        x_hat = regressor.update(x_train_sample, y_train_sample)
        x_hat_hist.append(x_hat)
        t_hist.append(trajectory.times)
        time = trajectory.times[0]
  
        pbar.update(k)
        C_lm, S_lm = format_coefficients(x_hat, N, M)
        file_name = os.path.curdir + "/GravNN/Files/GravityModels/Regressed/%s/%s/BLLS/N_%d/M_%d/%s/%d.csv" % (
            planet.__class__.__name__,
            trajectory.__class__.__name__,
            N,
            M,
            str(include_hoppers),
            total_samples
        )
        save(file_name, planet, C_lm, S_lm)
    plot_coef_history(t_hist, x_hat_hist, M, start_idx=0)
    
    #plt.show()


def main():
    # # 10 minute sample interval
    BLLS_SH(4, 0, 10*60, include_hoppers=True)
    BLLS_SH(8, 0, 10*60, include_hoppers=True)
    BLLS_SH(16, 0, 10*60, include_hoppers=True)

    # BLLS_SH(4, 0, 10*60, include_hoppers=False)
    # BLLS_SH(8, 0, 10*60, include_hoppers=False)
    # BLLS_SH(16, 0, 10*60, include_hoppers=False)

    # 1 minute sample interval
    # BLLS_SH(4, 0, 1*60)
    # BLLS_SH(8, 0, 1*60)
    # BLLS_SH(16, 0, 1*60)

if __name__ == "__main__":
    main()
