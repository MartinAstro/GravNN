import numpy as np
from sigfig import round
from GravNN.Regression.RLLS import RLLS
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import format_coefficients, populate_removed_degrees, save
from GravNN.Trajectories import RandomAsteroidDist, DHGridDist
from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data

from GravNN.Support.ProgressBar import ProgressBar
import matplotlib.pyplot as plt

def print_coef(C_lm, sigfigs=3):
    np.set_printoptions(precision=sigfigs)
    print(C_lm)

def plot_coef_history(x_hat_hist, P_hat_hist, sh_EGM2008, remove_deg, start_idx=0):
    x_hat_hist = np.array(x_hat_hist)
    P_hat_hist = np.array(P_hat_hist)

    l = remove_deg + 1
    m = 0

    for i in range(len(x_hat_hist[0])):
        plt.figure()
        plt.plot(x_hat_hist[start_idx:,i], c='b')
        plt.plot(x_hat_hist[start_idx:,i] + 3*np.sqrt(P_hat_hist[start_idx:,i]), c='r')
        plt.plot(x_hat_hist[start_idx:,i] - 3*np.sqrt(P_hat_hist[start_idx:,i]), c='r')

        letter = "C" if i % 2 == 0 else "S"
        plt.suptitle(letter+str(l)+str(m))

        if i % 2 != 0:
            if m < l:
                m += 1
            else:
                l += 1
                m = 0

def Kaula_zonal(deg):
    K_zonal_bennu = 0.084
    K_zonal_guess = 1E-2

    K_zonal = K_zonal_guess

    alpha_zonal = 2.08
    if deg != 0:
        kaula = K_zonal/deg**alpha_zonal
    else:
        kaula = K_zonal
    return kaula

def Kaula_RMS(deg):
    K_RMS_bennu = 0.026
    K_RMS_guess = 1E-2

    K_RMS = K_RMS_guess
    alpha_RMS = 2.01
    if deg != 0:
        kaula = K_RMS/deg**alpha_RMS
    else:
        kaula = K_RMS
    return kaula

def initialize_covariance(N, M):
    P0 = np.identity((N+2)*(N+1) - (M+2)*(M+1))
    P0[np.isnan(P0)] = 0.0
    # https://www.hou.usra.edu/meetings/lpsc2016/pdf/2129.pdf (Kaula's rule for Asteroids)
    K = 1E5
    K = 1E5
    K = 1

    deg = M + 1
    order = 0

    for i in range(0, (N+2)*(N+1) - (M+2)*(M+1)):
        if order != 0:
            P0[i,i] = Kaula_RMS(deg) 
        else:
            P0[i,i] = Kaula_zonal(deg) 
        
        order += 1
        if order > deg:
            deg += 1
            order = 0
        # print(P0[i,i])
    return P0

def initialize_state_est(batch_initialization, N, planet, M, x, batch_size, a, C_lm_start):
    if batch_initialization:
        batch_regressor = BLLS(N, planet, M)
        idx = np.random.choice(np.arange(len(x)),batch_size, replace=False)
        results = batch_regressor.update(x[idx,:], a[idx, :])
        x0 = results
    else:
        x0 = np.zeros(((N+2)*(N+1) - (M+2)*(M+1),))
        x0[0] = C_lm_start
    C_lm, S_lm = format_coefficients(x0, N, M)
    print_coef(C_lm)
    print(S_lm)
    return x0

def main():
    # This has the true SH coef of Bennu -- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.998.2986&rep=rep1&type=pdf
    planet = Bennu()
    model_file = planet.model_potatok
    regress_deg = 4
    remove_deg = -1

    samples = 1000

    batch_initialization = True
    batch_size = 60000
    sh_EGM2008 = Polyhedral(planet, model_file)
    #trajectory = RandomAsteroidDist(planet, [0, planet.radius+10000], 10000, grav_file=[model_file])
    #trajectory = RandomAsteroidDist(planet, [planet.radius, planet.radius+10000], 10000, grav_file=[model_file])
    trajectory = RandomAsteroidDist(planet, [planet.radius*2, planet.radius*3], 10000, grav_file=[model_file]) # confidently outside of Brill sphere
    trajectory = DHGridDist(planet, planet.radius*2, 90) # confidently outside of Brill sphere

    remove_point_mass = False if remove_deg == -1 else True
    x, a, u = get_poly_data(trajectory,model_file,remove_point_mass=[remove_point_mass], override=[False])
    C_lm_start = 0.0 if remove_point_mass else 1.0

    N = regress_deg 
    M = remove_deg 

    # Initial Coefficient Estimates and Covariances
    # N+1 accounts for the C00 terms
    x0 = initialize_state_est(batch_initialization, N, planet, M, x, batch_size, a, C_lm_start)
    P0 = initialize_covariance(N, M)
    Rk = np.identity(3)*np.std(a)/3.0

    # Initialize the regressor
    regressor = RLLS(N, planet, x0, P0, Rk, M)

    # Record time history of regressor
    x_hat_hist = []
    P_hat_hist = []

    # pbar = ProgressBar(len(x), enable=True)
    # i = 0
    # for x_inst, y_inst in zip(x, a):
    #     regressor.update(x_inst, y_inst)
    #     i +=1
    #     pbar.update(i)

    #     x_hat_hist.append(regressor.x_hat)
    #     P_hat_hist.append(np.diag(regressor.P_hat).tolist())
    
    #     if (i - 1) % samples == 0:
    #         C_lm, S_lm = format_coefficients(regressor.x_hat, regressor.N, regressor.remove_deg)
    #         if remove_deg != -1:
    #             C_lm[0,0] = 1.0
    #         file_name = '/Users/johnmartin/Documents/GraduateSchool/Research/ML_Gravity/GravNN/Files/GravityModels/Regressed/%s/%s/RLLS_%d_%d_%d.csv' % (planet.__class__.__name__,trajectory.__class__.__name__, N, M, i-1)
    #         save(file_name, planet, C_lm, S_lm)

    # plot_coef_history(x_hat_hist, P_hat_hist, sh_EGM2008, remove_deg, start_idx=0)
    # plt.show()






if __name__  == "__main__":
    main()
