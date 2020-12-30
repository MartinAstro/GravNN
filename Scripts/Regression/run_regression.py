
import numpy as np
from numpy.random import seed

from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Trajectories.RandomDist import RandomDist

from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics

import os, sys
from build.PinesAlgorithm import PinesAlgorithm
from build.Regression import Regression
from GravNN.Support.transformations import cart2sph, sphere2cart, project_acceleration, invert_projection

def removeA0(positions, accelerations, mu):
    method = 1
    if method == 0:
        pos_sphere = cart2sph(positions)
        acceleration_sphere = project_acceleration(pos_sphere, accelerations)
        rMag = np.linalg.norm(positions, axis=1)
        a0 = -mu/rMag**2
        acceleration_sphere[:,0] -= a0
        acceleration_cart = invert_projection(pos_sphere, acceleration_sphere)
        return acceleration_cart
    else:
        pos_sphere = cart2sph(positions)
        r = pos_sphere[:,0]
        theta = pos_sphere[:,1] - 180.0
        phi  = pos_sphere[:,2]

        a0 = -mu/r**2
        
        acc_sub_sph = [a0, theta, phi]
        acc_sub_sph = np.transpose(acc_sub_sph)
        acc_sub_cart = sphere2cart(acc_sub_sph)
        acc_cart = accelerations - acc_sub_cart
        return acc_cart

def compute_coef_error(CS_regress, C_lm, S_lm):
    """Compute the average error of the regressed coefficients

    Args:
        CS_regress (np.array): [Nx2] array of regressed coefficients
        C_lm (np.array): [lower-tri] true C_lm coefficients 
        S_lm (np.array): [lower-tri] true S_lm coefficents

    Returns:
        [float]: average error 
    """
    idx = 0
    N= len(CS_regress.C_lm)+1# Number of coefficients regressed
    error_list = np.zeros((N*(N+1),))
    for l in range(len(C_lm)): # l max
        for m in range(len(C_lm[l])): # m max
            try:
                error_list[l*(l+1) + 2*m] = abs((CS_regress.C_lm[l][m]- C_lm[l][m])/C_lm[l][m])
                error_list[l*(l+1) + 2*m+1] =  abs((CS_regress.S_lm[l][m] - S_lm[l][m])/S_lm[l][m])
            except:
                pass
    return np.nanmean(error_list)

def print_C2m_error(l_start, regress_coef, gravityModel):
    print(gravityModel.C_lm[2])
    print(gravityModel.S_lm[2])
    error = np.zeros((3,2))
    if l_start == 0:
        regress_coef = regress_coef[3:,:]
    print(regress_coef)
    error[0, 0] = (regress_coef[0,0] - gravityModel.C_lm[2][0])/gravityModel.C_lm[2][0]
    error[0, 1] = (regress_coef[0,1] - gravityModel.S_lm[2][0])/gravityModel.S_lm[2][0]
    error[1, 0] = (regress_coef[1,0] - gravityModel.C_lm[2][1])/gravityModel.C_lm[2][1]
    error[1, 1] = (regress_coef[1,1] - gravityModel.S_lm[2][1])/gravityModel.S_lm[2][1]
    error[2, 0] = (regress_coef[2,0] - gravityModel.C_lm[2][2])/gravityModel.C_lm[2][2]
    error[2, 1] = (regress_coef[2,1] - gravityModel.S_lm[2][2])/gravityModel.S_lm[2][2]
    print(error*100)
    return

def main():
    planet = Earth()
    trajectory = UniformDist(planet, planet.radius, 100)
    coef = Regression.Coefficients()
    #trajectory = RandomDist(planet, [planet.radius+100, planet.radius+500], 10000) 

    # Full fidelity model
    max_deg = 2
    gravityModel = SphericalHarmonics(planet.sh_file, degree=max_deg, trajectory=trajectory)
    accelerations = gravityModel.load()

    # partial fidelity model
    l_start = 0
    if l_start == 2:
        grav_model_partial = SphericalHarmonics(planet.sh_file, degree=0, trajectory=trajectory)
        acc_low_fidelity = grav_model_partial.load()
        accelerations = accelerations - acc_low_fidelity


    # Attempt to regress coefficients and compute precent error
    deg_list = [max_deg]
    tolerance = 1E-6 #1.7E-6 switches between 1 and max iterations
    error_bound = 10 # percent

    pos = trajectory.positions.reshape(-1)
    acc =  accelerations.reshape(-1)
    deg = int(max_deg)
    rad = planet.radius
    mu = planet.mu
    regression = Regression.Regression(pos, 
                                        acc, 
                                        deg, 
                                        rad, 
                                        mu)
    for deg in deg_list:
        regression = Regression.Regression(trajectory.positions.reshape(-1), 
                                                                            accelerations.reshape(-1), 
                                                                            int(deg), 
                                                                            planet.radius, 
                                                                            planet.mu)
        regress_coef = regression.perform_regression(l_start, tolerance)
        #regress_coef = np.array(regression.result).reshape((-1,2)) #[Mx2] coefficient list
        
        model_error = compute_coef_error(regress_coef, gravityModel.C_lm, gravityModel.S_lm)
        print("Model Error for l=" + str(deg) + " is: " + str(model_error))

        if model_error > error_bound:
            break



    #TODO: We need to save the results of the regression into gravity files. 


if __name__ == "__main__":
    main()