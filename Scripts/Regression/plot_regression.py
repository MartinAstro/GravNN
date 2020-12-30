
import numpy as np
from numpy.random import seed

from GravNN.Support.Grid import Grid
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Trajectories.DHGridDist import DHGridDist
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
        # acc_sphere = cart2sph(accelerations)
        # acc_sphere[:,0] += a0
        #acc_cart = sphere2cart(acc_sphere)
        acc_cart = accelerations - acc_sub_cart
        return acc_cart
        # accelerations[3*i + 0] = accelerations[3*i + 0] - a0x
        # accelerations[3*i + 1] = accelerations[3*i + 1] - a0y
        # accelerations[3*i + 2] = accelerations[3*i + 2] - a0z


    # for i in range(int(len(positions)/3)):
    #     x, y, z = positions[3*i: 3*i+3]
    #     ax, ay, az = accelerations[3*i: 3*i+3]
    #     acceleration_sphere = project_acceleration(positions[3*i: 3*i+3], accelerations[3*i: 3*i+3])


    #     # posSphere = cart2sph([x,y,z])
    #     # r, theta, phi = posSphere[0,:]
    #     # theta -= 180
    #     r = np.sqrt(x**2 + y**2 + z**2)
    #     a0 = -mu/r**2
    #     acceleration_sphere[0] -= a0

    #     acceleration_cart = invert_projection(positions[3*i: 3*i+3], acceleration_sphere)
    #     accelerations[3*i:3*(i+1)] = acceleration_cart
        # sphereCoord = [[a0, theta, phi]]
        # posCart = sphere2cart(sphereCoord)
        # a0x, a0y, a0z = posCart[0,:]
        # accelerations[3*i + 0] = accelerations[3*i + 0] - a0x
        # accelerations[3*i + 1] = accelerations[3*i + 1] - a0y
        # accelerations[3*i + 2] = accelerations[3*i + 2] - a0z


def check_coef_error(regressed_coef, true_gravity_model, percent_error_threshold):
    """Determine if the regressed coefficients are less than a % error. If not, attempt to reregress with a lower coefficient set. 

    Args:
        regressed_coef (np.array): list of regressed coefficient values
        real_coefficients (np.array): list of true coefficient

    Returns:
        bool: if the error is less than the desired tolerance
    """
    total_error = 0.0
    N = len(true_gravity_model.C_lm)
    Clm_truth = np.zeros((int(N*(N+1)/2),))
    Slm_truth = np.zeros((int(N*(N+1)/2),))

    idx = 0
    for i in range(N):
        for j in range(len(true_gravity_model.C_lm[i])):
            if idx > int(len(regressed_coef)/2):
                break
            if i < 2:
                continue
            Clm_truth[idx] = true_gravity_model.C_lm[i][j]
            Slm_truth[idx] = true_gravity_model.S_lm[i][j]
            idx += 1

    
    zeros = 0
    for i in range(int(len(regressed_coef)/2)):
        if Clm_truth[i] == 0:
            zeros += 1
        else: 
            total_error += abs((regressed_coef[2*i] - Clm_truth[i])/Clm_truth[i])*100.0
        
        if Slm_truth[i] == 0:
            zeros += 1
        else:
            total_error += abs((regressed_coef[2*i+1] - Slm_truth[i])/Slm_truth[i])*100.0
    
    total_error = total_error/(len(regressed_coef)-zeros)
    print(total_error)
    return (total_error < percent_error_threshold)

def regress(gravityModel, deg_start, deg_regress, tolerance):
    positions = gravityModel.trajectory.positions
    planet = gravityModel.trajectory.celestial_body
    accelerations = gravityModel.load()
    if deg_start == 2:
        accelerations = removeA0(positions, accelerations, planet.mu)
    regression = Regression.Regression(positions.reshape(-1), accelerations.reshape(-1), int(deg_regress), planet.radius, planet.mu)
    regression.perform_regression(deg_start, tolerance)
    return regression.solution_exists and regression.solution_unique, regression

def compute_error(grid, grid_truth):
    error_grid = (grid - grid_truth)/grid_truth * 100
    errorList = np.zeros((4))
    errorList[0] = np.average(abs(error_grid.total))
    errorList[1] = np.average(abs(error_grid.r))
    errorList[2] = np.average(abs(error_grid.theta))
    errorList[3] = np.average(abs(error_grid.phi))

    return errorList



def main():
    # Plotting values
    degree_values = []
    N_data_values = []
    error_values = []

    #Initialization
    planet = Earth()
    deg_start = 2
    density_deg = 55

    deg_list = np.arange(5, density_deg, 20)
    tolerance = 10E-2
    points = 100

    # True gravity across grid
    trajectory_surface = DHGridDist(planet, planet.radius, degree=density_deg) 
    gravity_model_truth = SphericalHarmonics(planet.sh_file, None, trajectory=trajectory_surface)
    grid_truth = Grid(gravity_model_truth)


    # Attempt to regress coefficients from 5 to density_deg in increments of 20 adding points in until a solution is found
    for deg_regress in deg_list:
        unique_existing_solution = False
        while not unique_existing_solution:
            trajectory = UniformDist(planet, planet.radius, points)
            grav_model = SphericalHarmonics(planet.sh_file, None, trajectory=trajectory)
            unique_existing_solution, regression = regress(grav_model, deg_start, deg_regress, tolerance)
            grav_model_regress = SphericalHarmonics(regression.coef_regress, None, trajectory=trajectory_surface)
            grid_regress = Grid(grav_model_regress, override=True)

            # Provide enough points for a unique solution
            if points < deg_regress*(deg_regress+1):
                points = deg_regress*(deg_regress+1)
            else:
                points += 100
        
        degree_values.append(deg_regress)
        N_data_values.append(points)
        error_values.append(compute_error(grid_regress, grid_truth))


if __name__ == "__main__":
    main()