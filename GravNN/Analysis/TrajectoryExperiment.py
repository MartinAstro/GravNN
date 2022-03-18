import os

from numpy.lib.polynomial import poly
from scipy.integrate import solve_bvp, solve_ivp
import numpy as np
from GravNN.Support.transformations import invert_projection, cart2sph
import GravNN
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from FrozenOrbits.boundary_conditions import *
from FrozenOrbits.bvp import solve_bvp_pos_problem
from FrozenOrbits.coordinate_transforms import *
from FrozenOrbits.gravity_models import pinnGravityModel, polyhedralGravityModel
from FrozenOrbits.ivp import solve_ivp_pos_problem
from FrozenOrbits.LPE import LPE
import OrbitalElements.orbitalPlotting as op
from FrozenOrbits.utils import (
    check_solution_validity,
    compute_period,
    get_energy,
    get_initial_orbit_guess,
    get_S_matrix,
    get_solution_metrics,
    sample_safe_trad_OE,
    Solution
)
from FrozenOrbits.visualization import plot_1d_solutions, plot_3d_solutions
from GravNN.CelestialBodies.Asteroids import Eros
from scipy.integrate import solve_bvp

from OrbitalElements.orbitalPlotting import plot1d



class TrajectoryExperiment:
    def __init__(self, test_grav_model, truth_grav_model, radius_bounds, t_mesh_density=100, random_seed=1234, x0=None):
        self.radius_bounds = radius_bounds
        self.positions = None
        self.test_accelerations = None
        self.test_potentials = None

        self.predicted_accelerations = None
        self.predicted_potentials = None

        self.percent_error_acc = None
        self.percent_error_pot = None
       
        self.test_grav_model = test_grav_model
        self.truth_grav_model = truth_grav_model

        self.t_mesh_density = t_mesh_density
        self.x0 = x0
        np.random.seed(random_seed)

    def generate_initial_condition(self):
        # Set the initial conditions and dynamics via OE
        if self.x0 is not None:
            trad_OE = cart2oe_tf(self.x0.reshape((1,6)), self.truth_grav_model.planet.mu).numpy()
        else:
            trad_OE = sample_safe_trad_OE(self.radius_bounds[0], self.radius_bounds[1])
        T = compute_period(self.truth_grav_model.planet.mu, trad_OE[0, 0])
        x0  = np.hstack(oe2cart_tf(trad_OE, self.truth_grav_model.planet.mu))
        t_mesh = np.linspace(0, T, self.t_mesh_density)
        return x0, t_mesh
        
    def generate_trajectory(self, model, X0, t_eval):
        def fun(x,y,IC=None):
            "Return the first-order system"
            R = np.array([y[0:3]])
            V = np.array([y[3:6]])
            a = model.generate_acceleration(R)
            dxdt = np.hstack((V, a)).squeeze()
            return dxdt
        
        sol = solve_ivp(fun, [0, t_eval[-1]], X0.reshape((-1,)), t_eval=t_eval, atol=1e-8, rtol=1e-10)
        return sol

    def generate_accelerations(self):
        true_acc = self.truth_grav_model.generate_acceleration(self.true_sol.y[0:3,:])
        test_acc = self.test_grav_model.generate_acceleration(self.true_sol.y[0:3,:])
        return true_acc, test_acc

    def generate_potentials(self):
        true_pot = self.truth_grav_model.generate_potential(self.true_sol.y[0:3,:])
        test_pot = self.test_grav_model.generate_potential(self.true_sol.y[0:3,:])
        return true_pot, test_pot

    def diff_and_error(self, true, test):
        diff = np.linalg.norm(true - test, axis=-1)
        percent_error = diff/np.linalg.norm(true, axis=-1) * 100
        return diff, percent_error

    def difference_trajectories(self):
        return Solution(self.test_sol.y - self.true_sol.y, self.t_mesh)

    def run(self):
        self.x0, self.t_mesh = self.generate_initial_condition()

        # Generate trajectories using the true and test grav models
        self.true_sol = self.generate_trajectory(self.truth_grav_model, self.x0, self.t_mesh)
        self.test_sol = self.generate_trajectory(self.test_grav_model, self.x0, self.t_mesh)
        self.diff_sol = self.difference_trajectories()

        # Evaluate the accelerations and potentials along the trajectory from the true
        # trajectory 
        self.true_acc, self.test_acc = self.generate_accelerations()
        self.true_pot, self.test_pot = self.generate_potentials()

        # Differences and error (1D)
        self.diff_acc, self.error_acc = self.diff_and_error(self.true_acc, self.test_acc)
        self.diff_pot, self.error_pot = self.diff_and_error(self.true_pot, self.test_pot)




def main():

    from GravNN.Visualization.TrajectoryVisualizer import TrajectoryVisualizer
    planet = Eros()
    min_radius = planet.radius * 5
    max_radius = planet.radius * 7

    # Load in the gravity model
    pinn_model = pinnGravityModel(
        os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_pinn_III_030822.data"
    )  
    poly_model = polyhedralGravityModel(planet, planet.obj_8k)

    init_state = np.array([
        4.61513747e+04, 
        8.12741755e+04, 
        -1.00860719e+04, 
        8.49819800e-01,
        -1.49764060e+00,
        2.47435298e+00
        ])

    experiment = TrajectoryExperiment(pinn_model, poly_model, [min_radius, max_radius], x0=init_state)
    experiment.run()
    vis = TrajectoryVisualizer(experiment, shape_model=planet.obj_8k)
    vis.plot_trajectories()
    vis.plot_trajectory_error()
    vis.plot_acceleration_differences()
    vis.plot_potential_differences()   
    vis.plot_trajectory_differences()
    vis.plot_trajectories_1d()
    plt.show()





if __name__ == "__main__":
    main()
