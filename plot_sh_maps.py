from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization
from GravityModels.SphericalHarmonics import SphericalHarmonics
from CelestialBodies.Planets import Earth
from Trajectories.DHGridDist import DHGridDist
from Trajectories.RandomDist import RandomDist
from Trajectories.UniformDist import UniformDist
from Support.transformations import cart2sph
import os
import pyshtools
import matplotlib.pyplot as plt
import numpy as np

def plot_grid_on_map():
    # Plot Grid Points on Perturbations
    planet = Earth()
    radius = planet.radius
    model_file = planet.sh_hf_file
    # Specify the grid density via the degree
    density_deg = 175
    max_deg = 1000
    trajectory_surf = DHGridDist(planet, radius, degree=density_deg)

    Call_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_surf)
    Call_grid = Grid(gravityModel=Call_gm)

    C20_gm = SphericalHarmonics(model_file, 2, trajectory=trajectory_surf)
    C20_grid = Grid(gravityModel=C20_gm)

    map_vis = MapVisualization()
    true_mC20_grid = Call_grid - C20_grid
    point_count_list = [1000, 10000]
    size = 0.1
    for point_count in point_count_list:
        trajectory_random = RandomDist(planet, [planet.radius, planet.radius*1.1], point_count)
        trajectory_uniform = UniformDist(planet, planet.radius, point_count)

        pos_sphere_random = cart2sph(trajectory_random.positions)
        pos_sphere_uniform = cart2sph(trajectory_uniform.positions)

        # Scale down to 360 and 180 respectively
        fig_pert, ax = map_vis.plot_grid(true_mC20_grid.total, "Acceleration Perturbations")
        ax.scatter(pos_sphere_random[:,1]/np.max(pos_sphere_random[:,1])*(len(true_mC20_grid.total)-1),
                            pos_sphere_random[:,2]/np.max(pos_sphere_random[:,2])*(len(true_mC20_grid.total[1]-1)),
                            color='r', s=size)
        map_vis.save(fig_pert, str(point_count) + "_Random_Grid_SH.png")      

        fig_pert, ax = map_vis.plot_grid(true_mC20_grid.total, "Acceleration Perturbations")
        ax.scatter(pos_sphere_uniform[:,1]/np.max(pos_sphere_uniform[:,1])*(len(true_mC20_grid.total)-1),
                            pos_sphere_uniform[:,2]/np.max(pos_sphere_uniform[:,2])*(len(true_mC20_grid.total[1]-1)),
                            color='r', s=size)
        map_vis.save(fig_pert, str(point_count) + "_Uniform_Grid_SH.png")      


def plot_sh_perturbations():
    # Phase 0: Plot Perturbations beyond C20
    planet = Earth()
    radius = planet.radius
    model_file = planet.sh_hf_file
    # Specify the grid density via the degree
    density_deg = 175
    max_deg = 1000
    trajectory_surf = DHGridDist(planet, radius, degree=density_deg)

    Call_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_surf)
    Call_grid = Grid(gravityModel=Call_gm)

    C20_gm = SphericalHarmonics(model_file, 2, trajectory=trajectory_surf)
    C20_grid = Grid(gravityModel=C20_gm)

    map_vis = MapVisualization()
    true_mC20_grid = Call_grid - C20_grid
    fig_total, ax = map_vis.plot_grid(Call_grid.total, "Acceleration Total")
    fig_pert, ax = map_vis.plot_grid(true_mC20_grid.total, "Acceleration Perturbations")
    map_vis.save(fig_total, "Full_SH_Acc.pdf")
    map_vis.save(fig_pert, "Full_SH_Acc_Pert.pdf")        


def plot_sh_v_altitude():
    # Phase 1: Plot Percent Map at varying altitudes
    planet = Earth()
    radius = planet.radius
    model_file = planet.sh_hf_file
    # Specify the grid density via the degree
    density_deg = 175
    max_deg = 1000
    trajectory_surf = DHGridDist(planet, radius, degree=density_deg)

    trajectory_leo = DHGridDist(planet, radius + 2000*1000, degree=density_deg) 
    trajectory_meo = DHGridDist(planet, radius + 10000*1000, degree=density_deg) 
    trajectory_geo = DHGridDist(planet, radius + 20200*1000, degree=density_deg) 
    traj_list = [trajectory_leo, trajectory_meo, trajectory_geo]

    map_vis = MapVisualization()
    for traj in traj_list:
        Call_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=traj)
        C20_gm = SphericalHarmonics(model_file, 2, trajectory=traj)
        Call_grid = Grid(Call_gm)
        C20_grid = Grid(C20_gm)
        map_vis.percent_maps(Call_grid, C20_grid, vlim=[0,100])

def phase_2():
        # Phase 2: Plot Percent Error at Different Degree Models
        planet = Earth()
        radius = planet.radius
        model_file = planet.sh_hf_file
        # Specify the grid density via the degree
        density_deg = 175
        max_deg = 1000
        trajectory_surf = DHGridDist(planet, radius, degree=density_deg)
        
        Call_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_surf)
        Call_grid = Grid(gravityModel=Call_gm)

        C20_gm = SphericalHarmonics(model_file, 2, trajectory=trajectory_surf)
        C20_grid = Grid(gravityModel=C20_gm)

        Call_grid -= C20_grid

        map_vis = MapVisualization()
        error_list = []
        coefficient_list = []
        degree_list = [5, 10, 25, 50]
        for deg in degree_list:
            Clm_gm = SphericalHarmonics(model_file, degree=deg, trajectory=trajectory_surf)
            Clm_grid = Grid(gravityModel=Clm_gm)
            Clm_grid -= C20_grid
            fig_abs_err, fig_rel_err = map_vis.percent_maps(Call_grid, Clm_grid, vlim=[0, 100])#, C20_grid, vlim=10)
            map_vis.save(fig_abs_err, str(deg) + "_SH_Abs_Error.pdf")
            map_vis.save(fig_rel_err, str(deg) + "_SH_Rel_Error.pdf")
            coefficient_list.append(deg*(deg+1))
            error_list.append(np.median(np.abs(100*((Clm_grid - Call_grid)/Call_grid).total)))
        map_vis.newFig()
        plt.plot(coefficient_list, error_list)
        plt.xlabel("N Coefficients in Model")
        plt.ylabel("Median Percent Error")
        map_vis.save(plt.gcf(), "SH_Rel_Error_2D.pdf")



def main():

    planet = Earth()
    radius = planet.radius
    model_file = planet.sh_hf_file
    # Specify the grid density via the degree
    density_deg = 175
    max_deg = 1000
    trajectory_surf = DHGridDist(planet, radius, degree=density_deg)
    

    plot_grid_on_map()
    #plot_sh_perturbations()
    #plot_sh_v_altitude()
    #phase_2()

    plt.show()
    # grid_list = []
    # deg_list = [5, 10, 20, 50, 100, 150]

    # deg_list = [10, 50, 100]
    # for i in range(len(deg_list)):
    #     grid = Grid(gravityModel=SphericalHarmonics(model_file, deg_list[i], trajectory=trajectory))
    #     map_vis.percent_error_maps(Call_grid, grid, None, vlim=100)
    #     grid_list.append(grid)
    # fig, ax = map_vis.component_error(grid_list, Call_grid, deg_list, "blue", C20_grid) 
    # plt.xlabel(r"SH Degree, $l$")
    # plt.show()




if __name__ == "__main__":
    main()
