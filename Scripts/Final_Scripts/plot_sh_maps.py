from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Support.transformations import cart2sph
import os
import pyshtools
import matplotlib.pyplot as plt
import numpy as np
import pickle

mapUnit = "m/s^2"
mapUnit = 'mGal'
map_vis = MapVisualization(unit=mapUnit)
#map_vis.fig_size = (3, 1.8)

# Plot Grid Points on Perturbations
planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
# Specify the grid density via the degree
density_deg = 175
max_deg = 1000
trajectory_surf = DHGridDist(planet, radius, degree=density_deg)
Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_surf)
Call_r0_grid = Grid(trajectory=trajectory_surf, accelerations=Call_r0_gm.load())
C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory_surf)
C20_r0_grid = Grid(trajectory=trajectory_surf, accelerations=C20_r0_gm.load())
R0_pert_grid = Call_r0_grid - C20_r0_grid

trajectory_leo = DHGridDist(planet, radius + 330*1000, degree=density_deg) 
Call_leo_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_leo)
Call_leo_grid = Grid(trajectory=trajectory_leo, accelerations=Call_leo_gm.load())
C20_leo_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory_leo)
C20_leo_grid = Grid(trajectory=trajectory_leo, accelerations=C20_leo_gm.load())
LEO_pert_grid = Call_leo_grid - C20_leo_grid


def plot_sh_model():
    fig_pert, ax = map_vis.plot_grid(Call_r0_grid.total, "Acceleration [mGal]")
    map_vis.save(fig_pert,"Full_SH_Grid.pdf")      

    fig_pert, ax = map_vis.plot_grid(C20_r0_grid.total, "Acceleration [mGal]")
    map_vis.save(fig_pert,"C22_SH_Grid.pdf")      

def plot_training_dist_map(point_count):
    """Plot the training distribution on top of the perturbation map
    """
    size = 0.1
    trajectory_random = RandomDist(planet, [planet.radius-2500, planet.radius+2500], point_count)
    trajectory_uniform = UniformDist(planet, planet.radius, point_count)

    pos_sphere_random = cart2sph(trajectory_random.positions)
    pos_sphere_uniform = cart2sph(trajectory_uniform.positions)

    # Scale down to 360 and 180 respectively
    fig_pert, ax = map_vis.plot_grid(R0_pert_grid.total, "Perturbations")
    ax.scatter(pos_sphere_random[:,1]/np.max(pos_sphere_random[:,1])*(len(R0_pert_grid.total)-1),
                        pos_sphere_random[:,2]/np.max(pos_sphere_random[:,2])*(len(R0_pert_grid.total[1]-1)),
                        color='r', s=size)
    map_vis.save(fig_pert, str(point_count) + "_Random_Grid_SH.png")      
    map_vis.save(fig_pert, str(point_count) + "_Random_Grid_SH.pdf")      

    fig_pert, ax = map_vis.plot_grid(R0_pert_grid.total, "Perturbations")
    ax.scatter(pos_sphere_uniform[:,1]/np.max(pos_sphere_uniform[:,1])*(len(R0_pert_grid.total)-1),
                        pos_sphere_uniform[:,2]/np.max(pos_sphere_uniform[:,2])*(len(R0_pert_grid.total[1]-1)),
                        color='r', s=size)
    map_vis.save(fig_pert, str(point_count) + "_Uniform_Grid_SH.png")      
    map_vis.save(fig_pert, str(point_count) + "_Uniform_Grid_SH.pdf")      

def plot_r0_leo_different_vlim():
    """
    Plot R0 and LEO at different vlims
    """
    fig_pert, ax = map_vis.plot_grid(R0_pert_grid.total, "Perturbations [mGal]")
    map_vis.save(fig_pert, "Full_SH_Acc_Pert.pdf")        

    map_vis.newFig()
    fig_pert = map_vis.new_map(R0_pert_grid.total,vlim=[0,10])
    map_vis.add_colorbar(fig_pert, "Perturbations [mGal]",vlim=[0,10], extend='max')
    map_vis.save(fig_pert, "Full_SH_Acc_Pert_[0,10].pdf")        

    map_vis.newFig()
    fig_pert = map_vis.new_map(LEO_pert_grid.total,vlim=[0,10])
    map_vis.add_colorbar(fig_pert, "Perturbations [mGal]",vlim=[0,10], extend='max')
    map_vis.save(fig_pert, "Full_SH_LEO_Acc_Pert_[0,10].pdf")        

    map_vis.newFig()
    fig_pert = map_vis.new_map(LEO_pert_grid.total,vlim=[0,5])
    map_vis.add_colorbar(fig_pert, "Perturbations [mGal]",vlim=[0,5], extend='max')
    map_vis.save(fig_pert, "Full_SH_LEO_Acc_Pert_[0,5].pdf")        

def plot_sh_pert_and_masks_r0_LEO():
    """
    Generates the maps of the perturbations at both altitudes
    Generates the mask of the perturbations at both altitudes
    """
    fig_pert = map_vis.new_map(R0_pert_grid.total,vlim=[0,10])
    map_vis.add_colorbar(fig_pert, "Perturbations [mGal]",vlim=[0,10], extend='max')
    map_vis.save(plt.gcf(), "R0_Pert.pdf")

    fig, ax = map_vis.plot_grid(LEO_pert_grid.total, 'Perturbations [mGal]')
    map_vis.save(fig, "LEO_Pert.pdf")

    std = np.std(R0_pert_grid.total)
    mask = R0_pert_grid.total > 3*std
    #fig_mask, ax = map_vis.plot_grid(R0_pert_grid.total*mask, "[mGal]")
    map_vis.newFig()
    fig_mask = map_vis.new_map(R0_pert_grid.total*mask,vlim=[0,10])
    map_vis.add_colorbar(fig_mask, "Perturbations [mGal]",vlim=[0,10], extend='max')
    map_vis.save(fig_mask, "R0_SH_Mask.pdf")

    std = np.std(LEO_pert_grid.total)
    mask = LEO_pert_grid.total > 3*std
    fig_mask, ax = map_vis.plot_grid(LEO_pert_grid.total*mask, "Perturbations [mGal]")
    map_vis.save(fig_mask, "LEO_SH_Mask.pdf")

def plot_rmse_v_deg(plot_save_coef):        
        rmse_list = []
        rmse_feat_list = []
        coefficient_list = []
        degree_list = [5, 10, 18, 25, 31, 40, 50, 65, 80, 100]
        degree_list = [10, 31, 100, 447]

        std = np.std(R0_pert_grid.total)
        mask = R0_pert_grid.total > 3*std

        for deg in degree_list:
            Clm_gm = SphericalHarmonics(model_file, degree=deg, trajectory=trajectory_surf)
            Clm_grid = Grid(trajectory=trajectory_surf, accelerations=Clm_gm.load())
            Clm_grid -= C20_r0_grid
            fig_rmse = map_vis.plot_grid_rmse(Clm_grid, R0_pert_grid, vlim=[0, 5], log_scale=False)
            map_vis.save(fig_rmse, str(deg) + "_SH_RMSE.pdf")
            coefficient_list.append(deg*(deg+1))
            rmse_list.append(np.average(np.sqrt(np.square(Clm_grid.total - R0_pert_grid.total))))
            rmse_feat_list.append(np.average(np.sqrt(np.square((Clm_grid.total - R0_pert_grid.total))),weights=mask))

        if plot_save_coef:
            map_vis.newFig()
            plt.semilogy(coefficient_list, rmse_list, linestyle='-', color='b', label='SH Map')

            fontSize = 12
            ax = plt.gca()
            ax.tick_params(labelsize=fontSize)
            #cbar.set_label('Speedup', fontsize=fontSize)
            ax.set_ylabel("Average RMSE", fontsize=fontSize)
            ax.set_xlabel("N Coefficients in Model", fontsize=fontSize)
            
            plt.semilogy(coefficient_list, rmse_feat_list, linestyle='--', color='b', label='SH Feature')
            plt.legend()
            map_vis.save(plt.gcf(), "SH_RMSE_2D.pdf")
            with open('./Files/Results/SH_RMSE.data', 'wb') as f:
                pickle.dump(coefficient_list, f)
                pickle.dump(rmse_list, f)
                pickle.dump(rmse_feat_list, f)

def plot_sh_multi_deg():        
        degree_list = [5, 10, 18, 25, 31, 40, 50, 65, 80, 100]
        degree_list = [10, 31, 100, 447]

        for deg in degree_list:
            Clm_gm = SphericalHarmonics(model_file, degree=deg, trajectory=trajectory_surf)
            Clm_grid = Grid(trajectory=trajectory_surf, accelerations=Clm_gm.load())
            Clm_grid -= C20_r0_grid
            fig_rmse = map_vis.plot_grid(Clm_grid.total, "Perturbations [mGal]", vlim=[0, 10])
            map_vis.save(fig_rmse, str(deg) + "_SH.pdf")


def plot_rmse_v_deg_at_leo():        
        rmse_list = []
        rmse_feat_list = []
        coefficient_list = []
 
        std = np.std(LEO_pert_grid.total)
        mask = LEO_pert_grid.total > 3*std

        degree_list = [5, 10, 25, 50, 180, 500]
        degree_list = [5, 50, 500]
        degree_list = [10, 31, 100]
        degree_list = [5, 10, 18, 25, 31, 40, 50, 65, 80, 100]
        for deg in degree_list:
            Clm_gm = SphericalHarmonics(model_file, degree=deg, trajectory=trajectory_leo)
            Clm_grid = Grid(trajectory=trajectory_leo, accelerations=Clm_gm.load())
            Clm_grid -= C20_leo_grid

            fig_rmse = map_vis.plot_grid_rmse(Clm_grid, LEO_pert_grid, vlim=[0, 0.75], log_scale=False)
            map_vis.save(fig_rmse, str(deg) + "_SH_LEO_RMSE.pdf")

            coefficient_list.append(deg*(deg+1))
            rmse_list.append(np.average(np.sqrt(np.square(Clm_grid.total - LEO_pert_grid.total))))
            rmse_feat_list.append(np.average(np.sqrt(np.square((Clm_grid.total - LEO_pert_grid.total))),weights=mask))

        map_vis.newFig()
        plt.semilogy(coefficient_list, rmse_list, linestyle='-', color='b', label='SH Map')
        
        fontSize = 12
        ax = plt.gca()
        ax.tick_params(labelsize=fontSize)
        #cbar.set_label('Speedup', fontsize=fontSize)
        ax.set_ylabel("Average RMSE", fontsize=fontSize)
        ax.set_xlabel("N Coefficients in Model", fontsize=fontSize)

        # plt.xlabel("N Coefficients in Model")
        # plt.ylabel("Average RMSE")
        plt.semilogy(coefficient_list, rmse_feat_list, linestyle='--', color='b', label='SH Feature')
        #ax2.tick_params(axis='y', labelcolor=color)
        plt.legend()
        map_vis.save(plt.gcf(), "SH_RMSE_LEO_2D.pdf")

        with open('./Files/Results/SH_LEO_RMSE.data', 'wb') as f:
            pickle.dump(coefficient_list, f)
            pickle.dump(rmse_list, f)
            pickle.dump(rmse_feat_list, f)

def plot_rmse_hist_mask():
        std = np.std(R0_pert_grid.total)
        mask = R0_pert_grid.total > 3*std

        fig, ax = map_vis.newFig()
        plt.hist(R0_pert_grid.total.reshape(-1), bins=30)
        map_vis.save(plt.gcf(), "R0_SH_Hist.pdf")

        fig, ax = map_vis.newFig()
        plt.hist(LEO_pert_grid.total.reshape(-1), bins=30)
        map_vis.save(plt.gcf(), "LEO_SH_Hist.pdf")
       
def main():
    # plot_sh_model()
    #plot_sh_multi_deg()
    #plot_training_dist_map(259200)
    #plot_r0_leo_different_vlim()
    plot_sh_pert_and_masks_r0_LEO()
    # plot_sh_v_altitude()
    # percent_error_maps()
    # plot_rmse_v_deg(plot_save_coef=False)
    # plot_rmse_v_deg_at_leo()
    # plot_rmse_hist_mask()
    # plt.show()




if __name__ == "__main__":
    main()
