import os
import numpy as np
import sys
import copy
path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path+'/../support')
sys.path.append(path+'/../nnPosAcc')


#from Data_IO import get_coef, extractInputs
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numpy.random import seed
from time import time
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pyshtools
from mpl_toolkits.axes_grid1 import make_axes_locatable


from VisualizationBase import VisualizationBase
from Interfaces.SHGridInterface import SHGridInterface

seed(1)

class CoefficientVisualization(VisualizationBase):
    def __init__(self, coef_file):
        super().__init__()
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.full_SH = pyshtools.SHGravCoeffs.from_file(coef_file, errors=False, header_units='m')
        return

    def computeError(self, x_truth, x_recreated):
        error = 0.0
        for i in range(len(x_truth)):
            for j in range(len(x_truth[0])):
                error += abs(x_recreated[i][j] - x_truth[i][j]) / \
                         abs(x_truth[i][j])
        error /= len(x_truth) * len(x_truth[0])
        return error

    def plot_acceleration_error(self, degree, error_bound):
        # Plot difference of SH and PCA on same color scale
        full_grid= copy.deepcopy(self.full_SH.expand())
        subset_grid = copy.deepcopy(self.full_SH.expand(lmax_calc=degree))

        full_grid.total.data -= subset_grid.total.data
        full_grid.total.data = np.divide(full_grid.total.data, self.full_SH.expand().total.data) * 100.0

        grid = full_grid.total.data
        self.newFig()
        im = plt.imshow(grid, vmin=-error_bound, vmax=error_bound)
        tick_interval = [30,30]
        yticks = np.linspace(-90, 90, num=180//tick_interval[1]+1, endpoint=True, dtype=int)
        xticks = np.linspace(0, 360, num=360//tick_interval[0]+1, endpoint=True, dtype=int)
        xloc = np.linspace(0, len(grid[0]), num=len(xticks), endpoint=True, dtype=int)
        yloc = np.linspace(0, len(grid), num=len(yticks), endpoint=True, dtype=int)

        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.xticks(xloc,labels=xticks)
        plt.yticks(yloc, labels=yticks)
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)

        cBar = plt.colorbar(im, cax=cax)
        cBar.ax.set_ylabel( 'Acceleration, $\%$ Error')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.

        return

    def plot_residual_acceleration(self, degree):
        full_grid= copy.deepcopy(self.full_SH.expand())
        subset_grid = copy.deepcopy(self.full_SH.expand(lmax_calc=degree))

        full_grid.rad.data -= subset_grid.rad.data
        full_grid.phi.data -= subset_grid.phi.data
        full_grid.theta.data -= subset_grid.theta.data
        full_grid.total.data -= subset_grid.total.data

        full_grid.plot_total()
        return plt.gcf()
    
        


def main():
    file_name = "/Users/johnmartin/Basilisk/supportData/LocalGravData/GGM03S.txt"
    #file_name = "/Users/johnmartin/Documents/GraduateSchool/Research/SH_GPU/GravityCompute/EGM2008_to2190_TideFree_E.txt"
    
    full_SH = pyshtools.SHGravCoeffs.from_file(file_name, errors=False, header_units='m')

    '''
    positions = SHGridInterface.getPositions(sh_grid, radius)
    accelerations = SH_Acceleration(positions) #TODO: This will become a swig wrapper of the C++ algorithm 
    SHGridInterface.assignAcceleration(sh_grid, accelerations)
    '''

    vis = CoefficientVisualization(file_name)
    fig = vis.plot_residual_acceleration(degree=10)
    plt.show()
    #vis.save(fig, "Acceleration.pdf")

    quantity = "Acceleration"
    quantity = "Potential"
    saveFig = True
    errorBound = 100




    # Get the Data
    if quantity == "Potential":
        truth_data = clm_all.expand().pot.data # Get all coef representation
    elif quantity == "Acceleration":
        truth_data = clm_all.expand().total.data  # Get all coef representation
    else:
        print("Wrong Quantity Dummy")
        exit()
    pca_data = None
    shape = np.shape(truth_data)

    '''
    Compute the individual Potential field layers and make them 1-D layers
    '''
    x_train = []
    for i in range(1,clm_all.lmax):
        if i == 1:
            if quantity == "Acceleration":
                diff = clm_all.expand(lmax_calc=i-1).total.data
            else:
                diff = clm_all.expand(lmax_calc=i-1).pot.data
        else:
            if quantity == "Acceleration":
                diff = clm_all.expand(lmax_calc=i).total.data - clm_all.expand(lmax_calc=i - 1).total.data
            else:
                diff = clm_all.expand(lmax_calc=i).pot.data - clm_all.expand(lmax_calc=i - 1).pot.data

        diff = diff.reshape(-1)
        x_train.append(diff)


    '''
    Plot the error of the SH and the PCA representation
    '''
    dimList = [2, 4, 6, 8, 10, 12, 14]
    maxList = np.array(dimList)
    errorList = []
    for max in maxList:
        clm_all_grid = clm_all.expand()
        clm_lx_grid = clm_all.expand(lmax_calc=max)

        truth_data = clm_all_grid.total.data
        clm_all_grid.total.data = clm_lx_grid.total.data - clm_all_grid.total.data #Error
        #lm_all_grid.plot_total(show=False)
        sh_data = clm_lx_grid.total.data

        error = analysis.computeError(truth_data, sh_data)
        print("SH Coef: " + str(max*(max+1)) + "\t Error: " + str(error*100) + "%")
        errorList.append(error*100)

    fig = plt.gcf()
    plt.plot(maxList*(maxList+1), errorList)
    plt.legend(['PCA','SH'])
    plt.savefig(plotPath + quantity + "_PCA_vs_SH_error.pdf",bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()