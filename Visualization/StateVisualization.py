import utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from mpl_toolkits.mplot3d import Axes3D
from support import transformations
from Visualization.VisualizationBase import  VisualizationBase

class StateVisualization:
    def __init__(self, data, path=None):
        self.data = data
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        return

    def plotScatter(self):
        plt.figure(num=None, figsize=(15, 10.5), dpi=200)
        components = [r"_{r}$", r"_{\phi}$",  r"_{\theta}$"]
        for i in range(3):
            for j in range(3):
                plt.subplot(3, 3, 3*i + j+1)
                plt.scatter(self.data.r[:,i], self.data.a[:,j],s=2)
                plt.xlabel(r"$r"+components[i])
                plt.ylabel(r"$a"+components[j])
                plt.ylim([min(self.data.a[:,j]), max(self.data.a[:,j])])
        plt.tight_layout()
        if self.path is not None:
            plt.savefig(self.path + "_scatter.pdf", bbox_inches='tight')


    def plotPlanetView(self):
        fig = plt.figure(num=None)
        ax = fig.add_subplot(111,projection='3d')
        dataCart = transformations.sphere2cart(self.data.r)
        ax.scatter(dataCart[:,0], dataCart[:,1], dataCart[:,2], c='r')
        ax.scatter([0], [0], [0], c='b', s=100)
        if self.path is not None:
            plt.savefig(self.path + "_planetView.pdf", bbox_inches='tight')


def run():
    dataPath = "/Users/johnmartin/Documents/GraduateSchool/Research/GravityML/data/states/" 
    plotPath = "/Users/johnmartin/Documents/GraduateSchool/Research/GravityML/minGravitySet/Files/Plots/GGM/"

    if not os.path.exists(plotPath):
        os.makedirs(plotPath)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


   

    fileName = "N_1000_deg_175_dist_Uniform_alt_50000_random_Full/"
    #TODO: Fill in state
    state = None#StateData(file=dataPath + fileName + "state_0.data", removeC00=True)
    stateVis = StateVisualization(state, plotPath+fileName)
    stateVis.plotScatter()
    #stateVis.plotPlanetView()
    #plt.show()



if __name__ == '__main__':
    run()