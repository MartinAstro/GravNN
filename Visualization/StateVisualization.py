import utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(__file__) + "/../")
from mpl_toolkits.mplot3d import Axes3D
from GravNN.Visualization.VisualizationBase import  VisualizationBase
from CelestialBodies.Planets import Earth
from GravNN.Trajectories.UniformDist import UniformDist

class StateVisualization(VisualizationBase):
    def __init__(self):
        super().__init__()
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        pass

    def plotScatter(self, trajectory):
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


    def plotPlanetView(self, trajectory):
        fig, ax = self.new3DFig()
        ax.scatter(trajectory.positions[:,0], trajectory.positions[:,1], trajectory.positions[:,2], c='r')
        ax.scatter([0], [0], [0], c='b', s=100)
        return


def run():
    degree = 2
    planet = Earth()
    trajectory = UniformDist(planet, planet.radius, 100)
    stateVis = StateVisualization()
    stateVis.plotPlanetView(trajectory)
    plt.show()


if __name__ == '__main__':
    run()