from lib2to3.pytree import convert
import os
from abc import ABC, abstractmethod

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def convert_string(string):
    string = string.replace(".","_")
    string = string.replace("[", "_")
    string = string.replace("]", "_")
    string = string.replace(" ", "")
    string = string.replace(",", "_")
    return string

class VisualizationBase(ABC):

    def __init__(self, save_directory=None, halt_formatting=False, formatting_style=None, **kwargs):
        """Default visualization base class. Generates consistent style formatting for all 
        figures, offers information about proper figure sizes.

        Args:
            save_directory (str, optional): path to directory in which all figures will be saved. Defaults to None.
            halt_formatting (bool, optional): flag determining if custom formatting should be applied. Defaults to False.
        """
        if save_directory is None:
            self.file_directory = os.path.abspath('.') +"/Plots/"
        else:
            self.file_directory = save_directory


        # (width, height)
        golden_ratio = (5**.5 - 1) / 2
        silver_ratio = 1/(1 + np.sqrt(2)) 

        self.full_page = (6.3, 6.3*golden_ratio) 
        self.half_page = (3.15, 3.15*golden_ratio) #
        self.tri_page = (2.1, 2.1*golden_ratio)

        # AAS textwidth is 6.5
        if formatting_style == 'AIAA':
            self.AIAA_full_page = (6.5, 6.5*golden_ratio)

        # Set default figure size
        self.fig_size = (5,3.5) #(3, 1.8) is half page. 

        # default figure styling
        plt.rc('font', size= 10.0)
        plt.rc('font', family='serif')
        plt.rc('axes', prop_cycle=mpl.cycler(color=[
            'blue', 'green', 'red', 
            'orange', 'gold',  'salmon',  
            'lime', 'magenta','lavender', 
            'yellow', 'black', 'lightblue',
            'darkgreen', 'pink', 'brown',  
            'teal', 'coral',  'turquoise',  
            'tan', 'gold']))
        plt.rc('axes.grid', axis='both')
        plt.rc('axes.grid', which='both')
        plt.rc('axes', grid=True)
        plt.rc('grid', linestyle='--')
        plt.rc('grid', linewidth='0.1')
        plt.rc('grid', color='.25')
        plt.rc('text', usetex=True)

        if halt_formatting:
            plt.rc('text', usetex=False)
  
        return

    def new3DFig(self, unit='m', **kwargs):
        figsize = kwargs.get('fig_size', self.fig_size)
        fig = plt.figure(num=None, figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r'$x$ ('+unit+r')')
        ax.set_ylabel(r'$y$ ('+unit+r')')
        ax.set_zlabel(r'$z$ ('+unit+r')')
        # ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        # ax.zaxis.set_major_locator(plt.MaxNLocator(6))
        # ax.tick_params(labelsize=10)
        ax.legend(prop={'size': 10})
        ax.get_legend().remove()
        return fig, ax

    def newFig(self, fig_size=None):
        if fig_size is None:
            fig_size = self.fig_size
        fig = plt.figure(num=None, figsize=fig_size)
        ax = fig.add_subplot(111)
        return fig, ax

    def save(self, fig, name):
        #If the name is an absolute path -- save to the path
        plt.tight_layout()
        if os.path.isabs(name):
            try:
                plt.figure(fig.number)
                plt.savefig(name, bbox_inches='tight')
            except Exception as e:
                print("Couldn't save " + name)
                print(e)
            return    
        
        # If not absolute, create a directory and save the plot
        directory = os.path.abspath(os.path.dirname(self.file_directory + name))
        directory = convert_string(directory)

        filename =  os.path.basename(name)
        os.makedirs(directory, exist_ok=True)
        try:
            plt.savefig(directory+"/" + filename, bbox_inches='tight')
        except:
            print("Couldn't save " + filename)

