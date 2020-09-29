from ..Support import transformations
from copy import deepcopy
import numpy as np

import sys, os
sys.path.append(os.path.dirname(__file__) + "/../")
from ..Support.transformations import cart2sph, sphere2cart, project_acceleration, invert_projection, check_fix_radial_precision_errors
from ..GravityModels.NN_Base import NN_Base



class Grid(object):
    lats = np.array([])
    lons = np.array([])

    def __init__(self, gravityModel, override=False):

        self.radius = gravityModel.trajectory.radius

        self.N_lat = gravityModel.trajectory.N_lat
        self.N_lon = gravityModel.trajectory.N_lon

        # Generate shapes for grids
        self.total = np.zeros((self.N_lon, self.N_lat))
        self.phi = np.zeros((self.N_lon, self.N_lat))
        self.theta = np.zeros((self.N_lon, self.N_lat))
        self.r = np.zeros((self.N_lon, self.N_lat))
        
        acc_cart = gravityModel.load(override=override)

        pos_sph = cart2sph(gravityModel.trajectory.positions)
        pos_sph = check_fix_radial_precision_errors(pos_sph)
        acc_sph = transformations.project_acceleration(pos_sph, acc_cart)

        self.positions = pos_sph
        self.acceleration = acc_sph

        self.total = np.linalg.norm(self.acceleration,axis=1).reshape((self.N_lon,self.N_lat))
        self.r = self.acceleration[:,0].reshape((self.N_lon,self.N_lat))
        self.theta = self.acceleration[:,1].reshape((self.N_lon,self.N_lat))
        self.phi = self.acceleration[:,2].reshape((self.N_lon,self.N_lat))


    def __sub__(self, other):
        newGrid = deepcopy(self)
        newGrid.acceleration -= other.acceleration

        newGrid.r =newGrid.acceleration[:,0].reshape((newGrid.N_lon,newGrid.N_lat))
        newGrid.theta = newGrid.acceleration[:,1].reshape((newGrid.N_lon,newGrid.N_lat))
        newGrid.phi = newGrid.acceleration[:,2].reshape((newGrid.N_lon,newGrid.N_lat))

        #newGrid.total = newGrid.total - other.total
        newGrid.total = np.linalg.norm(newGrid.acceleration,axis=1).reshape((newGrid.N_lon,newGrid.N_lat))


        return newGrid

    def __truediv__(self, other):
        newGrid = deepcopy(self)

        newGrid.acceleration = np.divide(newGrid.acceleration, other.acceleration)

        newGrid.r =newGrid.acceleration[:,0].reshape((newGrid.N_lon,newGrid.N_lat))
        newGrid.theta = newGrid.acceleration[:,1].reshape((newGrid.N_lon,newGrid.N_lat))
        newGrid.phi = newGrid.acceleration[:,2].reshape((newGrid.N_lon,newGrid.N_lat))

        newGrid.total =  np.divide(newGrid.total, other.total)
        #np.linalg.norm(newGrid.acceleration,axis=1).reshape((newGrid.N_lon,newGrid.N_lat))


        return newGrid

    def __mul__(self, other):
        newGrid = deepcopy(self)
        try:
            newGrid.total *= other.total
            newGrid.r *= other.r
            newGrid.theta *= other.theta
            newGrid.phi *= other.phi
        except:
            newGrid.total *= other
            newGrid.r *= other
            newGrid.theta *= other
            newGrid.phi *= other
        return newGrid





