from ..Support import transformations
from copy import deepcopy
import numpy as np
from numba import jit

import sys, os
sys.path.append(os.path.dirname(__file__) + "/../")
from ..Support.transformations import cart2sph, sphere2cart, project_acceleration, invert_projection, check_fix_radial_precision_errors
from ..GravityModels.NN_Base import NN_Base



class StateObject(object):

    def __init__(self, gravityModel, override=False):

        self.N = gravityModel.trajectory.points 

        # Generate shapes for grids
        acc_cart = gravityModel.load(override=override)

        pos_sph = cart2sph(gravityModel.trajectory.positions)
        pos_sph = check_fix_radial_precision_errors(pos_sph)
        acc_sph = transformations.project_acceleration(pos_sph, acc_cart)

        self.positions = pos_sph
        self.accelerations = acc_sph

        self.total = np.linalg.norm(self.accelerations,axis=1).reshape((self.N,))
        self.r = self.accelerations[:,0].reshape((self.N,))
        self.theta = self.accelerations[:,1].reshape((self.N,))
        self.phi = self.accelerations[:,2].reshape((self.N,))


    def __sub__(self, other):
        newStateObj = deepcopy(self)
        newStateObj.accelerations -= other.accelerations

        newStateObj.r =newStateObj.accelerations[:,0].reshape((newStateObj.N))
        newStateObj.theta = newStateObj.accelerations[:,1].reshape((newStateObj.N))
        newStateObj.phi = newStateObj.accelerations[:,2].reshape((newStateObj.N))

        #newStateObj.total = newStateObj.total - other.total
        newStateObj.total = np.linalg.norm(newStateObj.accelerations,axis=1).reshape((newStateObj.N))

        return newStateObj

    def __truediv__(self, other):
        newStateObj = deepcopy(self)

        newStateObj.accelerations = np.divide(newStateObj.accelerations, other.accelerations)

        newStateObj.r =newStateObj.accelerations[:,0].reshape((newStateObj.N))
        newStateObj.theta = newStateObj.accelerations[:,1].reshape((newStateObj.N))
        newStateObj.phi = newStateObj.accelerations[:,2].reshape((newStateObj.N))

        newStateObj.total =  np.divide(newStateObj.total, other.total)
        #np.linalg.norm(newStateObj.accelerations,axis=1).reshape((newStateObj.points))

        return newStateObj

    def __mul__(self, other):
        newStateObj = deepcopy(self)
        try:
            newStateObj.total *= other.total
            newStateObj.r *= other.r
            newStateObj.theta *= other.theta
            newStateObj.phi *= other.phi
        except:
            newStateObj.total *= other
            newStateObj.r *= other
            newStateObj.theta *= other
            newStateObj.phi *= other
        return newStateObj





