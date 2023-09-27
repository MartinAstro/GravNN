from copy import deepcopy

import numpy as np


class StateObject(object):
    def __init__(self, trajectory, accelerations, override=False):
        self.positions = trajectory.positions
        self.accelerations = accelerations

        self.total = np.linalg.norm(self.accelerations, axis=1)
        self.r = self.accelerations[:, 0]
        self.theta = self.accelerations[:, 1]
        self.phi = self.accelerations[:, 2]

    def __sub__(self, other):
        newStateObj = deepcopy(self)
        newStateObj.accelerations -= other.accelerations

        newStateObj.r = newStateObj.accelerations[:, 0]
        newStateObj.theta = newStateObj.accelerations[:, 1]
        newStateObj.phi = newStateObj.accelerations[:, 2]

        # newStateObj.total = newStateObj.total - other.total
        newStateObj.total = np.linalg.norm(newStateObj.accelerations, axis=1)

        return newStateObj

    def __truediv__(self, other):
        newStateObj = deepcopy(self)

        newStateObj.accelerations = np.divide(
            newStateObj.accelerations,
            other.accelerations,
        )

        newStateObj.r = newStateObj.accelerations[:, 0]
        newStateObj.theta = newStateObj.accelerations[:, 1]
        newStateObj.phi = newStateObj.accelerations[:, 2]

        newStateObj.total = np.divide(newStateObj.total, other.total)
        # np.linalg.norm(newStateObj.accelerations,axis=1).reshape((newStateObj.points))

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
