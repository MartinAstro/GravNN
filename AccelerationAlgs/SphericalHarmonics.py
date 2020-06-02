import numpy as np
import math
import os

from Basilisk.simulation.gravityEffector import loadGravFromFileToList
from Basilisk.utilities import simIncludeGravBody

from AccelerationAlgs.AccelerationBase import AccelerationBase

class SphericalHarmonics(AccelerationBase):    
    def __init__(self, trajectory, degree):
        self.body = trajectory.celestial_body
        self.body.loadSH()
        self.trajectory = trajectory
        self.degree = degree
        super().__init__()
        pass

    def generate_full_file_directory(self):
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0]  + "_Deg" + str(self.degree) + "/"
        pass
    
    def compute_acc(self):

        def legendres(degree, alpha):
            P = np.zeros((degree+1,degree+1))
            P[0,0] = 1
            P[1,0] = alpha
            cosPhi = np.sqrt(1-alpha**2)
            P[1,1] = cosPhi

            for l in range(2,degree+1):
                for m in range(0,l+1):
                    if m == 0 and l >= 2:
                        P[l,m] = ((2*l-1)*alpha*P[l-1,0]-(l-1)*P[l-2,0]) / l
                    elif m != 0 and m < l:
                        P[l, m] = (P[l-2, m]+(2*l-1)*cosPhi*P[l-1,m-1])
                    elif m == l and l != 0:
                        P[l,m] = (2*l-1)*cosPhi*P[l-1,m-1]
                    else:
                        print(l,", ", m)
            return P

        cList = self.body.grav_info.SH.C_lm
        sList = self.body.grav_info.SH.S_lm
        mu = self.body.grav_info.mu
        rad = self.body.geometry.radius
        max_degree = self.degree

        rVec = self.trajectory.positions
        aVec = np.zeros(np.shape(rVec))

      
        for l in range(0, max_degree+1):
            for m in range(0,l+1):
                if m == 0:
                    k = 1
                else:
                    k = 2
                num = math.factorial(l+m)
                den = math.factorial(l-m)*k*(2*l+1)
                PI = np.sqrt(float(num)/float(den))
                cList[l][m] = cList[l][m] / PI
                sList[l][m] = sList[l][m] / PI

        for i in range(len(rVec)):
            rI = rVec[i][0]
            rJ = rVec[i][1]
            rK = rVec[i][2]

            r = np.linalg.norm(rVec)
            rHat = rVec[i] / r
            gHat = rHat
            grav0 = -gHat * mu / r**2


            rIJ = np.sqrt(rI**2 + rJ**2)
            if rIJ != 0.:
                phi = math.atan(rK / rIJ) #latitude in radians
            else:
                phi = math.copysign(np.pi/2., rK)
            if rI != 0.:
                lambdaSat = math.atan(rJ / rI) #longitude in radians
            else:
                lambdaSat = math.copysign(np.pi/2., rJ)

            P = legendres(max_degree+1,np.sin(phi))

            dUdr = 0.
            dUdphi = 0.
            dUdlambda = 0.

            for l in range(2,self.degree+1): #can only do for max degree minus 1
                for m in range(0,l+1):
                    dUdr = dUdr + (((rad/r)**l)*(l+1)*P[l,m]) * (cList[l][m]*np.cos(m*lambdaSat)+sList[l][m]*np.sin(m*lambdaSat))
                    dUdphi = dUdphi + (((rad/r)**l)*P[l,m+1] - m*np.tan(phi)*P[l,m]) * (cList[l][m]*np.cos(m*lambdaSat) + sList[l][m]*np.sin(m*lambdaSat))
                    dUdlambda = dUdlambda + (((rad/r)**l)*m*P[l,m]) * (sList[l][m]*np.cos(m*lambdaSat) - cList[l][m]*np.sin(m*lambdaSat))

            dUdr = -mu * dUdr / r**2
            dUdphi = mu * dUdphi / r
            dUdlambda = mu * dUdlambda / r

            if rI != 0. and rJ != 0.:
                accI = (dUdr/r - rK*dUdphi/(r**2)/((rI**2+rJ**2)**0.5))*rI - (dUdlambda/(rI**2+rJ**2))*rJ + grav0[0]
                accJ = (dUdr/r - rK*dUdphi/(r**2)/((rI**2+rJ**2)**0.5))*rJ + (dUdlambda/(rI**2+rJ**2))*rI + grav0[1]
            else:
                accI = dUdr/r + grav0[0]
                accJ = dUdr/r + grav0[1]
            accK = (dUdr/r)*rK + (((rI**2+rJ**2)**0.5)*dUdphi/(r**2)) + grav0[2]

            accVector =np.array( [accI, accJ, accK])
            aVec[i,:] = accVector

        self.accelerations = aVec
        return self.accelerations
