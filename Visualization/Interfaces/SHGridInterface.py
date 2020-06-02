import pyshtools
import numpy as np
import os
import sys
path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path + '/../..')
import math
from AccelerationAlgs.SphericalHarmonics import SphericalHarmonics
from support.transformations import sphere2cart, cart2sph
from Trajectories.UniformDist import UniformDist
from CelestialBodies.Planets import Earth

class SHGridInterface():
    def __init__(self):
        pass

    def get_positions(self,sh_grid, radius):
        lons =sh_grid.total.lons()
        lats = sh_grid.total.lats()

        rSphere = np.zeros((len(lons)*len(lats),3))
        for i in range(len(lons)):
            for j in range(len(lats)):
                rSphere[i*len(lats) + j, 0] = radius
                rSphere[i*len(lats) + j, 1] = lons[i]
                rSphere[i*len(lats) + j, 2] = lats[j]

        rCart = sphere2cart(rSphere)
        return rCart
    
    def assign_accelerations(self, sh_grid, accelerations):
        sh_grid.rad.data = accelerations[0]
        sh_grid.theta.data = accelerations[1] #TODO: Check theta and phi
        sh_grid.phi.data = accelerations[2]
        return #TODO: check that the input parameter is modified directly (I think it is)
        
    # def recompute_acceleration(self, radius, degree):
    #     grid= sh.expand() 
    #     lons = grid.total.lons() 
    #     lats = grid.total.lats() 

    #     data = np.zeros(grid.total.data.shape)
    #     coordinates = np.zeros((len(lons)*len(lats),3))

    #     for i in range(len(lons)):
    #         for j in range(len(lats)):
    #             coordinates[i*len(lats) + j, 0] = radius
    #             coordinates[i*len(lats) + j, 1] = lons[i]
    #             coordinates[i*len(lats) + j, 2] = lats[j]

    #     self._compute_acc(coordinates, degree)
    #     return
    # def _compute_acc(self, rVec, degree):

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

        cList = self.sh.coeffs[0]
        sList = self.sh.coeffs[1]
        mu = self.sh.gm
        rad = self.sh.r0
        max_degree = degree
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
                cList[l][m]  = cList[l][m] / PI
                sList[l][m] = sList[l][m] / PI

        rVec = sphere2cart(rVec)
        for i in range(len(rVec)):
            [rI, rJ, rK ]= rVec[i,:] # takes in spherical coordinates in degrees and returns x,y,z

            r = np.linalg.norm(rVec[i,:])
            rHat = rVec[i,:] / r
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

            for l in range(2, degree+1): #can only do for max degree minus 1
                for m in range(0,l+1):
                    r_l = (rad/r)**l
                    c_m_lam = np.cos(m*lambdaSat)
                    s_m_lam = np.sin(m*lambdaSat)
                    c_lm = cList[l][m]
                    s_lm = sList[l][m]
                    
                    dUdr = dUdr + (r_l*(l+1)*P[l,m]) * (c_lm*c_m_lam+s_lm*s_m_lam)
                    dUdphi = dUdphi + (r_l*P[l,m+1] - m*np.tan(phi)*P[l,m]) * (c_lm*c_m_lam + s_lm*s_m_lam)
                    dUdlambda = dUdlambda + (r_l*m*P[l,m]) * (s_lm*c_m_lam - c_lm*s_m_lam)

            dUdr = -mu * dUdr / r**2
            dUdphi = mu * dUdphi / r
            dUdlambda = mu * dUdlambda / r

            denom = rI**2+rJ**2
            if rI != 0. and rJ != 0.:
                accI = (dUdr/r - rK*dUdphi/(r**2)/(denom**0.5))*rI - (dUdlambda/denom)*rJ + grav0[0]
                accJ = (dUdr/r - rK*dUdphi/(r**2)/(denom**0.5))*rJ + (dUdlambda/denom)*rI + grav0[1]
            else:
                accI = dUdr/r + grav0[0]
                accJ = dUdr/r + grav0[1]
            accK = (dUdr/r)*rK + ((denom**0.5)*dUdphi/(r**2)) + grav0[2]

            accVector =np.array( [accI, accJ, accK])
            aVec[i,:] = accVector

        aVec = cart2sph(aVec)
        self.accelerations = aVec