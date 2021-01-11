import numpy as np
from scipy.sparse.linalg import spsolve

def getK(l): 
    result = 1.0 if (l==0) else 2.0
    return result

class Regression:
    def __init__(self, max_deg, a, mu, rVec1D, aVec1D):
        self.N = max_deg  # Degree
        self.a = a # semi-major axis
        self.mu = mu
        self.rVec = rVec1D
        self.aVec = aVec1D

        self.N = self.N 

        self.P = len(rVec1D)
        self.rE = np.zeros((self.N+2,))
        self.iM = np.zeros((self.N+2,))
        self.rho = np.zeros((self.N+3,))

        self.A = np.zeros((self.N+2, self.N+2))

        self.n1 = []
        self.n2 = []
        for i in range(0, self.N+2):
            if i == 0:
                self.A[i,i] = 1.0 
            else:
                self.A[i,i] = np.sqrt((2.0*i+1.0)*getK(i)/(2.0*i*getK(i-1)))*self.A[i-1,i-1]

            n1Row = np.zeros((i+1,)).tolist()
            n2Row = np.zeros((i+1,)).tolist()

            for m in range(0, i+1): # Check the plus one
                if i >= m+2:
                    n1Row[m] = np.sqrt(((2.*i+1.)*(2.*i-1.))/((i-m)*(i+m)))
                    n2Row[m] = np.sqrt(((i+m-1.)*(2.*i+1.)*(i-m-1.))/((i+m)*(i-m)*(2.*i-3.)))
            self.n1.append(n1Row)
            self.n2.append(n2Row)

        self.n1 = np.array(self.n1)
        self.n2 = np.array(self.n2)


    def populate_variables(self, x, y, z):
        
        r = [x,y,z]
        rMag = np.linalg.norm(r)
        s, t, u = r/rMag

        # Eq 23
        for n in range(1, len(self.A)):
            self.A[n,n-1] = np.sqrt(((2.*n)*getK(n-1.))/getK(n)) * self.A[n,n] * u
        

        for m in range(0,len(self.A)):
            for n in range(m+2, len(self.A)):
                self.A[n,m] = u * self.n1[n][m] * self.A[n-1,m] - self.n2[n][m] * self.A[n-2,m]

        #Eq 24
        self.rE[0] = 1 # cos(m*lambda)*cos(m*alpha)
        self.iM[0] = 0 # sin(m*lambda)*cos(m*alpha)
        for m in range(1, len(self.rE)):
            self.rE[m] = s*self.rE[m-1] - t*self.iM[m-1]
            self.iM[m] = s*self.iM[m-1] + t*self.rE[m-1]

        #Eq 26 and 26a
        beta = self.a/rMag
        self.rho[0] = self.mu/rMag
        self.rho[1] = self.rho[0]*beta
        for n in range(2, len(self.rho)):
            self.rho[n] = beta*self.rho[n-1]
        
    def perform_regression(self):
        Q = self.N + 1 # Total Indicies Needed to store all coefficients
        M = np.zeros((self.P,Q*(Q+1) - 2*(2+1)))
   
        for p in range(0, int(self.P/3)):
            rVal = self.rVec[3*p:3*(p+1)]
            rMag = np.linalg.norm(rVal)
            x = self.rVec[3*p]
            y = self.rVec[3*p + 1]
            z = self.rVec[3*p + 2]

            s,t,u = rVal/rMag

            self.populate_variables(x, y, z)
            
            # NOTE: NO ESTIMATION OF C00, C10, C11 -- THESE ARE DETERMINED ALREADY
            for n in range(2,self.N+1):
            
                for m in range(0,n+1):
                
                    delta_m = 1 if (m == 0) else 0
                    delta_m_p1 = 1 if (m+1 == 0) else 0
                    n_lm_n_lm_p1 = np.sqrt((n-m)*(2.0-delta_m)*(n+m+1.)/(2.-delta_m_p1))
                    n_lm_n_l_p1_m_p1 = np.sqrt((n+m+2.0)*(n+m+1.)*(2.*n+1.)*(2.-delta_m)/((2.*n+3.)*(2.-delta_m_p1)))

                    c1 = n_lm_n_lm_p1 # Eq 79 BSK
                    c2 = n_lm_n_l_p1_m_p1 # Eq 80 BSK

                    #TODO: These will need the normalizaiton factor out in front (N1, N2)
                    # Coefficient contribution to X, Y, Z components of the acceleration
                    
                    if (m == 0):
                        rTerm = 0
                        iTerm = 0
                    else:
                        rTerm = self.rE[m-1]
                        iTerm = self.iM[m-1]
                    
                    f_Cnm_1 = (self.rho[n+2]/self.a)*(m*self.A[n,m]*rTerm -s*c2*self.A[n+1,m+1]*self.rE[m])
                    f_Cnm_2 = -(self.rho[n+2]/self.a)*(m*self.A[n,m]*rTerm + t*c2*self.A[n+1,m+1]*self.rE[m])
                    f_Cnm_3 = (self.rho[n+2]/self.a)*(c1*self.A[n,m+1] - u*c2*self.A[n+1,m+1])*self.rE[m]

                    f_Snm_1 = (self.rho[n+2]/self.a)*(m*self.A[n,m]*iTerm -s*c2*self.A[n+1,m+1]*self.iM[m])
                    f_Snm_2 = (self.rho[n+2]/self.a)*(m*self.A[n,m]*iTerm - t*c2*self.A[n+1,m+1]*self.iM[m])
                    f_Snm_3 = (self.rho[n+2]/self.a)*(c1*self.A[n,m+1] - u*c2*self.A[n+1,m+1])*self.iM[m]
                    
                    idx = n - 2 # The M matrix excludes columns for C00, C10, C11 so we need to subtract 2 from the current degree for proper indexing
                    #idx = n
                    degIdx = n*(n+1) - (2*(2+1))
                    M[3*p + 0, degIdx + 2*m + 0] = f_Cnm_1 # X direction
                    M[3*p + 0, degIdx + 2*m + 1] = f_Snm_1
                    M[3*p + 1, degIdx + 2*m + 0] = f_Cnm_2 # Y direction
                    M[3*p + 1, degIdx + 2*m + 1] = f_Snm_2
                    M[3*p + 2, degIdx + 2*m + 0] = f_Cnm_3 # Z direction
                    M[3*p + 2, degIdx + 2*m + 1] = f_Snm_3

        results = np.linalg.lstsq(M, self.aVec)
        #results = spsolve(M,self.aVec)
        print(results[0].reshape((-1,2)))


def main():
    from GravNN.Trajectories.DHGridDist import DHGridDist
    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data

    planet = Earth()
    trajectory = DHGridDist(planet, planet.radius, 45)
    gravity_model = SphericalHarmonics(planet.sh_hf_file, degree=180, trajectory=trajectory)
    acceleration = gravity_model.load()
    positions = trajectory.positions

    regressor = Regression(10, planet.radius*2, planet.mu, positions.reshape((-1,)), acceleration.reshape((-1,)))
    regressor.perform_regression()


if __name__  == "__main__":
    main()
