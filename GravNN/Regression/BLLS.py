import numpy as np
from scipy.sparse.linalg import spsolve
from numba import njit, prange
from GravNN.Regression.utils import compute_A, compute_euler, format_coefficients, getK

@njit(cache=True)#, parallel=True)
def populate_M(rVec1D, A, n1, n2, N, a, mu, remove_deg):
    P = len(rVec1D)
    k = remove_deg
    M = np.zeros((P,(N+2)*(N+1) - (k+2)*(k+1)))

    for p in range(0, int(P/3)):
                
        rVal = rVec1D[3*p:3*(p+1)]
        rMag = np.linalg.norm(rVal)
        s, t, u = rVal/rMag

        # populate variables
        A = compute_A(A, n1, n2, u)
        rE, iM, rho = compute_euler(N, a, mu, rMag, s, t)
        
        for n in range(k+1,N+1):
            for m in range(0,n+1):
                delta_m = 1 if (m == 0) else 0
                delta_m_p1 = 1 if (m+1 == 0) else 0
                n_lm_n_lm_p1 = np.sqrt((n-m)*(2.0-delta_m)*(n+m+1.)/(2.-delta_m_p1))
                n_lm_n_l_p1_m_p1 = np.sqrt((n+m+2.0)*(n+m+1.)*(2.*n+1.)*(2.-delta_m)/((2.*n+3.)*(2.-delta_m_p1)))

                c1 = n_lm_n_lm_p1 # Eq 79 BSK
                c2 = n_lm_n_l_p1_m_p1 # Eq 80 BSK

                #TODO: These will need the normalization factor out in front (N1, N2)
                # Coefficient contribution to X, Y, Z components of the acceleration
                
                if (m == 0):
                    rTerm = 0
                    iTerm = 0
                else:
                    rTerm = rE[m-1]
                    iTerm = iM[m-1]
                
                # Pines Derivatives -- but rho n+1 rather than n+2
                f_Cnm_1 = (rho[n+1]/a)*(m*A[n,m]*rTerm - s*c2*A[n+1,m+1]*rE[m])
                f_Cnm_2 = (rho[n+1]/a)*(-m*A[n,m]*iTerm - t*c2*A[n+1,m+1]*rE[m])

                if m < n:
                    f_Cnm_3 = (rho[n+1]/a)*(c1*A[n,m+1] - u*c2*A[n+1,m+1])*rE[m]
                else:
                    f_Cnm_3 = (rho[n+1]/a)*(-1.0*u*c2*A[n+1,m+1])*rE[m]

                f_Snm_1 = (rho[n+1]/a)*(m*A[n,m]*iTerm -s*c2*A[n+1,m+1]*iM[m])
                f_Snm_2 = (rho[n+1]/a)*(m*A[n,m]*rTerm - t*c2*A[n+1,m+1]*iM[m])
                if m < n:
                    f_Snm_3 = (rho[n+1]/a)*(c1*A[n,m+1] - u*c2*A[n+1,m+1])*iM[m]
                else:
                    f_Snm_3 = (rho[n+1]/a)*(-1.0*u*c2*A[n+1,m+1])*iM[m]


                degIdx = (n+1)*(n) - (k+2)*(k+1)

                M[3*p + 0, degIdx + 2*m + 0] = f_Cnm_1 # X direction
                M[3*p + 0, degIdx + 2*m + 1] = f_Snm_1
                M[3*p + 1, degIdx + 2*m + 0] = f_Cnm_2 # Y direction
                M[3*p + 1, degIdx + 2*m + 1] = f_Snm_2
                M[3*p + 2, degIdx + 2*m + 0] = f_Cnm_3 # Z direction
                M[3*p + 2, degIdx + 2*m + 1] = f_Snm_3
    
    return M





def iterate_lstsq(M, aVec, iterations):
    results = np.linalg.lstsq(M, aVec)[0]
    delta_a = aVec - np.dot(M, results)
    for i in range(iterations):
        delta_coef = np.linalg.lstsq(M,delta_a)[0]
        results -= delta_coef
        delta_a = aVec - np.dot(M, results)
    return results



class BLLS:
    def __init__(self, max_deg, planet, remove_deg=-1):
        self.N = max_deg  # Degree
        self.a = planet.radius
        self.mu = planet.mu
        self.remove_deg = remove_deg

        self.rE = np.zeros((self.N+2,))
        self.iM = np.zeros((self.N+2,))
        self.rho = np.zeros((self.N+3,))

        self.A = np.zeros((self.N+2, self.N+2))

        self.n1 = np.zeros((self.N+2, self.N+2))
        self.n2 = np.zeros((self.N+2, self.N+2))

        for i in range(0, self.N+2):
            if i == 0:
                self.A[i,i] = 1.0 
            else:
                self.A[i,i] = np.sqrt((2.0*i+1.0)*getK(i)/(2.0*i*getK(i-1)))*self.A[i-1,i-1]

            for m in range(0, i+1): # Check the plus one
                if i >= m+2:
                    self.n1[i,m] = np.sqrt(((2.*i+1.)*(2.*i-1.))/((i-m)*(i+m)))
                    self.n2[i,m] = np.sqrt(((i+m-1.)*(2.*i+1.)*(i-m-1.))/((i+m)*(i-m)*(2.*i-3.)))
        
    def update(self, rVec, aVec, iterations=5):
        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))
        self.P = len(self.rVec1D)

        M = populate_M(self.rVec1D, self.A, self.n1, self.n2, self.N, self.a, self.mu, self.remove_deg)
        results = iterate_lstsq(M, self.aVec1D, iterations)
        return results

   
def main():
    from GravNN.Trajectories import DHGridDist, RandomDist
    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
    import time
    max_true_deg = 4
    regress_deg = 4
    remove_deg = 1
    #remove_deg = 0 # C20 is very close, C22 isn't that close

    planet = Earth()
    sh_EGM2008 = SphericalHarmonics(planet.sh_hf_file, regress_deg)

    trajectory = DHGridDist(planet, sh_EGM2008.radEquator, 5)
    #trajectory = RandomDist(planet, [planet.radius, planet.radius+420], 1000)

    if remove_deg != -1:
        x, a, u = get_sh_data(trajectory,planet.sh_hf_file, max_deg=max_true_deg, deg_removed=remove_deg)
    else:
        x, a, u = get_sh_data(trajectory,planet.sh_hf_file, max_deg=max_true_deg, deg_removed=-1)


    regressor = BLLS(regress_deg, planet, remove_deg)
    start = time.time()
    results = regressor.update(x, a)
    C_lm, S_lm = format_coefficients(results, regress_deg, remove_deg)
    print(time.time() - start)
    # print(C_lm)
    # print(S_lm)

    k = len(C_lm)
    print(np.array2string((sh_EGM2008.C_lm[:k,:k] - C_lm)/ sh_EGM2008.C_lm[:k,:k]*100, precision=0))
    print(np.array2string((sh_EGM2008.S_lm[:k,:k] - S_lm)/ sh_EGM2008.S_lm[:k,:k]*100, precision=0))

    #regressor.save('C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\Regressed\\some.csv')
    #print(coefficients)

if __name__  == "__main__":
    main()
