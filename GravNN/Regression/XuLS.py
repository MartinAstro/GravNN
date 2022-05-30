import numpy as np
import sigfig
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize, Bounds
from numba import njit, prange
from GravNN.Regression.utils import compute_A, compute_euler, format_coefficients, getK
import matplotlib.pyplot as plt
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







class XuLS:
    def __init__(self, max_deg, planet, remove_deg=-1, algorithm='kaula'):
        self.algorithm = algorithm
        self.N = max_deg  # Degree
        self.a = planet.radius
        self.mu = planet.mu
        self.remove_deg = remove_deg
        self.compute_kaula_matrix()

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

    def compute_kaula_matrix(self):
        l = 1
        m = 0
        factor = 10E-5
        K = np.diag(np.zeros((int((self.N+1)*(self.N+2)))))
        K_inv = np.diag(np.zeros((int((self.N+1)*(self.N+2)))))
        for i in range(2,len(K)): # all coefficients (C and S) excluding C_00, S_00
            K[i,i] = (factor/l**2)**1
            K_inv[i,i] = (factor/l**2)**-1
            # K[i,i] = (l**2/10**-5) # this matrix should be the inverse of the degree variations
            if (i + 1) % 2 == 0: # every odd number, increment the m index (because we've iterated over a C and S pair)
                if l == m: # 
                    l +=1
                    m = 0
                else:
                    m += 1
        self.K = K
        self.K_inv = K_inv
    
    def compute_coefficients(self,M, aVec, iterations):
        if self.algorithm == 'kaula':
            coef = self.kaula_solution(M, aVec, iterations)
        if self.algorithm == 'kaula_inv':
            coef = self.kaula_inv_solution(M, aVec, iterations)
        if self.algorithm == 'single_parameter':
            coef = self.single_parameter_regression(M, aVec, iterations)
        if self.algorithm == 'least_squares':
            coef = self.least_squares_solution(M, aVec)
        return coef 

    def kaula_solution(self,M, aVec, iterations):
        q = len(M[0])
        K_used = self.K[-q:, -q:]        
        coef = np.linalg.inv(M.T@M + K_used)@M.T@aVec
        return coef

    def kaula_inv_solution(self,M, aVec, iterations):
        q = len(M[0])
        K_used = self.K_inv[-q:, -q:]        
        coef = np.linalg.inv(M.T@M + K_used)@M.T@aVec
        return coef

    def least_squares_solution(self,M, aVec):
        coef = np.linalg.pinv(M.T@M)@M.T@aVec
        return coef

    def single_parameter_regression(self, M, aVec, iterations=5):
        # with pseudoinverse 
        y = aVec #+ np.random.normal(0, 1E-8, size=np.shape(aVec))
        sigma = 1
        k = 1E-5
        A = M
        P = np.eye(len(A))
        I0 = np.eye(len(A[0]))
        l_max = 4
        I0[:l_max*(l_max+1), :l_max*(l_max+1)] = 0.0

        # Remove the zero rows from the invertible matrix
        APA = A.T@P@A
        empty_rows = np.array([np.all(row == 0.0) for row in APA])
        for i in range(len(empty_rows)):
            APA[i,i] = 1.0 if empty_rows[i] else APA[i,i]

        # compute solution, and iterate until optimal k 
        for q in range(iterations):
            beta = np.linalg.inv(APA + k*I0)@A.T@P@y
            def objective(k):
                N_s = APA + k*I0
                N_s_inv = np.linalg.inv(N_s)
                D_beta = sigma**2*N_s_inv@APA@N_s_inv
                f_s = np.trace(D_beta) + \
                    k**2*beta.T@I0@N_s_inv@N_s_inv@I0@beta
                return f_s


            bounds = Bounds([0.0],[1.0])
            res = minimize(objective, [k], tol=1E-10, bounds=bounds)
            print(f"Total iterations: {res.nit}")
            k = res.x[0]
        return np.linalg.inv(APA + k*I0)@A.T@P@y

    def update(self, rVec, aVec, iterations=5):
        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))
        self.P = len(self.rVec1D)

        M = populate_M(self.rVec1D, self.A, self.n1, self.n2, self.N, self.a, self.mu, self.remove_deg)
        results = self.compute_coefficients(M, self.aVec1D, iterations)
        return results


class AnalyzeRegression:
    def __init__(self, true_Clm, true_Slm, pred_Clm, pred_Slm):
        self.true_Clm = true_Clm
        self.true_Slm = true_Slm
        self.pred_Clm = pred_Clm
        self.pred_Slm = pred_Slm

    def compute_degree_variance(self,C_lm, S_lm):
        N = len(C_lm)
        rms = np.zeros((N,))
        for i in range(N):
            rms_sum = 0.0
            for j in range(i+1):
                rms_sum += (C_lm[i,j]**2 + S_lm[i,j]**2)
            rms_sum *= 1/(2*i + 1)
            rms[i] = np.sqrt(rms_sum)
        return rms

    def plot_coef_rms(self, C_lm, S_lm):
        N = len(C_lm)
        degrees = np.arange(0, N)
        rms = self.compute_degree_variance(C_lm, S_lm)
        plt.semilogy(degrees, rms)
        plt.xlim([2,None])


def print_coefficients(C_lm, S_lm):
    for i in range(len(C_lm)):
        for j in range(i+1):
            print(f"({i},{j}): {sigfig.round(float(C_lm[i,j]),sigfigs=2, notation='scientific')} \t {sigfig.round(float(S_lm[i,j]),sigfigs=2,notation='scientific')}")


def simple_experiment():
    from GravNN.Trajectories import DHGridDist, RandomDist
    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
    import time
    max_true_deg = 30
    regress_deg = 16
    remove_deg = 0
    # solver_algorithm='kaula'
    solver_algorithm='single_parameter'
    # solver_algorithm='least_squares'

    planet = Earth()
    sh_EGM2008 = SphericalHarmonics(planet.sh_file, regress_deg)

    # trajectory = DHGridDist(planet, sh_EGM2008.radEquator, regress_deg+1)
    trajectory = RandomDist(planet, [planet.radius, planet.radius+420], 1000)
    x, a, u = get_sh_data(trajectory,planet.sh_file, max_deg=max_true_deg, deg_removed=remove_deg)


    regressor = XuLS(regress_deg, planet, remove_deg, solver_algorithm)
    start = time.time()
    results = regressor.update(x, a)
    C_lm, S_lm = format_coefficients(results, regress_deg, remove_deg)
    print(time.time() - start)

    k = len(C_lm)
    true_C_lm = sh_EGM2008.C_lm[:k,:k]
    true_S_lm = sh_EGM2008.S_lm[:k,:k]

    C_lm_error = (true_C_lm - C_lm)/ true_C_lm*100
    S_lm_error = (true_S_lm - S_lm)/ true_S_lm*100

    C_lm_error = np.nan_to_num(C_lm_error, posinf=0, neginf=0)
    S_lm_error = np.nan_to_num(S_lm_error, posinf=0, neginf=0)

    print_coefficients(C_lm_error, S_lm_error)

    analyzer = AnalyzeRegression(true_C_lm, true_S_lm, C_lm, S_lm)
    analyzer.plot_coef_rms(true_C_lm, true_S_lm)
    analyzer.plot_coef_rms(C_lm, S_lm)
    analyzer.plot_coef_rms(C_lm-true_C_lm, S_lm-true_S_lm)
    plt.show()



if __name__  == "__main__":
    simple_experiment()