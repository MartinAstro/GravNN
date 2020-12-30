import numpy as np
import scipy
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics

class Coefficients:
    def __init__(self, degree):
        self.C_lm = np.zeros((degree+2, degree+2))
        self.S_lm = np.zeros((degree+2, degree+2))

def getK(i):
    return 1.0 if i == 0 else 2.0

def single_calculation_variables(N):
    # Load single calculation objects
    n1 = []
    n2 = []
    for l in range(0, N+2):
        n1Row = np.zeros((l+1,))
        n2Row = np.zeros((l+1,))

        for m in range(0, l+1): # m<=l
            if (l >= m + 2):
                n1Row[m] = np.sqrt((2.0*l+1.0)*(2.0*l-1))/((l-m)*(l+m))
                n2Row[m] = np.sqrt((l+m-1.0)*(2.0*l+1.0)*(l-m-1.0))/((l+m)*(l-m)*(2.0*l-3.0))

        n1.append(n1Row)
        n2.append(n2Row)
    return n1, n2

def set_regression_variables(position, n1, n2, N, a, mu):
        A = np.zeros((N+2, N+2))
        r = np.zeros((N+2,))
        i = np.zeros((N+2,))
        rho = np.zeros((N+2+1)) # Need to go to + 3 for the original rho[n+2] formulation. 

        x, y, z = position
        s, t, u = position / np.linalg.norm(position)

        for l in range(0, N+2):
            A[l][l] = 1.0 if (l == 0) else np.sqrt((2*l+1)*getK(l))/(2*l*getK(l-1)) * A[l-1][l-1] # Diagonal elements of A_bar

        #Eq 23
        for l in range(1, len(A)):
            A[l][l-1] = np.sqrt((2.0*l)*getK(l-1)/getK(l)) * A[l][l] * u

        for m in range(0, len(A)):
            for l in range(m + 2, len(A)):
                A[l][m] = u *n1[l][m] * A[l-1][m] - n2[l][m] * A[l-2][m]

        #Eq 24
        r[0] = 1 # cos(m*lambda)*cos(m*alpha)
        i[0] = 0 #sin(m*lambda)*cos(m*alpha)
        for m in range(1,len(r)):
            r[m] = s*r[m-1] - t*i[m-1]
            i[m] = s*i[m-1] + t*r[m-1]

        #Eq 26 and 26a
        rMag = np.linalg.norm(position)
        beta = a/rMag
        rho[0] = mu/rMag
        rho[1] = rho[0]*beta
        for n in range(2, len(rho)):
            rho[n] = beta*rho[n-1]
        
        return A, r, i, rho



def format_coefficients(coef_list, degree):
    l_i = 0
    l = l_i
    m = 0
    coef_regress = Coefficients(degree)
    # coef_regress.C_lm[0][0] = 1.0 if (l_i != 0) else 0.0 # If regressing only C20 and force the earlier coefficients

    for i in range(0, int(len(coef_list)/2)):
        if (m > l):
            l += 1
            m = 0
        coef_regress.C_lm[l][m] = coef_list[2*i]
        coef_regress.S_lm[l][m] = coef_list[2*i + 1]
        m += 1
    return coef_regress

def regress(positions, accelerations, N, planet):
    n1, n2 = single_calculation_variables(N)
    a = planet.radius
    mu = planet.mu

    l_i = 0 # Degree of first coefficient to regress (typically either 0 or 2)
    M_skip = l_i*(l_i+1) # Number of coefficients to skip
    M_total = (N + 1)*(N + 2) # Total number of coefficients up to degree l_{f+1} -- e.g if l_max = 2 then M_total = 12
    P = np.shape(positions)[0] * np.shape(positions)[1]
    M = np.zeros((P, M_total-M_skip))

    for p in range(0, len(positions)):
        s, t, u = positions[p] / np.linalg.norm(positions[p])
        A, r, i, rho = set_regression_variables(positions[p], n1, n2, N, a, mu)
        for l in range(l_i, N+1): #l <= N
            for m in range(0, l+1): # m <= l
                delta_m = 1 if (m == 0) else 0
                delta_m_p1 = 1 if (m+1 == 0) else 0
                c1 = np.sqrt((l-m)*(2.0-delta_m)*(l+m+1.0)/(2.0-delta_m_p1)) # n_lm_n_lm_p1 == Eq 79
                c2 = np.sqrt((l+m+2.0)*(l+m+1)*(2.0*l+1.0)*(2.0-delta_m)/((2.0*l+3.0)*(2.0-delta_m_p1))) # n_lm_n_l_p1_m_p1 == Eq 80 BSK

                rTerm = 0 if (m == 0) else r[m-1]
                iTerm = 0 if (m == 0) else i[m-1]

                # Coefficient contribution to X, Y, Z components of the acceleration
                # ORIGINAL CALL FROM PAPER
                f_Cnm_1 = (rho[l+2]/a)*(m*A[l][m]*rTerm - s*c2*A[l+1][m+1]*r[m])
                f_Cnm_2 = -(rho[l+2]/a)*(m*A[l][m]*rTerm + t*c2*A[l+1][m+1]*r[m])
                f_Cnm_3 = (rho[l+2]/a)*(c1*A[l][m+1] - u*c2*A[l+1][m+1])*r[m]

                f_Snm_1 = (rho[l+2]/a)*(m*A[l][m]*iTerm - s*c2*A[l+1][m+1]*i[m])
                f_Snm_2 = (rho[l+2]/a)*(m*A[l][m]*iTerm - t*c2*A[l+1][m+1]*i[m])
                f_Snm_3 = (rho[l+2]/a)*(c1*A[l][m+1] - u*c2*A[l+1][m+1])*i[m]

                # RHO N+1 RATHER THAN N+2 BC CAN'T FIGURE OUT MATH
                # f_Cnm_1 = (rho[l+1]/a)*(m*A[l][m]*rTerm - s*c2*A[l+1][m+1]*r[m])
                # f_Cnm_2 = -(rho[l+1]/a)*(m*A[l][m]*rTerm + t*c2*A[l+1][m+1]*r[m])
                # f_Cnm_3 = (rho[l+1]/a)*(c1*A[l][m+1] - u*c2*A[l+1][m+1])*r[m]

                # f_Snm_1 = (rho[l+1]/a)*(m*A[l][m]*iTerm - s*c2*A[l+1][m+1]*i[m])
                # f_Snm_2 = (rho[l+1]/a)*(m*A[l][m]*iTerm - t*c2*A[l+1][m+1]*i[m])
                # f_Snm_3 = (rho[l+1]/a)*(c1*A[l][m+1] - u*c2*A[l+1][m+1])*i[m]

                degIdx = l*(l+1) - M_skip
                M[3*p + 0, degIdx + 2*m + 0] = f_Cnm_1 # X direction
                M[3*p + 0, degIdx + 2*m + 1] = f_Snm_1
                M[3*p + 1, degIdx + 2*m + 0] = f_Cnm_2 # Y direction
                M[3*p + 1, degIdx + 2*m + 1] = f_Snm_2
                M[3*p + 2, degIdx + 2*m + 0] = f_Cnm_3 # Z direction
                M[3*p + 2, degIdx + 2*m + 1] = f_Snm_3



    # Initialize variables
    iterations = 0
    max_iterations = 1000
    precision = 1E-4

    Y = accelerations.reshape((-1,)) # true acceleration
    Y_hat = np.zeros(np.shape(Y))# regressed acceleration
    X_i = np.zeros((M_total - M_skip)) # original coefficient guess
    # X_f, dX (new guess, improvement) 

    # Initialize variables and sparse matrix
    solution_unique = len(Y) > (M_total - M_skip) # Check if the problem is underdetermined
    solution_exists = False

    while not solution_exists and iterations < max_iterations:
        dY =  Y - Y_hat

        #dX = np.linalg.solve(M, dY)
        dX = np.linalg.lstsq(M, dY, rcond=1E-12)[0]
        #dX = scipy.linalg.lstst(M, dY)
        
        X = np.linalg.lstsq(M,Y)[0]
        # Wiki iterative refinement
        while True:
            r = Y - np.dot(M,X)
            dX = np.linalg.lstsq(M, r)[0]
            X = X + dX
            
        rel_error = np.linalg.norm(np.dot(M,dX) - dY) / np.linalg.norm(dY)
        solution_exists = rel_error < precision

        X_f = X_i + dX

        sh_model = SphericalHarmonics(planet.sh_file, N)
        sh_model.load_regression(format_coefficients(X_f, N), planet)

        X_i = X_f
        Y_hat = sh_model.compute_acc(positions).reshape((-1,))
        
        iterations += 1

    print("Ran " + str(iterations) + " iterations!")    
    return coef_regress




