import numpy as np
from numba import jit, int32, float64
from numba.experimental import jitclass

spec = [
    ('N', int32),               
    ('mu', float64),          
    ('a', float64),          
    ('cbar', float64[:,::1]),          
    ('sbar', float64[:,::1]),          
    ('n1', float64[:,::1]),          
    ('n2', float64[:,::1]),          
    ('n1q', float64[:,::1]),          
    ('n2q', float64[:,::1]),          
]

@jitclass(spec)
class PinesAlgorithm():
    def __init__(self, r0, mu, degree, cbar, sbar):
        self.N = degree
        self.mu = mu
        self.a = r0
        self.cbar = cbar
        self.sbar = sbar
        
        self.n1 = np.ones((self.N+2, self.N+2))*np.nan
        self.n2 = np.ones((self.N+2, self.N+2))*np.nan
        self.n1q = np.ones((self.N+2, self.N+2))*np.nan
        self.n2q = np.ones((self.N+2, self.N+2))*np.nan
        
        for l in range(0, self.N+2):
            for m in range(0, l+1):
                if (l >= m + 2):
                    self.n1[l][m] = np.sqrt(((2.0*l+1.0)*(2.0*l-1.0))/((l-m)*(l+m)))
                    self.n2[l][m] = np.sqrt(((l+m-1.0)*(2.0*l+1.0)*(l-m-1.0))/((l+m)*(l-m)*(2.0*l-3.0)))
                if (l < self.N+1):
                    if (m < l): # this may need to also ensure that l < N+1 
                        self.n1q[l][m] = np.sqrt(((l-m)*self.getK(m)*(l+m+1.0))/self.getK(m+1))
                    self.n2q[l][m] = np.sqrt(((l+m+2.0)*(l+m+1.0)*(2.0*l+1.0)*self.getK(m))/((2.0*l+3.0)*self.getK(m+1.0)))

    def getK(self, x):
        return 1.0 if (x == 0) else 2.0
    

    def compute_acc(self, positions):
        acc = np.zeros(positions.shape)
        for i in range(0, int(len(positions)/3)):
            acc[3*i:3*(i+1)] = self.compute_acc_thread(positions[3*i:3*(i+1)])
        return acc
    
    def compute_acc_thread(self, positions):
        r = np.linalg.norm(positions[0:3])
        [s, t, u] = positions[0:3]/r

        rE = np.zeros((self.N+2,))
        iM = np.zeros((self.N+2,))

        rhol = np.zeros((self.N+2,))

        aBar = np.zeros((self.N+2, self.N+2))
        aBar[0,0] = 1.0

        rho = self.a/r
        rhol[0] = self.mu/r
        rhol[1] = rhol[0]*rho

        for l in range(1, self.N+2):
            aBar[l][l] = np.sqrt(((2.0*l+1.0)*self.getK(l))/((2.0*l*self.getK(l-1))))*aBar[l-1][l-1]
            aBar[l][l-1] = np.sqrt((2.0*l)*self.getK(l-1)/self.getK(l))*aBar[l][l]*u
        
        for m in range(0, self.N+2):
            for l in range(m+2, self.N+2):
                aBar[l][m] = u*self.n1[l][m]*aBar[l-1][m] - self.n2[l][m]*aBar[l-2][m]
            rE[m] = 1.0 if m == 0 else s*rE[m-1] - t*iM[m-1]
            iM[m] = 0.0 if m == 0 else s*iM[m-1] + t*rE[m-1]
        
        a1, a2, a3, a4 = 0.0, 0.0, 0.0, 0.0
        for l in range(1, self.N+1):
            rhol[l+1] = rho*rhol[l]
            sum_a1, sum_a2, sum_a3, sum_a4 = 0.0, 0.0, 0.0, 0.0
            for m in range(0, l+1):
                D = self.cbar[l][m]*rE[m] + self.sbar[l][m]*iM[m]
                E = 0.0 if m == 0 else self.cbar[l][m]*rE[m-1] + self.sbar[l][m]*iM[m-1]
                F = 0.0 if m == 0 else self.sbar[l][m]*rE[m-1] - self.cbar[l][m]*iM[m-1]

                sum_a1 += m*aBar[l][m]*E
                sum_a2 += m*aBar[l][m]*F

                if m < l:
                    sum_a3 += self.n1q[l][m]*aBar[l][m+1]*D
                sum_a4 += self.n2q[l][m]*aBar[l+1][m+1]*D
            a1 += rhol[l+1]/self.a*sum_a1
            a2 += rhol[l+1]/self.a*sum_a2
            a3 += rhol[l+1]/self.a*sum_a3
            a4 -= rhol[l+1]/self.a*sum_a4
        a4 -= rhol[1]/self.a

        return np.array([a1, a2, a3]) + np.array([s,t,u])*a4


   