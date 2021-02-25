import csv
import os, sys
import numpy as np
import json
#from GravNN.build.PinesAlgorithm import PinesAlgorithm
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
from GravNN.GravityModels.PinesAlgorithm import compute_acc_parallel, compute_n_matrices

def make_2D_array(lis):
    """Funciton to get 2D array from a list of lists"""
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len))

    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr


def get_sh_data(trajectory, gravity_file, **kwargs):

    try:
        max_deg = int(kwargs['max_deg'][0])
        deg_removed = int(kwargs['deg_removed'][0])
    except:
        max_deg = int(kwargs['max_deg'])
        deg_removed = int(kwargs['deg_removed'])

    Call_r0_gm = SphericalHarmonics(gravity_file, degree=max_deg, trajectory=trajectory)
    accelerations = Call_r0_gm.load()

    Clm_r0_gm = SphericalHarmonics(gravity_file, degree=deg_removed, trajectory=trajectory)
    accelerations_Clm = Clm_r0_gm.load()

    x = Call_r0_gm.positions # position (N x 3)
    a = accelerations - accelerations_Clm
    u = np.array([None for _ in range(len(a))]) # potential (N x 1)

    return x, a, u

class SphericalHarmonics(GravityModelBase):
    def __init__(self, sh_info, degree, trajectory=None):
        super().__init__()

        self.degree = degree

        self.mu = None
        self.radEquator = None
        self.C_lm = None
        self.S_lm = None

        if type(sh_info) == str:
            self.file = sh_info
        else:
            self.file = "./"
        
        self.configure(trajectory)

        if type(sh_info) == str:
            self.loadSH()
        else:
            self.load_regression(sh_info)
                
        pass

    def generate_full_file_directory(self):
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "_" + os.path.basename(self.file).split('.')[0] + "_" + str(self.degree) + "/"
        pass

    def loadSH_json(self):
        data = json.load(open(self.file,'r'))
        clm = np.zeros((40,40)).tolist()
        slm = np.zeros((40,40)).tolist()
        for i in range(len(data['Cnm_coefs'])):
            n = data['Cnm_coefs'][i]['n']
            m = data['Cnm_coefs'][i]['m']

            clm[n][m] = data['Cnm_coefs'][i]['value']

            n = data['Snm_coefs'][i]['n']
            m = data['Snm_coefs'][i]['m']

            slm[n][m] = data['Snm_coefs'][i]['value']

        for i in range(len(clm)):
            mask = np.ones(len(clm[i]), dtype=bool)
            mask[i+1:] = False
            clm_row = np.array(clm[i])[mask]
            slm_row = np.array(slm[i])[mask]

            clm[i] = clm_row.tolist()
            slm[i] = slm_row.tolist()

        self.mu = data['totalMass']['value']*6.67408*1E-11 
        self.radEquator = data['referenceRadius']['value']
        self.C_lm = make_2D_array(clm)
        self.S_lm = make_2D_array(slm)
        return
        
    def loadSH_csv(self):
        with open(self.file, 'r') as csvfile:
            gravReader = csv.reader(csvfile, delimiter=',')
            firstRow = next(gravReader)
            clmList = []
            slmList = []
            # Currently do not take the mu and radius values provided by the gravity file.
            try:
                valCurr = int(firstRow[0])
            except ValueError:
                self.mu = float(firstRow[1])
                self.radEquator = float(firstRow[0])

            clmRow = []
            slmRow = []
            currDeg = 0
            for gravRow in gravReader:
                #TODO: Check if this results in correct behavior
                if self.degree is not None:
                    if(int(float(gravRow[0])) > self.degree + 2):# if loading coefficients beyond the maximum desired (+2 for purposes of the algorithm)
                        break
                while int(float(gravRow[0])) > currDeg:
                    if (len(clmRow) < currDeg + 1):
                        clmRow.extend([0.0] * (currDeg + 1 - len(clmRow)))
                        slmRow.extend([0.0] * (currDeg + 1 - len(slmRow)))
                    clmList.append(clmRow)
                    slmList.append(slmRow)
                    clmRow = []
                    slmRow = []
                    currDeg += 1
                clmRow.append(float(gravRow[2]))
                slmRow.append(float(gravRow[3]))
            
            clmList.append(clmRow)
            slmList.append(slmRow)
            if self.degree is None:
                self.degree = currDeg -2
            
            self.C_lm = make_2D_array(clmList)
            self.S_lm = make_2D_array(slmList)
            return 

    def loadSH(self):
        if '.json' in self.file:
            self.loadSH_json()
        else:
            self.loadSH_csv()
        

    def load_regression(self, reg_solution, planet):
        self.file_directory += "_Regress"
        self.mu = planet.mu
        self.radEquator = planet.radius
        self.C_lm = reg_solution.C_lm
        self.S_lm = reg_solution.S_lm

        # self.C_lm = []
        # self.S_lm = []
        # C_lm = []
        # S_lm = []

        # # Check if C00 is included as it should be equal to 1.0
        # C00_11_included = False
        # l = 2
        # included_coef = 3
        # if abs(reg_solution.C_lm[0] - 1.0) < 10E-10:
        #     C00_11_included = True
        #     l = 0
        #     included_coef = 0
        # else:
        #     self.C_lm.append([1.0])
        #     self.C_lm.append([0.0, 0.0])
        #     self.S_lm.append([0.0])
        #     self.S_lm.append([0.0, 0.0])


        # # Separate C_lm and S_lm
        # for i in range(int(len(reg_solution)/2)):
        #     C_lm.append(reg_solution[2*i])
        #     S_lm.append(reg_solution[2*i + 1])
        
        # # Format coefficients into row by degree
        # C_row = []
        # S_row  = []
        # for i in range(len(C_lm)):
        #     N = (l+1)*(l+2)/2 - included_coef
        #     C_row.append(C_lm[i])
        #     S_row.append(S_lm[i])
        #     if i >= N -1:
        #         l += 1
        #         self.C_lm.append(C_row)
        #         self.S_lm.append(S_row)
        #         C_row = []
        #         S_row = []

        # if self.degree is None:
        #     self.degree = l -2
        
        return 
    
    def compute_acc(self, positions=None):
        "Compute the acceleration for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions
        
        positions = np.reshape(positions, (len(positions)*3))
        
        n1, n2, n1q, n2q = compute_n_matrices(self.degree)
        accelerations = compute_acc_parallel(positions, self.degree, self.mu, self.radEquator, n1, n2, n1q, n2q, self.C_lm, self.S_lm)      
        
        self.accelerations = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/3), 3))
        return self.accelerations

