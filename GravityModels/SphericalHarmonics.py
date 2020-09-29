import csv
import os, sys
import numpy as np
from GravNN.build.PinesAlgorithm import PinesAlgorithm
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase

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

    def loadSH(self):
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
                    if(int(gravRow[0]) > self.degree + 2):# if loading coefficients beyond the maximum desired (+2 for purposes of the algorithm)
                        break
                while int(gravRow[0]) > currDeg:
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
                
            if self.degree is None:
                self.degree = currDeg -2
            self.C_lm = clmList
            self.S_lm = slmList
            return 

    def load_regression(self, reg_solution):
        self.file_directory += "_Regress"
        self.mu = self.trajectory.celestial_body.mu
        self.radEquator =self.trajectory.celestial_body.radius

        reg_solution = np.array(reg_solution)
        self.C_lm = []
        self.S_lm = []
        C_lm = []
        S_lm = []

        # Check if C00 is included as it should be equal to 1.0
        C00_11_included = False
        l = 2
        included_coef = 3
        if abs(reg_solution[0] - 1.0) < 10E-10:
            C00_11_included = True
            l = 0
            included_coef = 0
        else:
            self.C_lm.append([1.0])
            self.C_lm.append([0.0, 0.0])
            self.S_lm.append([0.0])
            self.S_lm.append([0.0, 0.0])


        # Separate C_lm and S_lm
        for i in range(int(len(reg_solution)/2)):
            C_lm.append(reg_solution[2*i])
            S_lm.append(reg_solution[2*i + 1])
        
        # Format coefficients into row by degree
        C_row = []
        S_row  = []
        for i in range(len(C_lm)):
            N = (l+1)*(l+2)/2 - included_coef
            C_row.append(C_lm[i])
            S_row.append(S_lm[i])
            if i >= N -1:
                l += 1
                self.C_lm.append(C_row)
                self.S_lm.append(S_row)
                C_row = []
                S_row = []

        if self.degree is None:
            self.degree = l -2
        
        return 


    def compute_acc(self, positions=None):
        "Compute the acceleration for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions
        
        positions = np.reshape(positions, (len(positions)*3))
        pines = PinesAlgorithm.PinesAlgorithm(self.radEquator, self.mu, self.degree, self.C_lm, self.S_lm)
        accelerations = pines.compute_acc(positions)
        self.accelerations = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/3), 3))
        return self.accelerations

    def compute_acc_components(self, positions=None):
        "Compute the acceleration components for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions
        
        positions = np.reshape(positions, (len(positions)*3))
        pines = PinesAlgorithm.PinesAlgorithm(self.radEquator, self.mu, self.degree, self.C_lm, self.S_lm)
        accelerations = pines.compute_acc_components(positions.tolist())
        total_terms = int(self.degree*(self.degree+1)/2*3)
        components = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/total_terms),total_terms))
        return components