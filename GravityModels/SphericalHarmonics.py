import csv
import os, sys
import numpy as np
sys.path.append(os.path.dirname(__file__) + "/../build/PinesAlgorithm/")
import PinesAlgorithm
from GravityModels.GravityModelBase import GravityModelBase
from Trajectories.TrajectoryBase import TrajectoryBase

class SphericalHarmonics(GravityModelBase):
    def __init__(self, file_name, degree, trajectory=None):
        super().__init__()
        self.file = file_name
        self.degree = degree

        self.mu = None
        self.radEquator = None
        self.C_lm = None
        self.S_lm = None

        self.loadSH()
        self.configure(trajectory)
        pass

    def generate_full_file_directory(self):
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "_" + str(self.degree) + "/"
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

    def compute_acc(self, positions=None):
        "Compute the acceleration for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions
        
        positions = np.reshape(positions, (len(positions)*3))
        pines = PinesAlgorithm.PinesAlgorithm(self.radEquator, self.mu, self.degree)
        accelerations = pines.compute_acc(positions, self.C_lm, self.S_lm)
        self.accelerations = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/3), 3))
        return self.accelerations