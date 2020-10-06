import csv
import os, sys
import numpy as np
sys.path.append(os.path.dirname(__file__) + "/../build/PinesAlgorithm/")
import PinesAlgorithm
from GravityModels.GravityModelBase import GravityModelBase
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase

class ArtificialModel(GravityModelBase):
    def __init__(self, feature_range, trajectory=None):
        super().__init__()

        self.mu = None
        self.radEquator = None

        self.feature_range = feature_range
        
        self.configure(trajectory)

        pass

    def generate_full_file_directory(self):
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "_" + os.path.basename(self.file).split('.')[0] +  "/"
        pass



    def compute_acc(self, positions=None):
        "Compute the acceleration for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions
        
        positions = np.reshape(positions, (len(positions)*3))

        accelerations = np.zeros(positions.shape) # this could be a random normal distribution

        # add craters
        num_craters = 3
        x_craters = np.random.randint(0, positions.shape[0], num_craters)
        y_craters = np.random.randint(0, positions.shape[1], num_craters)
        mag_craters = np.random.rand(num_craters)

        num_mountains = 3
        x_mountain =  np.random.randint(0, positions.shape[0], num_mountains)
        y_mountain = np.random.randint(0, positions.shape[1], num_mountains)
        mag_craters = np.random.rand(num_mountains)

        


        
        pines = PinesAlgorithm.PinesAlgorithm(self.radEquator, self.mu, self.degree)
        accelerations = pines.compute_acc(positions, self.C_lm, self.S_lm)
        self.accelerations = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/3), 3))
        return self.accelerations