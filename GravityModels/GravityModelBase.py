import os, sys
import pickle
from abc import ABC, abstractmethod
class GravityModelBase(ABC):
    verbose = True
    def __init__(self):
        self.trajectory = None
        self.accelerations = None
        self.file_directory  = None
        return
    
    def configure(self, trajectory):
        if trajectory is not None:
            # If a trajectory is defined, save all corresponding data within that trajectory's directory
            self.file_directory = trajectory.file_directory 
            self.trajectory = trajectory
            self.positions = trajectory.positions
        else:
            self.file_directory = os.path.splitext(__file__)[0]  + "/../../Files/Trajectories/Custom/" 
        self.generate_full_file_directory()

    def save(self):
        # Create the directory/file and dump the acceleration
        if not os.path.exists(self.file_directory):
            os.makedirs(self.file_directory)
        with open(self.file_directory + "acceleration.data",'wb') as f:
            pickle.dump(self.accelerations, f)
        return

    def load(self, override=False):
        # Check if the file exists and either load the positions or generate the acceleration
        if os.path.exists(self.file_directory + "acceleration.data") and override == False:
            if self.verbose:
                print("Found existing acceleration.data at " + self.file_directory)
            with open(self.file_directory+"acceleration.data", 'rb') as f:
                self.accelerations = pickle.load(f)
                return self.accelerations
        else:
            if self.verbose:
                print("Generating acceleration at " + self.file_directory)
            self.compute_acc()
            self.save()
            return self.accelerations

    @abstractmethod
    def generate_full_file_directory(self):
        pass

    @abstractmethod
    def compute_acc(self):
        pass
        