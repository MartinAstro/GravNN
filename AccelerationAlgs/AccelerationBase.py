import os, sys
import pickle
from abc import ABC, abstractmethod
class AccelerationBase(ABC):
    body = None 
    trajectory = None
    accelerations = None
    file_directory = None
    def __init__(self):
        self.file_directory  = self.trajectory.file_directory
        self.generate_full_file_directory()
        self.load()
        return

    def save(self):
        # Create the directory/file and dump the acceleration
        if not os.path.exists(self.file_directory):
            os.makedirs(self.file_directory)
        with open(self.file_directory + "acceleration.data",'wb') as f:
            pickle.dump(self.accelerations, f)
        return

    def load(self):
        # Check if the file exists and either load the positions or generate the acceleration
        if os.path.exists(self.file_directory + "acceleration.data"):
            with open(self.file_directory+"acceleration.data", 'rb') as f:
                self.accelerations = pickle.load(f)
                return self.accelerations
        else:
            self.compute_acc()
            self.save()
            return self.accelerations

    @abstractmethod
    def generate_full_file_directory(self):
        pass

    @abstractmethod
    def compute_acc(self):
        pass
        