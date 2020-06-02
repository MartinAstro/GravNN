import os, sys
import pickle
from abc import ABC, abstractmethod
class TrajectoryBase(ABC):
    celestial_body = None # Celestial Body
    file_directory = os.path.splitext(__file__)[0]  + "/../../Files/Trajectories/" 

    positions = None
    def __init__(self, celestial_body):
        self.celestial_body = celestial_body
        self.generate_full_file_directory()
        self.load()
        return

    def save(self):
        # Create the directory/file and dump the position
        if not os.path.exists(self.file_directory):
            os.makedirs(self.file_directory)
        with open(self.file_directory + "trajectory.data",'wb') as f:
            pickle.dump(self.positions, f)
        return

    def load(self):
        # Check if the file exists and either load the positions or generate the position
        if os.path.exists(self.file_directory + "trajectory.data"):
            with open(self.file_directory + "trajectory.data", 'rb') as f:
                self.positions = pickle.load(f)
                return self.positions
        else:
            self.generate()
            self.save()
            return self.positions

    @abstractmethod
    def generate_full_file_directory(self):
        pass

    @abstractmethod
    def generate(self):
        pass
        
