import os, sys
import pickle
from abc import ABC, abstractmethod
class TrajectoryBase(ABC):

    def __init__(self, **kwargs):
        #positions
        self.file_directory = os.path.splitext(__file__)[0]  + "/../../Files/Trajectories/" 
        self.generate_full_file_directory()
        self.load(override=kwargs.get('override', False))
        return

    def save(self):
        # Create the directory/file and dump the position
        if not os.path.exists(self.file_directory):
            os.makedirs(self.file_directory, exist_ok=True)
        with open(self.file_directory + "trajectory.data",'wb') as f:
            pickle.dump(self.positions, f)
            try:
                pickle.dump(self.times, f)
            except:
                pass
        return

    def load(self, override=False):
        # Check if the file exists and either load the positions or generate the position
        if os.path.exists(self.file_directory + "trajectory.data") and not override:
            with open(self.file_directory + "trajectory.data", 'rb') as f:
                self.positions = pickle.load(f)
                try:
                    self.times = pickle.load(f)
                except:
                    pass
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
        
