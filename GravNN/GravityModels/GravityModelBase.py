import hashlib
import inspect
import json
import logging
import os
import pickle
from abc import ABC, abstractmethod


class SkipNonSerializable(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return None  # or return


class GravityModelBase(ABC):
    verbose = True

    def __init__(self, *args, **kwargs):
        """Base class responsible for generating the accelerations for a given trajectory / distribution"""
        self._trajectory = None
        self.accelerations = None
        self.potentials = None
        self.file_directory = None
        self.id = self.generate_hash(*args, **kwargs)
        return

    def generate_hash(self, *args, **kwargs):
        # map all arguments to their names
        params = inspect.signature(self.__init__).parameters
        args_names = list(params.keys())
        args_with_names = dict(zip(args_names, args))

        # Convert the sorted input arguments (wo kwargs) to a JSON string
        input_data = json.dumps(
            args_with_names,
            sort_keys=True,
            cls=SkipNonSerializable,
        )
        input_data += self.__class__.__name__

        # Generate a SHA256 hash of the input data based on string
        hash_obj = hashlib.sha256(input_data.encode())
        unique_hash = hash_obj.hexdigest()

        return unique_hash

    def configure(self, trajectory):
        if trajectory is not None:
            # If a trajectory is defined, save all corresponding data within that trajectory's directory
            self.file_directory = trajectory.file_directory
            self._trajectory = trajectory
            self.positions = trajectory.positions
        else:
            self.file_directory = (
                os.path.splitext(__file__)[0] + "/../../Files/Trajectories/Custom/"
            )
        self.generate_full_file_directory()

    def save(self):
        # Create the directory/file and dump the acceleration and potential if computed
        if not os.path.exists(self.file_directory):
            os.makedirs(self.file_directory)

        if self.accelerations is not None:
            with open(self.file_directory + "acceleration.data", "wb") as f:
                pickle.dump(self.accelerations, f)

        if self.potentials is not None:
            with open(self.file_directory + "potential.data", "wb") as f:
                pickle.dump(self.potentials, f)
        return

    def load(self, override=False):
        """Load saved acceleration and potential values for a given trajectory / distribution, or
        generate them if they dont exist

        Args:
            override (bool, optional): Flag determining if the acceleration and potentials should be overwritten. Defaults to False.

        Returns:
            GravityModelBase: self
        """
        self.load_acceleration(override)
        self.load_potential(override)
        return self

    def load_acceleration(self, override=False):
        # Check if the file exists and either load the acceleration or generate it
        if (
            os.path.exists(self.file_directory + "acceleration.data")
            and override is False
        ):
            if self.verbose:
                print(
                    "Found existing acceleration.data at "
                    + os.path.relpath(self.file_directory),
                )
            with open(self.file_directory + "acceleration.data", "rb") as f:
                self.accelerations = pickle.load(f)
                return self.accelerations
        else:
            if self.verbose:
                print(
                    "Generating acceleration at "
                    + os.path.relpath(self.file_directory),
                )
            # Note: compute_acceleration() might generate the potential values too for some representations.
            # For example, computing the potential of the polyhedral model adds one extra step
            # to the acceleration calculation. Rather than rerunning the entire algorithm to
            # compute the potential, the calculations are bundled together for efficiency.
            self.compute_acceleration()
            self.save()
        return

    def load_potential(self, override=False):
        # Check if the file exists and either load the potential or generate it
        if os.path.exists(self.file_directory + "potential.data") and override is False:
            if self.verbose:
                print(
                    "Found existing potential.data at "
                    + os.path.relpath(self.file_directory),
                )
            with open(self.file_directory + "potential.data", "rb") as f:
                self.potentials = pickle.load(f)
                return self.potentials
        else:
            if self.verbose:
                print("Generating potential at " + os.path.relpath(self.file_directory))
            self.compute_potential()
            self.save()
            return self.potentials

    @abstractmethod
    def generate_full_file_directory(self):
        pass

    @abstractmethod
    def compute_acceleration(self):
        pass

    @abstractmethod
    def compute_potential(self):
        pass

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value):
        logging.info("Setting Trajectory + rerunning configure!")
        self._trajectory = value
        self.configure(self._trajectory)
