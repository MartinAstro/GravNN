import hashlib
import inspect
import json
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np

import GravNN
from GravNN.Networks.Model import PINNGravityModel


class SkipNonSerializable(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return json.JSONEncoder.default(self, obj)
        except TypeError:
            return None  # or return


class ExperimentBase(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.experiment_name = self.__class__.__name__
        self.unique_hash = self.generate_hash(*args, **kwargs)

        self.override = False
        self.loaded = False
        self.set_save_directory()
        pass

    def get_grav_model_id(self, args_with_names):
        grav_model = args_with_names.get("model", None)
        if grav_model is None:
            if "model" in list(args_with_names.keys())[0]:
                grav_model = args_with_names.values()[0]
            else:
                exit("No gravity model found in to experiment")

        if isinstance(grav_model, PINNGravityModel):
            model_id = str(grav_model.config["id"][0])
        else:
            model_id = grav_model.id
        return model_id

    def generate_hash(self, *args, **kwargs):
        # map all arguments to their names
        params = inspect.signature(self.__init__).parameters
        args_names = list(params.keys())
        args_with_names = dict(zip(args_names, args))

        # Hash should account for the gravity model used
        grav_model_id = self.get_grav_model_id(args_with_names)

        # Convert the sorted input arguments to a JSON string
        all_args = {**args_with_names, **kwargs, "grav_model_id": grav_model_id}
        input_data = json.dumps(all_args, sort_keys=True, cls=SkipNonSerializable)

        # Generate a SHA256 hash of the input data based on string
        hash_obj = hashlib.sha256(input_data.encode())
        unique_hash = hash_obj.hexdigest()

        return unique_hash

    def __hash__(self):
        return self.unique_hash

    def set_save_directory(self):
        GravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__))
        self.experiment_dir = (
            f"{GravNN_dir}/../Data/Experiments/{self.experiment_name}/"
        )
        self.experiment_dir += f"{self.unique_hash}/"
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _set_attributes(self, data):
        for key, value in data.items():
            setattr(self, key, value)

    def save(self, data):
        with open(self.experiment_dir + "exp.data", "wb") as file:
            pickle.dump(data, file)
            print("Data saved successfully.")

    def load(self):
        try:
            with open(self.experiment_dir + "exp.data", "rb") as file:
                data = pickle.load(file)
                print("Data loaded successfully.")
                self._set_attributes(data)
            return data
        except FileNotFoundError:
            print("No data found. Generating...")
            return None

    def run(self, override=False):
        # Load expensive data if it exists and isn't being overriden
        if not override:
            self.load()

        data = self.generate_data()
        self.save(data)
        return data

    @abstractmethod
    def generate_data(self):
        pass
