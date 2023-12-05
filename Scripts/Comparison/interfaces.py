import copy
import os
import pickle
import time
from pathlib import Path

import numpy as np

import GravNN
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Analysis.SurfaceExperiment import SurfaceExperiment
from GravNN.Analysis.TimeEvaluationExperiment import TimeEvaluationExperiment
from GravNN.Analysis.TrajectoryExperiment import TestModel, TrajectoryExperiment
from GravNN.GravityModels.ELM import ELM
from GravNN.GravityModels.HeterogeneousPoly import (
    generate_heterogeneous_model,
)
from GravNN.GravityModels.Mascons import Mascons
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Networks.Configs import *
from GravNN.Networks.Model import PINNGravityModel, load_config_and_model
from GravNN.Networks.Saver import ModelSaver, count_nonzero_params
from GravNN.Networks.utils import (
    configure_run_args,
    configure_tensorflow,
)
from GravNN.Regression.BLLS import BLLS_PM, format_coefficients
from GravNN.Regression.ELMRegressor import OS_ELM
from GravNN.Regression.MasconRegressor import MasconRegressorSequential
from GravNN.Regression.SHRegression import SHRegressorSequential
from GravNN.Regression.utils import save


# Decorators
def time_train_method(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        self.train_duration = end_time - start_time
        return result

    return wrapper


def save_time(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        filename = self.filename
        time_file = os.path.splitext(filename)[0] + "_time.data"
        with open(str(time_file), "wb") as f:
            pickle.dump(self.train_duration, f)
        return result

    return wrapper


class ModelInterface:
    def configure(self, config):
        "Configure the gravity model regressor"
        pass

    def load(self):
        "Take the configuration data, locate idx, and load trained model"
        raise NotImplementedError("Must implement load method")

    def get_model(self):
        "Get the gravity model that can be evaluated"
        raise NotImplementedError("Must implement get_model method")

    def count_params(self):
        "Function that computes number of parameters in model"
        raise NotImplementedError("Must implement count_params method")

    def save_location(self, unique_idx, ext):
        directory = Path(os.path.abspath(os.path.dirname(GravNN.__file__)))
        filename = directory / f"../Data/Comparison/{unique_idx}.{ext}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = str(filename)

    def train(self, data):
        "Fit the gravity model to the data"
        raise NotImplementedError("Must implement train method")

    def evaluate(self, override=False):
        model = self.get_model()
        exp = ExtrapolationExperiment(
            model,
            self.config,
            points=5000,
            extrapolation_bound=100,
        )
        exp.test_dist_2_surf_idx = None  # hack to avoid needing to reorder indices
        exp.test_r_surf = None
        exp.run(override)
        self.extrap_exp = exp

        planet = self.config["planet"][0]
        R = planet.radius
        exp = PlanesExperiment(
            model,
            self.config,
            bounds=[-3 * R, 3 * R],
            samples_1d=200,
        )
        exp.run(override)
        self.plane_exp = exp

        true_model = generate_heterogeneous_model(planet, planet.obj_200k)
        initial_state = np.array(
            [
                2.88000000e04,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -1.81246412e-07,
                4.14643442e00,
            ],
        )
        period = 24 * 3600
        rot_rate = 2 * np.pi / (3600 * 24)
        omega_vec = np.array([0, 0, rot_rate * 10])
        test_model = TestModel(model, "PINN", "g")
        exp = TrajectoryExperiment(
            true_model,
            [test_model],
            initial_state=initial_state,
            pbar=True,
            period=period,
            t_mesh_density=1000,
            omega_vec=omega_vec,
        )
        exp.run(override)
        self.trajectory_exp = exp

        surface_exp = SurfaceExperiment(model, true_model)
        surface_exp.run(override)
        self.surface_exp = surface_exp

        self.time_exp = TimeEvaluationExperiment(model, 1000, R)
        self.time_exp.run(override)

        self.params = self.count_params()

    def save(self):
        "Save the fit gravity model"
        raise NotImplementedError("Must implement save method")


class SphericalHarmonicWrapper(ModelInterface):
    def configure(self, config):
        degree = config["deg"][0]
        self.planet = config["planet"][0]
        model_name = config["model_name"][0]
        self.config = config
        self.degree = degree

        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        unique_idx = f"{model_name}_{self.degree}_{N_train}_{acc_noise}"
        self.save_location(unique_idx, "txt")

    @time_train_method
    def train(self, dataset):
        x = dataset.raw_data["x_train"]
        a = dataset.raw_data["a_train"]

        # Configure Parameters
        N = self.degree
        REMOVE_DEG = -1

        # Initialize the regressor
        self.regressor = SHRegressorSequential(
            N,
            max_param=5000,
            planet=self.planet,
            max_batch_size=100,
        )
        results = self.regressor.update(x, a)

        self.C_lm, self.S_lm = format_coefficients(results, self.degree, REMOVE_DEG)
        self.results = results

    def load(self):
        self.model = self.get_model()

    @save_time
    def save(self):
        save(self.filename, self.planet, self.C_lm, self.S_lm)

    def get_model(self):
        self.model = SphericalHarmonics(self.filename, self.degree)
        return self.model

    def count_params(self):
        return self.degree * (self.degree + 1)


class PMWrapper(SphericalHarmonicWrapper):
    def configure(self, config):
        degree = 0
        self.planet = config["planet"][0]
        model_name = config["model_name"][0]
        self.regressor = BLLS_PM(degree, self.planet)
        self.config = config
        self.degree = degree

        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        unique_idx = f"{model_name}_{self.degree}_{N_train}_{acc_noise}"
        self.save_location(unique_idx, "txt")

    @time_train_method
    def train(self, dataset):
        x = dataset.raw_data["x_train"]
        a = dataset.raw_data["a_train"]
        results = self.regressor.update(x, a)
        results = np.concatenate((results, np.zeros((1))), axis=0)
        REMOVE_DEG = -1
        self.C_lm, self.S_lm = format_coefficients(results, self.degree, REMOVE_DEG)
        self.results = results

    def load(self):
        self.model = self.get_model()

    def get_model(self):
        SH_model = SphericalHarmonics(self.filename, self.degree)
        mu = SH_model.C_lm[0, 0] * self.planet.mu
        planet = copy.deepcopy(self.config["planet"][0])
        planet.mu = mu
        PM_model = PointMass(planet)
        self.model = PM_model
        return PM_model

    def count_params(self):
        return 1


class PolyhedralWrapper(ModelInterface):
    def configure(self, config):
        self.config = config
        self.shape = config["shape"][0]
        self.model_name = config["model_name"][0]
        unique_idx = f"{self.model_name}_{self.shape}"
        self.save_location(unique_idx, "txt")

    @time_train_method
    def train(self, dataset):
        pass

    @save_time
    def save(self):
        pass

    def load(self):
        self.model = self.get_model()

    def get_model(self):
        planet = self.config["planet"][0]
        self.model = Polyhedral(planet, self.shape)
        return self.model

    def count_params(self):
        return (
            self.model.mesh.vertices.shape[0] * 3
            + self.model.mesh.faces.shape[0] * 3 / 2
        )  # Float + long int


class PINNWrapper(ModelInterface):
    def configure(self, config):
        hparams = {}
        configure_run_args(config, hparams)
        configure_tensorflow(config)

        # if the data is noisy, implement an early stop
        if self.config["acc_noise"][0] == 0.1:
            self.config["early_stop"] = [True]

        self.model = PINNGravityModel(config)
        self.config = config
        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        num_units = self.config["num_units"][0]
        model_name = self.config["model_name"][0]

        unique_idx = f"{model_name}_{num_units}_{N_train}_{acc_noise}"
        self.save_location(unique_idx, "data")

    @time_train_method
    def train(self, dataset):
        self.model.train(dataset)

    @save_time
    def save(self):
        saver = ModelSaver(self.model)
        saver.save(self.filename)

    def load(self):
        config, model = load_config_and_model(self.filename, idx=-1)
        self.config = config
        self.model = model

    def get_model(self):
        return self.model

    def count_params(self):
        return count_nonzero_params(self.model)


class MasconWrapper(ModelInterface):
    def configure(self, config):
        planet = config["planet"][0]
        model_name = config["model_name"][0]
        obj_file = config["obj_file"][0]
        N_masses = config["elements"][0]
        acc_noise = config["acc_noise"][0]
        N_train = config["N_train"][0]
        self.config = config
        self.regressor = MasconRegressorSequential(planet, obj_file, N_masses)
        unique_idx = f"{model_name}_{N_masses}_{N_train}_{acc_noise}"
        self.save_location(unique_idx, "csv")

    @time_train_method
    def train(self, dataset):
        x = dataset.raw_data["x_train"]
        a = dataset.raw_data["a_train"]
        self.regressor.update(x, a, mass_batch_size=1000)

    @save_time
    def save(self):
        self.regressor.save(self.filename)

    def load(self):
        self.model = self.get_model()

    def get_model(self):
        planet = self.config["planet"][0]
        self.model = Mascons(planet, self.filename)
        return self.model

    def count_params(self):
        return self.regressor.N_masses * 4


class ELMWrapper(ModelInterface):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        configure_tensorflow(config)
        self.config = config
        self.regressor = OS_ELM(
            n_input_nodes=3,
            n_hidden_nodes=config["num_units"][0],
            n_output_nodes=3,
            k=2e-6,
        )
        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        num_units = self.config["num_units"][0]
        model_name = self.config["model_name"][0]

        unique_idx = f"{model_name}_{num_units}_{N_train}_{acc_noise}"
        self.save_location(unique_idx, "data")

    @time_train_method
    def train(self, dataset):
        x = dataset.raw_data["x_train"]
        a = dataset.raw_data["a_train"]
        self.regressor.update(x, a, init_batch=1000)

    @save_time
    def save(self):
        # save model to pickle
        self.regressor.save(self.filename)

    def load(self):
        self.model = self.get_model()

    def get_model(self):
        self.model = ELM(self.filename)
        return self.model

    def count_params(self):
        return self.model.n_hidden_nodes * 3 + self.model.n_hidden_nodes * 3


def make_experiments(exp_list):
    experiments = []
    for exp in exp_list:
        for dictionary in exp:
            for k, v in dictionary.items():
                dictionary[k] = [v]
            experiments.append(dictionary)
    return experiments


def select_model(model_name):
    if "PINN" in model_name.upper() or "TNN" in model_name.upper():
        return PINNWrapper()
    elif model_name.upper() == "MASCONS":
        return MasconWrapper()
    elif model_name.upper() == "POLYHEDRAL":
        return PolyhedralWrapper()
    elif model_name.upper() == "SH":
        return SphericalHarmonicWrapper()
    elif model_name.upper() == "PM":
        return PMWrapper()
    elif model_name.upper() == "ELM":
        return ELMWrapper()
    else:
        raise Exception("Model name not recognized")
