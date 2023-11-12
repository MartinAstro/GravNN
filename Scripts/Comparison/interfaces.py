import os
import pickle
from pathlib import Path

import numpy as np

import GravNN
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Analysis.TrajectoryExperiment import TestModel, TrajectoryExperiment
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
from GravNN.Regression.ELM import OS_ELM
from GravNN.Regression.MasconRegressor import MasconRegressor
from GravNN.Regression.RLLS import RLLS
from GravNN.Regression.utils import save


class ModelInterface:
    def configure(self, config):
        "Configure the gravity model regressor"
        pass

    def train(self, data):
        "Fit the gravity model to the data"
        pass

    def save(self):
        "Save the fit gravity model"
        pass

    def load(self):
        "Take the configuration data, locate idx, and load trained model"

    def get_model(self):
        "Get the gravity model that can be evaluated"
        pass

    def count_params(self):
        "Function that computes number of parameters in model"
        pass

    def evaluate(self, override=False):
        model = self.get_model()
        exp = ExtrapolationExperiment(model, self.config, points=5000)
        exp.run(override)
        self.extrap_exp = exp

        planet = self.config["planet"][0]
        R = planet.radius
        exp = PlanesExperiment(
            model,
            self.config,
            bounds=[-3 * R, 3 * R],
            samples_1d=100,
        )
        exp.run(override)
        self.plane_exp = exp

        true_model = generate_heterogeneous_model(planet, planet.obj_8k)
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
            period=period,
            t_mesh_density=1000,
            omega_vec=omega_vec,
        )
        exp.run(override)
        self.trajectory_exp = exp
        self.params = self.count_params()


class SphericalHarmonicWrapper(ModelInterface):
    def configure(self, config):
        degree = config["deg"][0]
        self.planet = config["planet"][0]
        # self.regressor = BLLS(degree, self.planet)
        # self.regressor = RLLS(degree, self.planet)
        self.config = config
        self.degree = degree

        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        unique_idx = f"{self.degree}_{N_train}_{acc_noise}"
        directory = Path(os.path.abspath(os.path.dirname(GravNN.__file__)))
        self.filename = (
            directory / f"Files/GravityModels/SH/BLLS/Comparison/{unique_idx}.txt"
        )
        self.filename = str(self.filename)

    def train(self, dataset):
        x = dataset.raw_data["x_train"]
        a = dataset.raw_data["a_train"]

        # Configure Parameters
        N = self.degree
        REMOVE_DEG = -1
        P0_diag = 0
        dim = (N + 2) * (N + 1) - (REMOVE_DEG + 2) * (REMOVE_DEG + 1)
        init_batch = 1000 if len(x) > 1000 else len(x)
        # Configure State
        x0 = np.zeros((dim,))
        x0[0] = 0.0 if REMOVE_DEG != -1 else 1.0

        # Define uncertainty
        P0 = np.identity(dim) * P0_diag

        # Functionally no meas uncertainty, but 1e-16 needed to avoid singularities
        Rk = np.identity(3) + 1e-16

        # Initialize the regressor
        self.regressor = RLLS(N, self.planet, x0, P0, Rk, REMOVE_DEG)
        results = self.regressor.update(x, a, init_batch=init_batch, history=True)

        self.C_lm, self.S_lm = format_coefficients(results, self.degree, REMOVE_DEG)
        self.results = results

    def load(self):
        self.model = self.get_model()

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
        self.regressor = BLLS_PM(degree, self.planet)
        self.config = config
        self.degree = degree

        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        unique_idx = f"{self.degree}_{N_train}_{acc_noise}"
        directory = Path(os.path.abspath(os.path.dirname(GravNN.__file__)))
        self.filename = (
            directory / f"Files/GravityModels/PM/BLLS/Comparison/{unique_idx}.txt"
        )
        self.filename = str(self.filename)

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
        planet = self.config["planet"][0]
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

    def train(self, dataset):
        pass

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
            self.mesh.vertices.shape[0] * 3 + self.mesh.faces.shape[0] * 3 / 2
        )  # Float + long int


class PINNWrapper(ModelInterface):
    def configure(self, config):
        hparams = {
            "learning_rate": [0.0001],
            "batch_size": [2**11],
            "epochs": [20000],
        }
        configure_run_args(config, hparams)
        configure_tensorflow(config)
        self.model = PINNGravityModel(config)

        directory = Path(os.path.abspath(os.path.dirname(GravNN.__file__)))
        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        num_units = self.config["num_units"][0]
        model_name = self.config["model_name"][0]

        unique_idx = f"{num_units}_{N_train}_{acc_noise}_{model_name}"
        self.filename = directory + f"../Data/Comparison/{unique_idx}.data"

    def train(self, dataset):
        self.model.train(dataset)

    def save(self):
        saver = ModelSaver(self.model)
        saver.save(self.filename)

    def load(self):
        config, model = load_config_and_model(self.filename, idx=-1)
        self.model = model

    def get_model(self):
        return self.model

    def count_params(self):
        return count_nonzero_params(self.model)


class NNWrapper(ModelInterface):
    def configure(self, config):
        hparams = PINN_I()
        hparams.update(ReduceLrOnPlateauConfig())
        hparams.update(
            {
                "epochs": [20000],
            },
        )
        configure_run_args(config, hparams)
        configure_tensorflow(config)
        self.model = PINNGravityModel(config)

        directory = Path(os.path.abspath(os.path.dirname(GravNN.__file__)))
        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        num_units = self.config["num_units"][0]
        model_name = self.config["model_name"][0]

        unique_idx = f"{num_units}_{N_train}_{acc_noise}_{model_name}"
        self.filename = directory + f"../Data/Comparison/{unique_idx}.data"

    def train(self, dataset):
        self.model.train(dataset)

    def save(self):
        saver = ModelSaver(self.model)
        saver.save(self.filename)

    def load(self):
        config, model = load_config_and_model(self.filename, idx=-1)
        self.model = model

    def get_model(self):
        return self.model

    def count_params(self):
        return count_nonzero_params(self.model)


class MasconWrapper(ModelInterface):
    def configure(self, config):
        planet = config["planet"][0]
        obj_file = config["obj_file"][0]
        N_masses = config["elements"][0]
        acc_noise = self.config["acc_noise"][0]
        N_train = self.config["N_train"][0]
        self.filename = f"{N_masses}_{N_train}_{acc_noise}.csv"
        self.regressor = MasconRegressor(planet, obj_file, N_masses)

    def train(self, dataset):
        x = dataset.raw_data["x_train"]
        a = dataset.raw_data["a_train"]
        self.regressor.update(x, a)

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
    def configure(self, config):
        configure_tensorflow(config)
        self.model = OS_ELM(
            n_input_nodes=3,
            n_hidden_nodes=config["num_units"][0],
            n_output_nodes=3,
            k=2e-6,
        )
        N_train = self.config["N_train"][0]
        acc_noise = self.config["acc_noise"][0]
        num_units = self.config["num_units"][0]
        model_name = self.config["model_name"][0]

        directory = Path(os.path.abspath(os.path.dirname(GravNN.__file__)))

        unique_idx = f"{num_units}_{N_train}_{acc_noise}_{model_name}"
        self.filename = directory + f"../Data/Comparison/{unique_idx}.data"
        unique_idx = f"{num_units}_{N_train}_{acc_noise}_{model_name}"
        self.filename = directory + f"../Data/Comparison/elm_{unique_idx}.data"

    def train(self, dataset):
        x = dataset.raw_data["x_train"]
        a = dataset.raw_data["a_train"]
        self.model.update(x, a)

    def save(self):
        # save model to pickle
        with open(self.filename, "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        # load pickle
        with open(self.filename, "rb") as f:
            self.model = pickle.load(f)

    def get_model(self):
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
    if "PINN" in model_name.upper():
        return PINNWrapper()
    if "TNN" in model_name.upper():  # traditional NN
        return NNWrapper()
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
