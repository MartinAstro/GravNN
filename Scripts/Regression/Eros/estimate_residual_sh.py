from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Configs import *
from GravNN.Networks.Data import get_preprocessed_data
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import format_coefficients, save


def compute_coefficients(dataset, max_deg, planet):
    x_train, a_train = dataset[0], dataset[2]
    regressor = BLLS(max_deg=max_deg, planet=planet, remove_deg=0)
    results = regressor.update(x_train, a_train, iterations=10)
    C_lm, S_lm = format_coefficients(results, regressor.N, regressor.remove_deg)
    return C_lm, S_lm


def main():
    planet = Eros()
    config = get_default_eros_config()

    for noise in [0.0, 0.2]:
        for max_deg in [4, 8, 16]:
            config["radius_max"] = [Eros().radius * 3]
            config["radius_min"] = [Eros().radius * 2]
            config["acc_noise"] = [noise]
            config["N_train"] = [5000]
            config["scale_by"] = ["none"]

            dataset, _, _ = get_preprocessed_data(config)
            C_lm, S_lm = compute_coefficients(dataset, max_deg, planet)
            file_name = (
                "GravNN/Files/GravityModels/Regressed/Eros/Residual/2R_3R/N_%d_Noise_%.2f.csv"
                % (max_deg, config["acc_noise"][0])
            )
            save(file_name, planet, C_lm, S_lm)

    for noise in [0.0, 0.2]:
        for max_deg in [4, 8, 16]:
            config["radius_max"] = [Eros().radius * 3]
            config["radius_min"] = [0]
            config["acc_noise"] = [noise]
            config["N_train"] = [5000]
            config["scale_by"] = ["none"]

            dataset, _, _ = get_preprocessed_data(config)
            C_lm, S_lm = compute_coefficients(dataset, max_deg, planet)
            file_name = (
                "GravNN/Files/GravityModels/Regressed/Eros/Residual/0R_3R/N_%d_Noise_%.2f.csv"
                % (max_deg, config["acc_noise"][0])
            )
            save(file_name, planet, C_lm, S_lm)

    for noise in [0.0, 0.2]:
        for max_deg in [4, 8, 16]:
            config["radius_max"] = [Eros().radius * 3]
            config["radius_min"] = [0]
            config["acc_noise"] = [noise]
            config["N_train"] = [5000]
            config["scale_by"] = ["none"]

            config["extra_distribution"] = [RandomDist]
            config["extra_radius_min"] = [0]
            config["extra_radius_max"] = [Eros().radius * 2]
            config["extra_N_dist"] = [1000]
            config["extra_N_train"] = [500]
            config["extra_N_val"] = [500]

            dataset, _, _ = get_preprocessed_data(config)
            C_lm, S_lm = compute_coefficients(dataset, max_deg, planet)
            file_name = (
                "GravNN/Files/GravityModels/Regressed/Eros/Residual/2R_3R_Plus/N_%d_Noise_%.2f.csv"
                % (max_deg, config["acc_noise"][0])
            )
            save(file_name, planet, C_lm, S_lm)


if __name__ == "__main__":
    main()
