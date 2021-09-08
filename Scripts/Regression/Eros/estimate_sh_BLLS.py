from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Configs import *
from GravNN.Regression.BLLS import BLLS
from GravNN.Networks.Data import get_raw_data, get_preprocessed_data
from GravNN.Regression.utils import format_coefficients, save

def main():  
    planet = Eros()
    config = get_default_eros_config()
    max_deg = 4
    remove_deg = -1
    for N_train in [2500, 2500//2, 2500//4]:
        for noise in [0.0, 0.1, 0.2]:
            for max_deg in [4, 8, 16]:
                for remove_deg in [-1, 0]:
                    config['acc_noise'] = [noise]
                    config["N_train"] = [N_train]
                    config['scale_by'] = ['none']
                    #x_train, a_train, u_train, x_val, a_val, u_val = get_preprocessed_data(config)
                    dataset, val_dataset, transformers = get_preprocessed_data(config)
                    x_train = dataset[0]
                    a_train = dataset[2]
                    regressor = BLLS(max_deg=max_deg, planet=planet, remove_deg=remove_deg)
                    results = regressor.update(x_train, a_train, iterations=10)
                    C_lm, S_lm = format_coefficients(results, regressor.N, regressor.remove_deg)
                    file_name = "GravNN/Files/GravityModels/Regressed/Eros/%s/BLLS/N_%d/M_%d/%d/regressed_%.2f.csv" % \
                        (config['distribution'][0].__name__, max_deg, remove_deg, config['N_train'][0], config['acc_noise'][0])
                    save(file_name, planet, C_lm, S_lm)

if __name__ == "__main__":
    main()