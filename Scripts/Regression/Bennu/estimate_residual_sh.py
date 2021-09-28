from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.Networks.Configs import *
from GravNN.Regression.BLLS import BLLS
from GravNN.Networks.Data import get_raw_data, get_preprocessed_data
from GravNN.Regression.utils import format_coefficients, save

def compute_coefficients(dataset, max_deg, planet):
    x_train, a_train = dataset[0],  dataset[2]
    regressor = BLLS(max_deg=max_deg, planet=planet, remove_deg=0)
    results = regressor.update(x_train, a_train, iterations=10)
    C_lm, S_lm = format_coefficients(results, regressor.N, regressor.remove_deg)
    return C_lm, S_lm

def main():  
    planet = Bennu()
    config = get_default_bennu_config()

    for noise in [0.0, 0.2]:
        for max_deg in [4, 8, 16]:
            config["radius_max"] = [Bennu().radius * 3]
            config["radius_min"] = [Bennu().radius * 2]
            config['acc_noise'] = [noise]
            config["N_train"] = [2500]
            config['scale_by'] = ['none']
            
            dataset, _, _ = get_preprocessed_data(config)
            C_lm, S_lm = compute_coefficients(dataset, max_deg, planet)
            file_name = "GravNN/Files/GravityModels/Regressed/Bennu/Residual/2R_3R/N_%d_Noise_%.2f.csv" % \
                            (max_deg, config['acc_noise'][0])
            save(file_name, planet, C_lm, S_lm)

    for noise in [0.0, 0.2]:
        for max_deg in [4, 8, 16]:
            config["radius_max"] = [Bennu().radius * 3]
            config["radius_min"] = [0]
            config['acc_noise'] = [noise]
            config["N_train"] = [2500]
            config['scale_by'] = ['none']
            
            dataset, _, _ = get_preprocessed_data(config)
            C_lm, S_lm = compute_coefficients(dataset, max_deg, planet)
            file_name = "GravNN/Files/GravityModels/Regressed/Bennu/Residual/0R_3R/N_%d_Noise_%.2f.csv" % \
                            (max_deg, config['acc_noise'][0])
            save(file_name, planet, C_lm, S_lm)

    for noise in [0.0, 0.2]:
        for max_deg in [4, 8, 16]:
            config["radius_max"] = [Bennu().radius * 3]
            config["radius_min"] = [0]
            config['acc_noise'] = [noise]
            config["N_train"] = [2500]
            config['scale_by'] = ['none']
                        
            config["extra_distribution"] = [RandomAsteroidDist]
            config["extra_radius_min"] = [0]
            config["extra_radius_max"] = [Bennu().radius*2]
            config["extra_N_dist"] = [1000]
            config["extra_N_train"] = [250]
            config["extra_N_val"] = [500]

            dataset, _, _ = get_preprocessed_data(config)
            C_lm, S_lm = compute_coefficients(dataset, max_deg, planet)
            file_name = "GravNN/Files/GravityModels/Regressed/Bennu/Residual/2R_3R_Plus/N_%d_Noise_%.2f.csv" % \
                            (max_deg, config['acc_noise'][0])
            save(file_name, planet, C_lm, S_lm)

if __name__ == "__main__":
    main()