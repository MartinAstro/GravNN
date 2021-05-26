from GravNN.Networks.Constraints import *
from GravNN.Networks.Activations import * 
from GravNN.Networks.Configs.Default_Configs import *
from GravNN.Networks.Networks import *
def load_hparams_to_config(hparams, config):

    for key, value in hparams.items():
        config[key] = [value]

    if config['PINN_constraint_fcn'][0] == 'pinn_A':
        config['PINN_constraint_fcn'] = [pinn_A]
    elif config['PINN_constraint_fcn'][0] == 'no_pinn':
        config['PINN_constraint_fcn'] = [no_pinn]
    else:
        exit("Couldn't load the constraint!")

    if config['activation'][0] == 'bent_identity':
        config['activation'] = [bent_identity]
    
    try:
        if 'adam' in config['optimizer'][0]:
            config['optimizer'][0] = tf.keras.optimizers.Adam()
    except:
        pass

    try: 
        if 'rmsprop' in config['optimizer'][0]:
            config['optimizer'][0] = tf.keras.optimizers.RMSprop()
    except:
        pass
        
    if 'num_units' in config:
        for i in range(1, len(config['layers'][0])-1):
            config['layers'][0][i] = config['num_units'][0]
    
    if config['network_type'][0] == 'traditional':
        config['network_type'] = [TraditionalNet]
    elif config['network_type'][0] == 'resnet':
        config['network_type'] = [ResNet]
    else:
        exit("Network type (%s) is not defined! Exiting." % config['network_type'][0])

    return config



def get_default_config(PINN_constraint_fcn_val, planet_val):
    if PINN_constraint_fcn_val == 'no_pinn':
        if planet_val == 'earth':
            configurations = {"Default" : get_default_earth_config() }
        elif planet_val == 'moon':
            configurations = {"Default" : get_default_moon_config() }
        elif planet_val == 'eros':
            configurations = {"Default" : get_default_eros_config() }
        else:
            exit("Configuration not specified")
    else:
        if planet_val == 'earth':
            configurations = {"Default" : get_default_earth_pinn_config() }
        elif planet_val == 'moon':
            configurations = {"Default" : get_default_moon_pinn_config() }
        elif planet_val == 'eros':
            configurations = {"Default" : get_default_eros_pinn_config() }
        else:
            exit("Configuration not specified")
    return configurations
