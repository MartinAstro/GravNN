
import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"

import copy
import pickle
import sys
import time

from GravNN.Configs.Default_Configs import get_default_earth_config, get_default_eros_config

np.random.seed(1234)
tf.random.set_seed(0)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def get_default_earth_suite():
    config = get_default_earth_config()
    config_1 = copy.deepcopy(config)
    config_1.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 3]]})
    
    config_2 = copy.deepcopy(config)
    config_2.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]]})

    config_3 = copy.deepcopy(config)
    config_3.update({'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]})

    configurations = {
          "1" : config_1,
          "2" : config_2,
          "3" : config_3,
          }  
    return configurations

def get_default_eros_suite():
    config = get_default_eros_config()
    config_1 = copy.deepcopy(config)
    config_1.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 3]]})
    
    config_2 = copy.deepcopy(config)
    config_2.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]]})

    config_3 = copy.deepcopy(config)
    config_3.update({'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]})

    configurations = {
          "1" : config_1,
          "2" : config_2,
          "3" : config_3,
          }  
    return configurations


def get_default_moon_suite():
    config = get_default_moon_config()
    config_1 = copy.deepcopy(config)
    config_1.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 3]]})
    
    config_2 = copy.deepcopy(config)
    config_2.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]]})

    config_3 = copy.deepcopy(config)
    config_3.update({'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]})

    configurations = {
          "1" : config_1,
          "2" : config_2,
          "3" : config_3,
          }  
    return configurations
