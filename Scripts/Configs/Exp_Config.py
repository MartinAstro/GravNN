    data_config = {
        'planet' : [Earth()],
        'grav_file' : [Earth().sh_hf_file],
        'distribution' : [ExponentialDist],
        'N_dist' : [1000000],
        'N_train' : [40000], 
        'N_val' : [4000],
        'radius_min' : [Earth().radius],
        'radius_max' : [Earth().radius + 420000.0],
        'acc_noise' : [0.00],
        'basis' : [None],# ['spherical'],
        'deg_removed' : [2],
        'include_U' : [False],
        'max_deg' : [1000], 
        'dtype' : ['float32']
        'sh_truth' : ['sh_stats_']
    }
    network_config = {
        'network_type' : [TraditionalNet],
        'PINN_flag' : [False],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
        'activation' : ['tanh'],
        'init_file' : [None],#'2459192.4530671295'],
        'epochs' : [100000],
        'optimizer' : [tf.keras.optimizers.Adam()], #(learning_rate=config['lr_scheduler'][0])
        'batch_size' : [160000],
        'dropout' : [0.0], 
        'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        'a_transformer' : [MinMaxScaler(feature_range=(-1,1))]
    }
    
    # ResNet -- 'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
    # DenseNet -- 'layers' : [[3, dense_layer, [10], dense_layer, [10], dense_layer, [10], dense_layer, 3]],
    # InceptionNet -- 'layers' : [[3, inception_layer, inception_layer, inception_layer, inception_layer, 1]],


    config = {}
    config.update(data_config)
    config.update(network_config)

    orbit = 420000.0 # Earth LEO
    #orbit = 1000.0 # Bennu Survey A 

    config['PINN_flag'] = [False]
    config['N_train'] = [950000] 
    config['batch_size'] = [40000]
    config['epochs'] = [100000]
    config['basis'] = ['spherical']
    config['scale_parameter'] = [orbit/3.0]
    config['invert'] = [False]


    config['distribution'] = [ExponentialDist]
    config['scale_parameter'] = [orbit/10.0]
    config_exp_2_1 = copy.deepcopy(config)
    config_exp_2_1.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 3]],
                    'invert' : [False]
                    })

    
    config_exp_2_2 = copy.deepcopy(config)
    config_exp_2_2.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 3]],

                    'invert' : [True]
                    })

    config_exp_2_3 = copy.deepcopy(config)
    config_exp_2_3.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
                    'invert' : [False]
                    })

    
    config_exp_2_4 = copy.deepcopy(config)
    config_exp_2_4.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
                    'invert' : [True]
                    })

    config_exp_2_5 = copy.deepcopy(config)
    config_exp_2_5.update({'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
                    'invert' : [False]
                    })

    
    config_exp_2_6 = copy.deepcopy(config)
    config_exp_2_6.update({'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
                    'invert' : [True]
                    })
   


    configurations = {

         "1_exp2" : config_exp_2_1,
         "2_exp2" : config_exp_2_2,
         "3_exp2" : config_exp_2_3,
         "4_exp2" : config_exp_2_4,
         "5_exp2" : config_exp_2_5,
         "6_exp2" : config_exp_2_6,
          }  
