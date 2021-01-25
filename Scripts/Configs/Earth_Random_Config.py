    #df_file = 'Data/Dataframes/temp.data'
    #df_file = 'Data/Dataframes/param_study.data'
    #df_file = "Data/Dataframes/tensorflow_2_results.data"
    #df_file = "Data/Dataframes/tensorflow_2_ablation.data"
    #df_file = 'Data/Dataframes/N_1000000_study.data'
    df_file = 'Data/Dataframes/N_1000000_PINN_study.data'
    #df_file = 'Data/Dataframes/N_1000000_exp_norm_study.data'
    #df_file = 'Data/Dataframes/N_1000000_exp_norm_sph_study.data'

    #df_file = 'Data/Dataframes/N_1000000_bennu_study.data'
    df_file = 'Data/Dataframes/N_1000000_bennu_PINN_study.data'


    inception_layer = [3, 7, 11]
    dense_layer = [10, 10, 10]
    data_config = {
        'planet' : [Earth()],
        'grav_file' : [Earth().sh_hf_file],
        'distribution' : [RandomDist],
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
        'sh_truth' : ['sh_stats_']
    }
    network_config = {
        'network_type' : [TraditionalNet],
        'PINN_flag' : [False],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 1]],
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

    config['PINN_flag'] = [True]
    config['N_dist'] = [1000000] 
    config['N_train'] = [950000] 
    config['N_val'] = [50000] 
    config['batch_size'] = [40000]
    config['epochs'] = [100000]
   
    config_1 = copy.deepcopy(config)
    config_1.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 1]]})

    config_2 = copy.deepcopy(config)
    config_2.update({'layers' : [[3, 60, 60, 60, 60, 60, 60, 60, 60, 1]]})

    config_3 = copy.deepcopy(config)
    config_3.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 1]]})

    config_4 = copy.deepcopy(config)
    config_4.update({'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 1]]})

    config_5 = copy.deepcopy(config)
    config_5.update({'layers' : [[3, 10, 10, 10, 10, 10, 10, 10, 10, 1]]})
   
    configurations = {
         #"original" : config,

         #"1" : config_1,
         #"2" : config_2,
         "3" : config_3,
         "4" : config_4,
         "5" : config_5,

          }  

    for key, config in configurations.items():
        tf.keras.backend.clear_session()

        utils.check_config_combos(config)
        config = utils.format_config_combos(config)
        
        
        # TODO: Trajectories should take keyword arguments so the inputs dont have to be standard, just pass in config.
        trajectory = config['distribution'][0](config['planet'][0], [config['radius_min'][0], config['radius_max'][0]], config['N_dist'][0], **config)#points=1000000)
        x_unscaled, a_unscaled, u_unscaled = get_sh_data(trajectory, config['grav_file'][0],config['max_deg'][0], config['deg_removed'][0])
        