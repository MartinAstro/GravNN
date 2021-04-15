

    # df_file = 'Data/Dataframes/hyperparameter_v1.data'
    # configurations = {"Default" : get_default_earth_config() }

    # HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([20, 40, 80]))
    # HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.1))
    # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'rmsprop']))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'bent_identity'])) 
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([2048, 131072]))
    # HP_DATA_SIZE = hp.HParam('N_train', hp.Discrete([500000, 950000]))
    # HP_EPOCHS = hp.HParam('epochs', hp.Discrete([5000, 10000]))

    # df_file = 'Data/Dataframes/hyperparameter_v2.data'
    # directory = 'logs/hparam_tuning/'

    # df_file = 'Data/Dataframes/useless_board.data'
    # directory = 'logs/useless/'

    
    # df_file = 'Data/Dataframes/hyperparameter_v3.data'
    # directory = 'logs/hparam_tuning_v3/'

    # configurations = {"Default" : get_default_earth_config() }
    # HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([20, 40, 80]))
    # HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.1))
    # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'rmsprop']))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'bent_identity'])) 
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8196, 32768, 131072]))

    # HP_DATA_SIZE = hp.HParam('N_train', hp.Discrete([125000, 250000, 500000]))
    # HP_EPOCHS = hp.HParam('epochs', hp.Discrete([2500, 5000, 7500]))


                            

    # df_file = 'Data/Dataframes/hyperparameter_v4.data'
    # directory = 'logs/hparam_tuning_v4/'

    # configurations = {"Default" : get_default_earth_config() }
    # HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([20, 40, 80]))
    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2, 1E-3, 1E-4]))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8196, 32768]))
    # HP_DATA_SIZE = hp.HParam('N_train', hp.Discrete([250000, 500000]))
    # HP_EPOCHS = hp.HParam('epochs', hp.Discrete([1250, 2500]))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu', 'elu', 'relu', 'swish']))

                            
    # df_file = 'Data/Dataframes/hyperparameter_moon_v1.data'
    # directory = 'logs/hyperparameter_moon_v1/'

    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-3, 5E-4, 5E-5]))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([2048, 8196, 32768]))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'gelu', 'relu', 'swish']))
    # HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform', 'glorot_normal']))
    # HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional', 'resnet']))


    # # Just Batch Size
    # df_file = 'Data/Dataframes/hyperparameter_moon_v2.data'
    # directory = 'logs/hyperparameter_moon_v2/'

    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-5]))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([512, 1024, 2048]))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh']))
    # HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_normal']))
    # HP_NETWORK = hp.HParam('network_type', hp.Discrete(['resnet']))

    
    # # More comprehensive Batch Size
    # df_file = 'Data/Dataframes/hyperparameter_moon_v3.data'
    # directory = 'logs/hyperparameter_moon_v3/'


    # configurations = {"Default" : get_default_moon_config() }
    # configurations['Default']['layers'] =  [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
    # configurations['Default']['N_dist'] =  [55000]
    # configurations['Default']['N_train'] =  [50000]
    # configurations['Default']['N_val'] =  [4450]
    # configurations['Default']['radius_max'] = [Moon().radius + 5000]
    # configurations['Default']['epochs'] = [50000]



    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-3, 5E-5]))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([512, 1024, 2048]))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'gelu']))
    # HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_normal', 'glorot_uniform']))
    # HP_NETWORK = hp.HParam('network_type', hp.Discrete(['resnet', 'traditional']))



    # # Small Batch Learning Rate Config
    # df_file = 'Data/Dataframes/hyperparameter_moon_v6.data'
    # directory = 'logs/hyperparameter_moon_v6/'

    # HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([1.0, 5.0]))
    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-3, 1E-4]))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1024, 2048]))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh']))
    # HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    # HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))


    # # Large Batch Learning / Exponential Rate Decay Config
    # df_file = 'Data/Dataframes/hyperparameter_moon_v7.data'
    # directory = 'logs/hyperparameter_moon_v7/'

    # HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0, 5.0, 10.0]))
    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2]))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32768, 131072]))
    # HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh']))
    # HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    # HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))

    # Large Batch Learning / Exponential Rate Decay + Hyperparams
    df_file = 'Data/Dataframes/hyperparameter_moon_v8.data'
    directory = 'logs/hyperparameter_moon_v8/'

    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32768, 131072]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'gelu', 'swish']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform', 'glorot_normal']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))

    # Large Batch Learning / Exponential Rate Decay + Hyperparams
    df_file = 'Data/Dataframes/hyperparameter_moon_v9.data'
    directory = 'logs/hyperparameter_moon_v9/'

    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32768, 131072]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'gelu', 'swish']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform', 'glorot_normal']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))


    # Large Batch Learning / Exponential Rate Decay + longer decay paeriods
    df_file = 'Data/Dataframes/hyperparameter_moon_v10.data'
    directory = 'logs/hyperparameter_moon_v10/'

    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([10000, 15000, 20000]))

    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))



    # Large Batch Learning / Exponential Rate Decay + longer decay paeriods + Longer training times (100000 epochs)
    df_file = 'Data/Dataframes/hyperparameter_moon_v11.data'
    directory = 'logs/hyperparameter_moon_v11/'

    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([10000, 30000, 50000]))

    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))

    # Large Batch Learning / Exponential Rate Decay + longer decay paeriods + Even Longer training times (200000 epochs)
    df_file = 'Data/Dataframes/hyperparameter_moon_v12.data'
    directory = 'logs/hyperparameter_moon_v12/'

    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([30000, 50000]))

    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))


    # 5,000,000 data --- 50,000 radius / large BS / Exponential Decay / Long Training Times
    df_file = 'Data/Dataframes/hyperparameter_moon_v_50000_0.data'
    directory = 'logs/hyperparameter_moon_v_50000_0/'

    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([50000]))
    HP_DECAY_EPOCH_0 = hp.HParam('decay_epoch_0', hp.Discrete([25000]))
    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-2]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072*2]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))

    # 5,000,000 data --- 50,000 radius / large BS / Exponential Decay / Long Training Times / Much longer decay epoch 0
    df_file = 'Data/Dataframes/hyperparameter_moon_v_50000_1.data'
    directory = 'logs/hyperparameter_moon_v_50000_1/'

    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([50000]))
    HP_DECAY_EPOCH_0 = hp.HParam('decay_epoch_0', hp.Discrete([100000]))
    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-2, 1E-2]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072*4]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))

    # 5,000,000 data --- 50,000 radius / large BS / Exponential Decay / Long Training Times / Much longer decay epoch 0 
    df_file = 'Data/Dataframes/hyperparameter_moon_v_50000_1_opt.data'
    directory = 'logs/hyperparameter_moon_v_50000_1_opt/'

    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([100000]))
    HP_DECAY_EPOCH_0 = hp.HParam('decay_epoch_0', hp.Discrete([100000]))
    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-3, 2.5E-3]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072*4]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))


    # Varying Data amounts over 50,000 epochs with ReduceLROnPlateau (0.5, 1000 epoch patience)
    df_file = 'Data/Dataframes/hyperparameter_moon_v_data.data'
    directory = 'logs/hyperparameter_moon_v_data/'

    HP_N_DIST = hp.HParam('N_data', hp.Discrete([4900000, 500000, 1000000, 2000000]))
    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([100000]))
    HP_DECAY_EPOCH_0 = hp.HParam('decay_epoch_0', hp.Discrete([100000]))
    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([5E-3]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072*4]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))



    
    # Varying Data amounts over 50,000 epochs with ReduceLROnPlateau (0.5, 1000 epoch patience)
    df_file = 'Data/Dataframes/hyperparameter_moon_v_40.data'
    directory = 'logs/hyperparameter_moon_v_40/'

    HP_N_DIST = hp.HParam('N_data', hp.Discrete([500000, 750000, 1000000]))
    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([100000]))
    HP_DECAY_EPOCH_0 = hp.HParam('decay_epoch_0', hp.Discrete([100000]))
    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-3, 5E-3]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))


    # Fixed Data 30,000 epochs, different network architectures,  with Exponential Decay 
    df_file = 'Data/Dataframes/hyperparameter_moon_v_41.data'
    directory = 'logs/hyperparameter_moon_v_41/'

    HP_N_DIST = hp.HParam('N_data', hp.Discrete([500000]))
    HP_NETWORK_SHAPE = hp.HParam('network_shape', hp.Discrete(['deep', 'wide', 'normal', 'funnel']))
    HP_DECAY_RATE_EPOCH = hp.HParam('decay_rate_epoch', hp.Discrete([100000]))
    HP_DECAY_EPOCH_0 = hp.HParam('decay_epoch_0', hp.Discrete([20000]))
    HP_DECAY_RATE = hp.HParam('decay_rate', hp.Discrete([2.0]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1E-3]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([131072]))
    HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
    HP_INITIALIZER = hp.HParam('initializer', hp.Discrete(['glorot_uniform']))
    HP_NETWORK = hp.HParam('network_type', hp.Discrete(['traditional']))


     # Maybe preprocessing
    # Maybe weight initialization
    args = []
    session_num = 0
    for network_shape in HP_NETWORK_SHAPE.domain.values:
        for n_dist in HP_N_DIST.domain.values:
            for decay_epoch_0 in HP_DECAY_EPOCH_0.domain.values:
                for decay_rate_epoch in HP_DECAY_RATE_EPOCH.domain.values:
                    for decay_rate in HP_DECAY_RATE.domain.values:
                        for learning_rate in HP_LEARNING_RATE.domain.values:
                            for batch_size in HP_BATCH_SIZE.domain.values:
                                for activation in HP_ACTIVATION.domain.values:
                                    for initializer in HP_INITIALIZER.domain.values:
                                        for network in HP_NETWORK.domain.values:
                                            hparams = {
                                                HP_NETWORK_SHAPE: network_shape,
                                                HP_N_DIST: n_dist,
                                                HP_DECAY_EPOCH_0 : decay_epoch_0,
                                                HP_DECAY_RATE_EPOCH: decay_rate_epoch,
                                                HP_DECAY_RATE : decay_rate,
                                                HP_LEARNING_RATE: learning_rate,
                                                HP_BATCH_SIZE: batch_size,
                                                HP_ACTIVATION: activation,
                                                HP_INITIALIZER : initializer,
                                                HP_NETWORK : network
                                            }
                                            run_name = "run-%d" % session_num
                                            print('--- Starting trial: %s' % run_name)
                                            print({h.name: hparams[h] for h in hparams})
                                            args.append((df_file, directory + run_name, hparams))
                                            session_num += 1

