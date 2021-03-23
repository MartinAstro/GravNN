    df_file = "Data/Dataframes/basis_test.data"
    df_file = "Data/Dataframes/deeper_networks2.data"
    df_file = "Data/Dataframes/deeper_networks3.data"
    df_file = "Data/Dataframes/deeper_networks4.data" # ResNet -- leaky
    df_file = "Data/Dataframes/deeper_networks5.data" # Traditional Net -- leaky
    df_file = "Data/Dataframes/deeper_networks6.data" # ResNet  -- tanh
    # Everything before was 250000
    df_file = "Data/Dataframes/deeper_networks7.data" # 950000 -- leaky -- resnet -- 8196
    df_file = "Data/Dataframes/deeper_networks8.data" # 950000 -- leaky -- resnet -- 2048 batch size
    df_file = "Data/Dataframes/deeper_networks9.data" # 1950000 -- leaky -- resnet -- 2048 batch size
    df_file = "Data/Dataframes/deeper_networks10.data" # 950000 -- leaky -- resnet -- 512 batch size
    df_file = "Data/Dataframes/deeper_networks11.data" # 950000 -- leaky -- resnet -- 8196 batch size -- closer altitude 

    df_file = "Data/Dataframes/old_test.data" # 950000 -- tanh -- trad -- 40000 batch size -- 100000 epochs 
    df_file = "Data/Dataframes/dropout_test.data" # 950000 -- leaky -- resnet -- 8196 batch size -- dropout

    configurations = {"32_14" : get_default_earth_config(), # 40 equivalent
                        "20_30" : get_default_earth_config(), #40 equivalent
                        "10_30" : get_default_earth_config(), # 20 equivalent
                        "40_30" : get_default_earth_config(), # 80 equivalent
                         "32_45" : get_default_earth_config(), # 80 equivalent
                        }


    #config['PINN_flag'] = [False]
    #config['basis'] = ['spherical']
   
    configurations['32_14']['layers'] = [[3, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 3]]
    configurations['20_30']['layers'] = [[3, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
    configurations['10_30']['layers'] = [[3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3]]
    configurations['40_30']['layers'] =  [[3, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 3]]
    configurations['32_45']['layers'] = [[3, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 3]]

    configurations = {"lower_lr" : get_default_earth_config(), # 40 equivalent
                        #"higher_lr" : get_default_earth_config()
                        }



    df_file = "Data/Dataframes/learning_rates.data" # 950000 -- leaky -- resnet -- 8196 batch size -- dropout
    df_file = "Data/Dataframes/learning_rates2.data" # 950000 -- leaky -- resnet -- 8196 batch size -- dropout
   
    configurations['lower_lr']['layers'] = [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
    configurations['lower_lr']['layers'] = [[3, 4, 8, 16, 32, 64, 128, 256, 3]]

    configurations['lower_lr']['optimizer'] = [tf.keras.optimizers.Adam(learning_rate=0.0005)]
    configurations['lower_lr']['learning_rate'] = [0.0005]
    configurations['higher_lr']['layers'] = [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
    configurations['higher_lr']['optimizer'] = [tf.keras.optimizers.Adam(learning_rate=0.01)]


    df_file = "Data/Dataframes/widening_networks.data" # 950000 -- leaky -- resnet -- 8196 batch size -- dropout

    configurations = {"wide_5" : get_default_earth_config(), # 40 equivalent
                     "wide_6" : get_default_earth_config(),
                     "wide_7" : get_default_earth_config()
                    }

    configurations['wide_5']['layers'] = [[3, 8, 16, 32, 64, 3]]
    configurations['wide_5']['optimizer'] = [tf.keras.optimizers.Adam(learning_rate=0.0005)]
    configurations['wide_5']['learning_rate'] = [0.0005]
    configurations['wide_6']['layers'] = [[3, 8, 16, 32, 64, 128, 3]]
    configurations['wide_6']['learning_rate'] = [0.001]
    configurations['wide_7']['layers'] = [[3, 8, 16, 32, 64, 128, 256, 3]]
    configurations['wide_7']['learning_rate'] = [0.001]


    df_file = "Data/Dataframes/widening_activation.data" # 950000 -- leaky -- resnet -- 8196 batch size -- dropout
    df_file = "Data/Dataframes/widening_activation_long.data" # 950000 -- leaky -- resnet -- 8196 batch size -- dropout

    configurations = {
        "gelu" : get_default_earth_config(), # 40 equivalent
                     "swish" : get_default_earth_config(),
                     "elu" : get_default_earth_config(),
                     "rbf" : get_default_earth_config(),
                     "relu" : get_default_earth_config(),
                     'tanh' : get_default_earth_config(),
                      "gelu_long" : get_default_earth_config(), # 40 equivalent
                     "swish_long" : get_default_earth_config(),
                     "elu_long" : get_default_earth_config(),
                     "rbf_long" : get_default_earth_config(),
                     "relu_long" : get_default_earth_config()
                    }



    configurations['tanh']['layers'] = [[3, 8, 16, 32, 64, 3]]
    configurations['tanh']['activation'] = ['tanh']

    configurations['gelu']['layers'] = [[3, 8, 16, 32, 64, 3]]
    configurations['gelu']['activation'] = ['gelu']

    configurations['swish']['layers'] = [[3, 8, 16, 32, 64, 3]]
    configurations['swish']['activation'] = ['swish']

    configurations['elu']['layers'] = [[3, 8, 16, 32, 64, 3]]
    configurations['elu']['activation'] = ['elu']

    configurations['rbf']['layers'] = [[3, 8, 16, 32, 64, 3]]
    configurations['rbf']['activation'] = [radial_basis_function]

    configurations['relu']['layers'] = [[3, 8, 16, 32, 64, 3]]
    configurations['relu']['activation'] = ['relu']

    configurations['gelu_long']['layers'] = [[3, 8, 16, 32, 64, 128, 3]]
    configurations['gelu_long']['activation'] = ['gelu']

    configurations['swish_long']['layers'] = [[3, 8, 16, 32, 64,128, 3]]
    configurations['swish_long']['activation'] = ['swish']

    configurations['elu_long']['layers'] = [[3, 8, 16, 32, 64,128, 3]]
    configurations['elu_long']['activation'] = ['elu']

    configurations['rbf_long']['layers'] = [[3, 8, 16, 32, 64,128, 3]]
    configurations['rbf_long']['activation'] = [radial_basis_function]

    configurations['relu_long']['layers'] = [[3, 8, 16, 32, 64,128, 3]]
    configurations['relu_long']['activation'] = ['relu']

    # configurations['wide_5']['optimizer'] = [tf.keras.optimizers.Adam(learning_rate=0.0005)]
    # configurations['wide_5']['learning_rate'] = [0.0005]
    # configurations['wide_6']['layers'] = [[3, 8, 16, 32, 64, 128, 3]]
    # configurations['wide_6']['learning_rate'] = [0.001]
    # configurations['wide_7']['layers'] = [[3, 8, 16, 32, 64, 128, 256, 3]]
    # configurations['wide_7']['learning_rate'] = [0.001]





    df_file = "Data/Dataframes/low_lr_large_batch_long_train.data" # 950000 -- leaky -- resnet -- 8196 batch size -- dropout

    configurations = {"default" : get_default_earth_config(), # 40 equivalent
                        "adam" : get_default_earth_config(),
                        "SGD" : get_default_earth_config()
                        }



    configurations['adam']['optimizer'] = [tf.keras.optimizers.Adam(learning_rate=0.0005)]
    configurations['adam']['learning_rate'] = [0.0005]

    configurations['SGD']['optimizer'] = [tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9)]
    configurations['SGD']['learning_rate'] = [0.0005]