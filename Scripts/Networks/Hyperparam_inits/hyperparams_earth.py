    df_file = 'Data/Dataframes/hyperparameter_earth_prelim.data'
    directory = 'logs/hyperparameter_earth_prelim/'


    df_file = 'Data/Dataframes/hyperparameter_earth_v1.data'
    directory = 'logs/hyperparameter_earth_v1/'

    hparams = {
        'N_train' : [50000, 100000],
        'epochs' : [30000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [10000],
        'decay_epoch_0' : [20000],
        'decay_rate' : [2.0],
        'learning_rate' : [1E-3, 5E-4],
        'batch_size': [131072],
        'activation' : ['gelu', 'tanh', 'swish'],
        'initializer' : ['glorot_uniform', 'glorot_normal'],
        'network_type' : ['traditional','resnet'],
        'num_units' : [20],
    }


    # Giving longer to train, keeping same amount of data to see if there is a data bias
    df_file = 'Data/Dataframes/hyperparameter_earth_v2.data'
    directory = 'logs/hyperparameter_earth_v2/'
    hparams = {
        'N_train' : [50000, 100000],
        'epochs' : [50000, 75000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [25000, 50000],
        'decay_epoch_0' : [25000],
        'decay_rate' : [2.0],
        'learning_rate' : [1E-3, 5E-4],
        'batch_size': [131072],
        'activation' : ['gelu', 'tanh'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
    }

     #Full Long run, with all the data 
    df_file = 'Data/Dataframes/hyperparameter_earth_v3.data'
    directory = 'logs/hyperparameter_earth_v3/'
    hparams = {
        'N_train' : [950000],
        'epochs' : [200000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [50000, 100000],
        'decay_epoch_0' : [50000],
        'decay_rate' : [2.0],
        'learning_rate' : [1E-3, 5E-3],
        'batch_size': [131072],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
    }

    #Shorter run, higher learning rates, with all the data 
    df_file = 'Data/Dataframes/hyperparameter_earth_v4.data'
    directory = 'logs/hyperparameter_earth_v4/'
    hparams = {
        'N_train' : [950000],
        'epochs' : [300000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [25000, 50000],
        'decay_epoch_0' : [25000],
        'decay_rate' : [2.0],
        'learning_rate' : [2E-2, 1E-2],
        'batch_size': [131072],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
    }

    # Show me the plateau!
    hparams = {
        'N_train' : [950000],
        'epochs' : [300000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [25000],
        'decay_epoch_0' : [300000],
        'decay_rate' : [2.0],
        'learning_rate' : [2E-2],
        'batch_size': [131072],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
    }


    #5,000,000 data --  Plus a little extra epochs for the best performing two 
    df_file = 'Data/Dataframes/hyperparameter_earth_v5.data'
    directory = 'logs/hyperparameter_earth_v5/'
    hparams = {
        'N_dist' : [5000000],
        'N_train' : [4900000],
        'epochs' : [300000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [50000, 100000, 150000, 5000000000], # the last one is to simulate virtually no decay
        'decay_epoch_0' : [50000],
        'decay_rate' : [2.0],
        'learning_rate' : [2E-2],
        'batch_size': [131072],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
        #'init_file' : [2459314.280798611]
    }

    hparams = {
        'N_dist' : [5000000],
        'N_train' : [4900000],
        'epochs' : [50000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [50000], # the last one is to simulate virtually no decay
        'decay_epoch_0' : [-300000],
        'decay_rate' : [2.0],
        'learning_rate' : [2E-2],
        'batch_size': [131072],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
        'init_file' : [2459314.280798611] # 2459314.2311921297
    }

    # Very large learning rate, and large batch size -- hoping that this will take less time to train. 
    hparams = {
        'N_dist' : [5000000],
        'N_train' : [4900000],
        'epochs' : [100000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [15000], # the last one is to simulate virtually no decay
        'decay_epoch_0' : [50000],
        'decay_rate' : [2.0],
        'learning_rate' : [2E-2],
        'batch_size': [131072*2*2*2], # 524,288 is really the largest that it should be # 2,097,152 breaks the GPU
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
        #'init_file' : [2459314.280798611] # 2459314.2311921297
    }


    df_file = 'Data/Dataframes/hyperparameter_earth_v6.data'
    directory = 'logs/hyperparameter_earth_v6/'
    # Trained with a ton of data and very large batch sizes for 100,000 epochs
    # Now I'm going to train with different (lower learning rates) to see what learning rate should we be at after 100,000 epochs
    hparams = {
        'N_dist' : [5000000],
        'N_train' : [4900000],
        'epochs' : [10000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [10000], # the last one is to simulate virtually no decay
        'decay_epoch_0' : [0],#50000],
        'decay_rate' : [2.0],
        'learning_rate' : [0.005, 0.001, 0.0001], #[0.02 (started here), 0.002 (15000 decay rate for 50,000), 0.00002 (5000 decay rate for 50,000)]
        'batch_size': [131072*2*2*2], # 524,288 is really the largest that it should be # 2,097,152 breaks the GPU
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
        'init_file' : [2459315.719537037] # 2459314.2311921297
    }


    # Giving longer to train, keeping same amount of data to see if there is a data bias
    df_file = 'Data/Dataframes/hyperparameter_earth_40_v1.data'
    directory = 'logs/hyperparameter_earth_40_v1/'
    hparams = {
        'N_train' : [100000],
        'epochs' : [100000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [25000, 12500],
        'decay_epoch_0' : [25000],
        'decay_rate' : [2.0],
        'learning_rate' : [5E-3, 1E-3, 5E-4],
        'batch_size': [131072],
        'activation' : ['gelu', 'tanh'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [40],
    }

    # Giving longer to train, keeping same amount of data to see if there is a data bias
    df_file = 'Data/Dataframes/hyperparameter_earth_40_v2.data'
    directory = 'logs/hyperparameter_earth_40_v2/'
    hparams = {
        'N_dist' : [5000000],
        'N_train' : [4900000],
        'epochs' : [100000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [25000],
        'decay_epoch_0' : [25000],
        'decay_rate' : [2.0],
        'learning_rate' : [5E-3],
        'batch_size': [131072*2],
        'activation' : ['gelu', 'tanh'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [40],
    }

    # Giving longer to train, keeping same amount of data to see if there is a data bias
    df_file = 'Data/Dataframes/hyperparameter_earth_80_v1.data'
    directory = 'logs/hyperparameter_earth_80_v1/'
    hparams = {
        'N_dist' : [5000000],
        'N_train' : [4900000],
        'epochs' : [100000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [25000],
        'decay_epoch_0' : [25000],
        'decay_rate' : [2.0],
        'learning_rate' : [5E-3],
        'batch_size': [131072*2],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [80],
    }

    # Take trained network and see if a larger learning rate would have let it continue learning. 
    df_file = 'Data/Dataframes/hyperparameter_earth_10_v1.data'
    directory = 'logs/hyperparameter_earth_10_v1/'
    hparams = {
        'N_dist' : [5000000],
        'N_train' : [4900000],
        'epochs' : [100000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [25000],
        'decay_epoch_0' : [25000],
        'decay_rate' : [2.0],
        'learning_rate' : [5E-3, 1E-3, 5E-4], # 5e-3 (highest), 5e-3*(1/2)^1=0.0025, (5e-3)*(1/2)^(2)=0.00125 (lowest) 
        'batch_size': [131072*2],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [10],
        #'init_file' : [2459318.211111111]
    }
