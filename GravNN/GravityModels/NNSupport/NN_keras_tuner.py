import os
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch


def NN_keras_tuner(hp):
    model = keras.Sequential()
    dropout_rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(layers.Dense(
            units=hp.Int('units_'+str(i), min_value=32, max_value=512, step=32),
            activation=hp.Choice("act_"+str(i), values=['relu', 'tanh'])))
        model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(units=3, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                                loss='mean_squared_error',
                                metrics=['mse', 'mae'])
    return model 






   
    if save_location is not None:
        os.makedirs(save_location,exist_ok=True)
        model_json = model.to_json()
        with open(save_location+"model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(save_location + "model.h5")
        with open(save_location + "history.data", 'wb') as f:
            pickle.dump(history.history, f)
    
    return history, model
