from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import Ones

from keras import backend as K


# Base 2 layers
def Deep_128x3_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=128, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Deep_128x3_LReLU(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=128, activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=128))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=128))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model