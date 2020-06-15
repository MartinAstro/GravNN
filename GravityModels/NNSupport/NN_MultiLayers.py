from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import Ones

from keras import backend as K


# Base 10 layers
def Multi_100_10_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=100, activation='tanh', activity_regularizer=l1(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=10, activation='tanh'))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Multi_100_100_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=100, activation='tanh', activity_regularizer=l1(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=100, activation='tanh'))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Multi_100_100_LeakyReLU(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=100, activation=LeakyReLU(0.3), activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=100, activation=LeakyReLU(0.1)))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Multi_1000_100_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=1000, activation='tanh', activity_regularizer=l1(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=100, activation='tanh'))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model


def Multi_1k_1k_tanh(x_train, y_train):
    # Should be underdetermined for degree 10 (5.18%)
    # Should be underdetermined for degree 100 (0.39%)
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=1000, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=1000, activation='tanh'))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Multi_10k_1k_tanh(x_train, y_train):
    # Should be underdetermined for degree 10 (48.82%)
    # Should be underdetermined for degree 100 (0.86%)
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=10000, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=1000, activation='tanh'))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

# Base 2 Layers
def Multi_128_128_LReLU(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=128, activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=128))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Multi_128_128_softplus_LReLU(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=128, activation='softplus',activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=128))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model
