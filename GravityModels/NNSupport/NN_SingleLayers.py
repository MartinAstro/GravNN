from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import Ones

from keras import backend as K

###################
# SINGLE-LAYER NN #
###################

def Single_10_tanh(x_train, y_train): # Best performance thus far!
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=10, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_10_tanh_Dropout(x_train, y_train): 
    activityReg = 0.3
    model = Sequential()
    model.add(Dense(units=10, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dropout(0.1))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model


def Single_100_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=100, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_100_tanh_Dropout(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dropout(0.15))
    model.add(Dense(units=100, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dropout(0.05))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_1000_tanh(x_train, y_train): 
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=1000, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

# Base 2 Layers
def Single_32_LReLU(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=32, activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_64_LReLU(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=64, activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_128_LReLU(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=128, activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_256_LReLU(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=256, activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(LeakyReLU(0.3))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

# tanh
def Single_32_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=32, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_64_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=64, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_128_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=128, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

def Single_256_tanh(x_train, y_train):
    activityReg = 0.00
    model = Sequential()
    model.add(Dense(units=256, activation='tanh', activity_regularizer=l2(activityReg), input_shape=(len(x_train[0]),)))
    model.add(Dense(units=len(y_train[0]), activation='linear'))
    return model

