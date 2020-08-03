from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Support.transformations import sphere2cart, cart2sph
from sklearn.model_selection import train_test_split
import copy
class PreprocessorBase:
    input_train = None
    input_test = None
    output_train = None
    output_test = None
    percentTest = 0.1

    def __init__(self):
        self.r_scaler = None
        self.theta_scaler = None  
        self.phi_scaler = None  

        self.a_r_scaler = None 
        self.a_theta_scaler = None 
        self.a_phi_scaler = None 

        return
        
    def split(self, position, acceleration):
        if not self.percentTest == 0:
            self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(position, acceleration, test_size=self.percentTest, random_state=13)
            return self.input_train, self.input_test, self.output_train, self.output_test
        else:
            self.input_train, self.input_test, self.output_train, self.output_test = position, np.array([]), acceleration, np.array([])
            return self.input_train, self.output_train
       
    def format_data(self, data):
        if data is None:
            exit("Data Error: Need to specify training data before pre-processing!")
        data_copy = copy.deepcopy(data)
        data_copy = data_copy.reshape((len(data_copy), 1, 3))
        return data_copy

    def unformat_data(self, data):
        if data is None:
            exit("Data Error: Need to specify training data before pre-processing!")
        data_copy = copy.deepcopy(data)
        data_copy = data.reshape((len(data_copy), 3))
        return data_copy

    def fit(self):
        #Formats data and checks if data has already been processed
        self.input_train = self.format_data(self.input_train) #Formats data and checks if data has already been processed
        self.input_test = self.format_data(self.input_test)
        self.output_train = self.format_data(self.output_train)
        self.output_test = self.format_data(self.output_test)

        self.r_scaler.fit(self.input_train[:, :, 0])
        self.theta_scaler.fit(self.input_train[:, :, 1])
        self.phi_scaler.fit(self.input_train[:, :, 2])
        self.a_r_scaler.fit(self.output_train[:, :, 0])
        self.a_theta_scaler.fit(self.output_train[:, :, 1])
        self.a_phi_scaler.fit(self.output_train[:, :, 2])

    def apply_transform(self, x=None, y=None):
        if x is None and y is None:
            input_train = self.input_train
            input_test = self.input_test
            output_train = self.output_train
            output_test = self.output_test

            input_train[:, :, 0] = self.r_scaler.transform(input_train[:, :, 0])
            input_train[:, :, 1] = self.theta_scaler.transform(input_train[:, :, 1])
            input_train[:, :, 2] = self.phi_scaler.transform(input_train[:, :, 2])
            output_train[:, :, 0] = self.a_r_scaler.transform(output_train[:, :, 0])
            output_train[:, :, 1] = self.a_theta_scaler.transform(output_train[:, :, 1])
            output_train[:, :, 2] = self.a_phi_scaler.transform(output_train[:, :, 2])
            self.input_train = input_train.reshape(len(input_train), -1)
            self.output_train = output_train.reshape(len(output_train), -1)
            
            # If no test data was supplied, don't return test data with the output
            if not input_test.size == 0:
                input_test[:, :, 0] = self.r_scaler.transform(input_test[:, :, 0])
                input_test[:, :, 1] = self.theta_scaler.transform(input_test[:, :, 1])
                input_test[:, :, 2] = self.phi_scaler.transform(input_test[:, :, 2])
                output_test[:, :, 0] = self.a_r_scaler.transform(output_test[:, :, 0])
                output_test[:, :, 1] = self.a_theta_scaler.transform(output_test[:, :, 1])
                output_test[:, :, 2] = self.a_phi_scaler.transform(output_test[:, :, 2])
                self.input_test = input_test.reshape(len(input_test), -1)
                self.output_test = output_test.reshape(len(output_test), -1)
                return self.input_train, self.input_test, self.output_train, self.output_test
            else:
                return  self.input_train, self.output_train

        else:
            x_new = None
            y_new = None
            if x is not None:
                x_new = self.format_data(x)
                #If the radius of the data is different at the level of machine precision, round the values to be the same. 
                if abs(x_new[:,:,0].max() - x_new[:,:,0].min()) < 1E-8:
                    x_new[:,:,0] = np.round(x_new[:,:,0], 4)
                x_new[:, :, 0] = self.r_scaler.transform(x_new[:, :, 0])
                x_new[:, :, 1] = self.theta_scaler.transform(x_new[:, :, 1])
                x_new[:, :, 2] = self.phi_scaler.transform(x_new[:, :, 2])
                x_new = self.unformat_data(x_new)
            
            if y is not None:
                y_new = self.format_data(y)
                y_new[:, :, 0] = self.a_r_scaler.transform(y_new[:, :, 0])
                y_new[:, :, 1] = self.a_theta_scaler.transform(y_new[:, :, 1])
                y_new[:, :, 2] = self.a_phi_scaler.transform(y_new[:, :, 2])
                y_new = self.unformat_data(y_new)

            return x_new, y_new
            

        
    def invert_transform(self, x=None, y=None):
        if x is None and y is None:
            input_train = self.input_train
            input_test = self.input_test
            output_train = self.output_train
            output_test = self.output_test

            input_train[:, :, 0] = self.r_scaler.inverse_transform(input_train[:, :, 0])
            input_train[:, :, 1] = self.theta_scaler.inverse_transform(input_train[:, :, 1])
            input_train[:, :, 2] = self.phi_scaler.inverse_transform(input_train[:, :, 2])
            output_train[:, :, 0] = self.a_r_scaler.inverse_transform(output_train[:, :, 0])
            output_train[:, :, 1] = self.a_theta_scaler.inverse_transform(output_train[:, :, 1])
            output_train[:, :, 2] = self.a_phi_scaler.inverse_transform(output_train[:, :, 2])

            input_test[:, :, 0] = self.r_scaler.inverse_transform(input_test[:, :, 0])
            input_test[:, :, 1] = self.theta_scaler.inverse_transform(input_test[:, :, 1])
            input_test[:, :, 2] = self.phi_scaler.inverse_transform(input_test[:, :, 2])
            output_test[:, :, 0] = self.a_r_scaler.inverse_transform(output_test[:, :, 0])
            output_test[:, :, 1] = self.a_theta_scaler.inverse_transform(output_test[:, :, 1])
            output_test[:, :, 2] = self.a_phi_scaler.inverse_transform(output_test[:, :, 2])

            self.input_train = input_train.reshape(len(input_train), -1)
            self.input_test = input_test.reshape(len(input_test), -1)

            self.output_train = output_train.reshape(len(output_train), -1)
            self.output_test = output_test.reshape(len(output_test), -1)

            return self.input_train, self.input_test, self.output_train, self.output_test

        else:
            x_new = None
            y_new = None
            if x is not None:
                x_new = self.format_data(x)
                x_new[:, :, 0] = self.r_scaler.inverse_transform(x_new[:, :, 0])
                x_new[:, :, 1] = self.theta_scaler.inverse_transform(x_new[:, :, 1])
                x_new[:, :, 2] = self.phi_scaler.inverse_transform(x_new[:, :, 2])
                x_new = self.unformat_data(x_new)
            if y is not None:
                y_new = self.format_data(y)
                y_new[:, :, 0] = self.a_r_scaler.inverse_transform(y_new[:, :, 0])
                y_new[:, :, 1] = self.a_theta_scaler.inverse_transform(y_new[:, :, 1])
                y_new[:, :, 2] = self.a_phi_scaler.inverse_transform(y_new[:, :, 2])
                y_new = self.unformat_data(y_new)

            return x_new, y_new
    

