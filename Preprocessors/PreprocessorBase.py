from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Support.transformations import sphere2cart, cart2sph
from sklearn.model_selection import train_test_split
class PreprocessorBase:
    input_train = None
    input_test = None
    output_train = None
    output_test = None
    percentTest = 0.1

    preprocessed = False
    def __init__(self):
        return
        
    def split(self, position, acceleration):
        self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(position, acceleration, test_size=self.percentTest, random_state=13)

        return   self.input_train, self.input_test, self.output_train, self.output_test
    
    def fit_transform(self):
        return


    def format_data(self, data):
        if data is None:
            exit("Data Error: Need to specify training data before pre-processing!")
        data = data.reshape((len(data), 1, 3))
        return data

    def unformat_data(self, data):
        if data is None:
            exit("Data Error: Need to specify training data before pre-processing!")
        data = data.reshape((len(data), 3))
        return data

    def apply_transform(self):
        if self.preprocessed:
            exit("Data is already pre-processed!")
        self.preprocessed = True
        return

    def invert_transform(self):
        '''
        Return data to original scaling
        If a prediction was provided, also scale that to original values
        :param output_pred: Predicted output values to scale
        :return: Scaled predicted output
        '''
        if not self.preprocessed:
            exit("Data Error: The data was never preprocessed, and therefore can't be post-processed!")

        self.preprocessed = False
        return 
    

