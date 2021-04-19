import numpy as np

class UniformScaler:
    def __init__(self,feature_range=(-1,1)):
        self.feature_range = feature_range
        pass

    def fit_transform(self,data):
        data_max = np.max(data)
        data_min = np.min(data)

        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range


        X = data*self.scale_ + self.min_
        return X
    
    def transform(self, data):   
        return data*self.scale_ + self.min_
    
    def inverse_transform(self, data):
        return (data - self.min_)/self.scale_