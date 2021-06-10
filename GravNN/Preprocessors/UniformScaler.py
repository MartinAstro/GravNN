import numpy as np

class UniformScaler:
    """Scale the variable by the min and max of the range or by a constant scaler
    """
    def __init__(self,feature_range=(-1,1)):
        self.feature_range = feature_range
        pass
    
    def fit(self, data, scaler=None):
        self.scaler = scaler

        data_max = np.max(data)
        data_min = np.min(data)

        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
   

    def fit_transform(self,data, scaler=None):
        self.scaler = scaler
        data_max = np.max(data)
        data_min = np.min(data)

        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

        if self.scaler is not None:
            X = data*self.scaler
        else:
            X = data*self.scale_ + self.min_
        return X
    
    def transform(self, data):   

        if self.scaler is not None:
            X = data*self.scaler
        else:
            X = data*self.scale_ + self.min_
        return X
    
    
    def inverse_transform(self, data):
        if self.scaler is not None:
            return data/self.scaler
        else:
            return (data - self.min_)/self.scale_