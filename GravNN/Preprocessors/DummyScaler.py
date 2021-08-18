class DummyScaler:
    def __init__(self):
        """Placeholder preprocessor that doesn't actually scale the inputs"""
        pass

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x
