import pandas as pd
from sklearn.preprocessing import LabelEncoder

class AllFeaturesLabelEncoder:

    def __init__(self):
        self._encoders = {}
    
    def fit(self, data: pd.DataFrame):
        for col in data.columns.values:
            self._encoders[col] = LabelEncoder().fit(data.loc[:, col])
        return self
    
    def transform(self, data, omit_invalid_rows = False):
        data = data.copy()
        is_valid = pd.Series([True] * len(data))

        for col, encoder in self._encoders.items():
            if omit_invalid_rows:
                is_valid = is_valid & (data.loc[:, col].isin(encoder.classes_))
                data.loc[is_valid, col] = encoder.transform(data.loc[is_valid, col])
            else:
                data.loc[:, col] = encoder.transform(data.loc[:, col])
        
        return data if not omit_invalid_rows else data.loc[is_valid, :]

    def inverse_transform(self, data):
        data = data.copy()
        for col, encoder in self._encoders.items():
            data.loc[:, col] = encoder.inverse_transform(data.loc[:, col])
        return data

    def is_valid(self, data):
        is_valid = pd.Series([True] * len(data))
        for col, encoder in self._encoders.items():
            is_valid &= data.loc[:, col].isin(encoder.classes_)
        return is_valid