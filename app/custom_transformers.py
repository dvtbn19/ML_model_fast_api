import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class YearExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='timeOpen'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        X['year'] = X[self.datetime_col].dt.year
        return X


class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to standardize all numeric columns in the DataFrame
    using sklearn's StandardScaler.
    Automatically drops the 'timeOpen' column (if present) before scaling.
    """

    def __init__(self, time_col='timeOpen'):
        self.time_col = time_col
        self.scaler = StandardScaler()
        self.numeric_cols_ = None

    def fit(self, X, y=None):
        X = X.copy()
        X = X.drop(columns=[self.time_col], errors='ignore')
        self.numeric_cols_ = X.select_dtypes(include='number').columns
        self.scaler.fit(X[self.numeric_cols_])
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=[self.time_col], errors='ignore')
        X_scaled = X.copy()
        X_scaled[self.numeric_cols_] = self.scaler.transform(X[self.numeric_cols_])
        return X_scaled
