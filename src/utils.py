import os
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import dill
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        try:
            for col in self.columns:
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders[col] = le
            return self
        except Exception as e:
            raise CustomException(e,sys)

    def transform(self, X):
        try:
            X_copy = X.copy()
            for col, le in self.encoders.items():
                X_copy[col] = le.transform(X_copy[col])
            return X_copy
        except Exception as e:
            raise CustomException(e,sys)

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column
        self.target_encoding = {}

    def fit(self, X, y=None):
        try:
            for category in X[self.target_column].unique():
                self.target_encoding[category] = X[X[self.target_column] == category]['Price'].mean()
            return self
        except Exception as e:
            raise CustomException(e,sys)

    def transform(self, X):
        try:
            X_copy = X.copy()
            X_copy[self.target_column] = X_copy[self.target_column].map(self.target_encoding)
            return X_copy
        except Exception as e:
            raise CustomException(e,sys)
