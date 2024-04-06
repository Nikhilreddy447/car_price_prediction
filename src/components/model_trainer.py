import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("Split training and test input data")
            train_array = train_array[~np.isnan(train_array).any(axis=1)]
            test_array = test_array[~np.isnan(test_array).any(axis=1)]
            
            X_train, y_train, X_test, y_test = (
            train_array[:, :-2],    
            train_array[:, -2],
            test_array[:, :-2], 
            test_array[:, -2],     
            )
            X_train = np.concatenate((X_train, train_array[:, -1][:, np.newaxis]), axis=1)
            X_test = np.concatenate((X_test, test_array[:, -1][:, np.newaxis]), axis=1)
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regression": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBossting Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
                
            }
            
            model_report:dict = evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print(best_model_name)
            best_model = models[best_model_name]
            
            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both trainning and test dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_squred = r2_score(y_test,predicted)
            
            return r2_squred

        except Exception as e:
            raise CustomException(e,sys)