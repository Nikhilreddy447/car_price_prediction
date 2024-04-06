import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import CustomLabelEncoder,TargetEncoder,save_object
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            categorical_cols = ['City', 'Body_type', 'Transmission', 'Fuel_type']
            
            preprocessor = Pipeline([
            ('label_encoder', CustomLabelEncoder(columns=categorical_cols)),
            ('target_encoder_Make_Model', TargetEncoder(target_column='Make_Model')),
            ('target_encoder_Color', TargetEncoder(target_column='color')),
            ])
            
            
            logging.info("preprocessing completed")
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)
            
            logging.info("Read train and test data completed")
            logging.info("obtaining preprocesssor object")
            
            preprocesssor_obj = self.get_data_transformer_object()
            
            logging.info("Applying preprocessor object on training and testing dataframe")
            
            input_feature_train_arr = preprocesssor_obj.fit_transform(train_df)
            input_feature_test_arr = preprocesssor_obj.transform(test_df)
            
            train_arr = np.c_[
                input_feature_train_arr
            ]
            test_arr = np.c_[
                input_feature_test_arr
            ]
            
            logging.info("Saved preprocessing object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file,
                obj = preprocesssor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
        except Exception as e:
            raise CustomException(e,sys)