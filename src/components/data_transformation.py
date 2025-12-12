import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransforamtion:
    def __init__(self):
        self.data_transforamtion_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            numerical_col = ['writing score', 'reading score']
            categorical_col = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course']
            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("OneHotEncoder",OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False)),
                ("scaler",StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_col),
                ("cat_pipeline",cat_pipeline,categorical_col)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Imported train and test data successfully")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj()
            target_col = "math score"
            num_col = ['writing score', 'reading score']
            input_feature_train_df = train_df.drop(target_col,axis=1)
            target_feature_train_df = train_df[target_col]
            input_feature_test_df = test_df.drop(target_col,axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("applying the preprocessor.....")


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Saved preprocessing objects")


            save_object(
                file_path=self.data_transforamtion_config.preprocessor_obj_file_path,obj=preprocessing_obj
            )

            return (train_arr,test_arr,self.data_transforamtion_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)