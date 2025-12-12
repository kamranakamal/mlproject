import os 
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from src.utils import save_object,evaluate_model
from dataclasses import dataclass


@dataclass
class ModelTrainer_config:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainer_config()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input")

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            logging.info("Train test split successfully done.")
            models = {
                "Linear Regression":LinearRegression(),
                "K-Neighbours":KNeighborsRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "Ada-Boost":AdaBoostRegressor(),
                "Gradient Boost":GradientBoostingRegressor()
            }
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
                
        except Exception as e:
            raise CustomException(e,sys)
        finally:
            if best_model_score<0.7:
                raise CustomException("No best model found")
            save_object(self.model_trainer_config.trained_model_file_path,best_model)
            logging.info("Best Model found on both test and training dataset")
            logging.info(f"model score is {model_report[best_model_name]}")
            return best_model_score,best_model
            
