import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            joblib.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_score = r2_score(y_train,y_train_pred)
            test_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_score
        
    except Exception as e:
        raise CustomException(e,sys)
    finally:
        logging.info(f"The Full Model Report{report}")
        return report


def hyperparameter_tuning(models, param_grid, X_train, y_train):
    try:
        logging.info("Starting hyperparameter tuning")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = param_grid[list(models.keys())[i]]
            grid_search = GridSearchCV(estimator=model, param_grid=param, scoring='r2', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_model_score_tuned = grid_search.best_score_
            logging.info(f"Best parameters for {list(models.keys())[i]}: {best_params} with score {best_model_score_tuned}")
            logging.info("Hyperparameter tuning completed")
            return best_params,best_model_score_tuned
    except Exception as e:
        logging.error("Error during hyperparameter tuning", exc_info=True)
        raise CustomException(e, sys)