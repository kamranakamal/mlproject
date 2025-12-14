import os 
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from src.utils import save_object,evaluate_model
from dataclasses import dataclass
import xgboost as xgb
from catboost import CatBoostRegressor
from src.utils import hyperparameter_tuning


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
                "random_forest":RandomForestRegressor(),
                "xgboost":xgb.XGBRegressor(),
                "ridge":Ridge(),
                "lasso":Lasso(),
                "elasticnet":ElasticNet()
            }
            param_grid = {
    "Linear Regression": {
        "fit_intercept": [True, False],
        "copy_X": [True, False]
    },
    
    "ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],   # regularization strength
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
    },
    
    "lasso": {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0], # regularization strength
        "fit_intercept": [True, False],
        "max_iter": [1000, 5000, 10000],
        "selection": ["cyclic", "random"]
    },
    
    "elasticnet": {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0], # overall regularization strength
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],    # balance between L1 (lasso) and L2 (ridge)
        "fit_intercept": [True, False],
        "max_iter": [1000, 5000, 10000],
        "selection": ["cyclic", "random"]
    },
    
    "random_forest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False]
    },
    
    "xgboost": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0.5, 1.0, 2.0]
    }
}
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            top_model_score = list(sorted(model_report.values()))[-3:]
            top_models = {}
            for score in top_model_score:
                model_name = list(model_report.keys())[list(model_report.values()).index(score)]
                top_models[model_name]=models[model_name]

            best_name, best_params, best_score_cv = hyperparameter_tuning(top_models,param_grid,X_train,y_train)

            if best_name is None:
                raise CustomException("No best model found during hyperparameter tuning")

            best_model = models[best_name]
            if best_params:
                best_model.set_params(**best_params)
            # fit the tuned model
            best_model.fit(X_train, y_train)
            # evaluate on held-out test
            y_test_pred = best_model.predict(X_test)
            best_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"Best model after tuning: {best_name} | CV score {best_score_cv} | Test r2 {best_model_score}")

            if best_model_score < 0.7:
                raise CustomException("No best model found")

            save_object(self.model_trainer_config.trained_model_file_path,best_model)
            logging.info("Best Model saved")
            return best_model_score,best_model

        except Exception as e:
            raise CustomException(e,sys)
            
