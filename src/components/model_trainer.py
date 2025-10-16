import sys
from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.utils import save_obj,evaluate_models
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info(f"Train test split")
            X_train,y_train,X_test,y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
                                           )
            models={'Linear Regression':LinearRegression(),'Lasso Regression':Lasso(),'Ridge Regression':Ridge(),'Elastic Net':ElasticNet()}
            params={'Linear Regression':{'fit_intercept':[True,False]},'Lasso Regression':{'alpha':[0.1,0.5,1],'fit_intercept':[True,False]},'Ridge Regression':{'alpha':[0.1,0.5,1],'fit_intercept':[True,False]},'Elastic Net':{'alpha':[0.1,0.5,1],'l1_ratio':[0.1,0.5,1],'fit_intercept':[True,False]}}
            model_report:dict=evaluate_models(X_train,X_test,y_train,y_test,models,params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            # fit the selected best model on the training data before predicting
            best_model.fit(X_train, y_train)
            predicted = best_model.predict(X_test)
            model_r2 = r2_score(y_test, predicted)
            # save the trained model
            save_obj(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info(f"Best model ({best_model_name}) saved to {self.model_trainer_config.trained_model_file_path}")
            return model_r2
        except Exception as e:
            raise CustomException(e,sys)       