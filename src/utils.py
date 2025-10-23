import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)    
    
    
def evaluate_models(X_train,X_test,y_train,y_test,models,params):
    try:
        report = {}
        # models is expected to be a dict mapping names to estimator instances
        for name, model in models.items():
            # params should be a dict mapping model name to param grid
            param = params.get(name, {}) if isinstance(params, dict) else {}
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            # use the best estimator found by GridSearch
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
            # r2_score expects (y_true, y_pred)
            model_score = r2_score(y_test, y_pred)
            report[name] = model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)      
    
def load_object(file_path):
    try:
         
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)