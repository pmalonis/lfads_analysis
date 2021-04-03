import numpy as np
import pandas as pd
import h5py
import new_predict_targets as pt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def 


if __name__=='__main__':
    datasets = ['rockstar', 'raju', 'mack']
    params = ['8QTVEk', '2OLS24', '2OLS24']
    win_start = 0
    win_stop = 0.5
    min_height_list = [[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]

    rockstar_dict = {'lfads_params': '2OLS24', 
                     'min_heights':[0.3, 0.3],
                     'file_root':'rockstar'}
    raju_dict = {'lfads_params': '2OLS24', 
                 'min_heights':[0.3, 0.3],
                'file_root':'raju'}
    mack_dict = {'lfads_params': '2OLS24', 
                 'min_heights':[0.3, 0.3],
                 'file_root':'mk08011M1m_mack_RTP'}

    svr_dict = {'kernel':['linear', 'rbf']}
    rf_dict = {'n_estimators':[50, 100]}
    xgb_dict = {'n_estimators':[50, 100]}

    estimator_dict = {'SVR': (SVR, svr_dict), 
                  'Random Forest': (RandomForestRegressor, rf_dict), 
                  'Gradient Boosted Trees': (XGBRegressor, xgb_dict)}


    for estimator_name, (estimator, param_grid) in estimator_dict.items():
        results = GridSearchCV(estimator, param_grid)
