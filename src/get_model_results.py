import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from optimize_target_prediction import get_inputs_to_model
import psutil
import os
import yaml

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

cv = cfg['target_prediction_cv_splits']
n_cores = psutil.cpu_count()

def get_model_results(pre_param_dict, args):
  
    peak_df, co, trial_len, dt, df, scoring, dir_scoring, dataset_name, lfads_params, estimator_dict = args
    if pre_param_dict.get('align_peaks'):
        X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, win_start=0.05, win_stop=0.1, **pre_param_dict)            
    else:
        X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, **pre_param_dict)

    for estimator_name, (estimator, param_grid) in estimator_dict.items():
        # if estimator_name == 'SVR' and ('reduce_time' not in pre_param_dict.keys() 
        #                                 or pre_param_dict['reduce_time']==False):
        #     continue
        if isinstance(estimator, MultiOutputRegressor):
            param_grid = {'estimator__' + k:v for k,v in param_grid.items()}

        if pre_param_dict.get('fit_direction'):
            model = GridSearchCV(estimator, param_grid, scoring=dir_scoring, refit=False, cv=cv)
        else:
            model = GridSearchCV(estimator, param_grid, scoring=scoring, refit=False, cv=cv)

        model.fit(X,y)
        n_params = len(model.cv_results_['params']) #number of parameters in sklearn Grid Search
        lfads_param_df = pd.DataFrame({'dataset':[dataset_name]*n_params, 
                                        'lfads_params':[lfads_params]*n_params})
        pre_param_df = pd.DataFrame({k:[v]*n_params for k,v in pre_param_dict.items()})
        model.cv_results_.pop('params')
        estimator_param_df = pd.DataFrame(model.cv_results_)
        estimator_param_df['estimator'] = [estimator_name] * n_params
        
        #removing estimator__ prefix
        mapper = lambda s: s.replace('estimator__','')
        estimator_param_df.rename(mapper=mapper,
                                  axis='columns', inplace=True)
        
        print('param results computed')

        return pd.concat([lfads_param_df, pre_param_df, estimator_param_df], axis=1)