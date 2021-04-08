import numpy as np
import pandas as pd
import h5py
import new_predict_targets as pt
import timing_analysis as ta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy import io
import utils
import importlib
import itertools
import os
import yaml
importlib.reload(utils)
importlib.reload(ta)
importlib.reload(pt)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

np.random.seed(1748)

def get_inputs_to_model(peak_df, co, trial_len, dt, used_inds=None, reference='hand'):
    #removing targets for which we don't have a full window of controller inputs
    peak_df = peak_df.iloc[np.where(peak_df.index.get_level_values('time') < trial_len - cfg['post_target_win_stop'])]
    
    if used_inds is None:
        assert(peak_df.index[-1][0] + 1 == co.shape[0])
        used_inds = range(co.shape[0])
    
    k = 0 # target counter
    win_size = int((cfg['post_target_win_stop'] - cfg['post_target_win_start'])/dt)
    X = np.zeros((peak_df.shape[0], win_size*co.shape[2]))
    for i in used_inds:
        trial_peak_df = peak_df.loc[i]
        target_times = trial_peak_df.index
        for target_time in target_times:
            idx_start = int((target_time + cfg['post_target_win_start'])/dt)
            idx_stop = int((target_time + cfg['post_target_win_stop'])/dt)
            X[k,:] = co[i,idx_start:idx_stop,:].flatten()
            k += 1

    if reference == 'hand':
        y = peak_df[['target_x', 'target_y']].values - peak_df[['x', 'y']].values
    elif reference == 'shoulder':
        y = peak_df[['target_x', 'target_y']].values

    return X, y

if __name__=='__main__':

    output_filename = '../data/peaks/target_prediction_grid_search.csv'
    datasets = ['rockstar', 'raju', 'mack']
    params = ['2OLS24', '2OLS24', '2OLS24']
    win_start = 0
    win_stop = 0.5
    min_height_list = [[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]

    rockstar_dict = {'lfads_params': ['2OLS24'], 
                     'file_root':'rockstar'}
    raju_dict = {'lfads_params': ['2OLS24'], 
                 'file_root':'raju'}
    mack_dict = {'lfads_params': ['2OLS24'], 
                 'file_root':'mack'}

    testing_dict =  {'lfads_params': ['2OLS24'], 
                     'file_root':'rockstar-testing'}

    dataset_dicts = {'testing':testing_dict}

    # dataset_dicts = {'Rockstar': rockstar_dict}
    #                  'Raju': raju_dict,
    #                  'Mack': mack_dict}    

    svr_dict = {'kernel':['linear', 'rbf']}
    rf_dict = {'n_estimators':[50, 100]}
    xgb_dict = {'n_estimators':[50, 100]}
    
    preprocess_dict = {'reference':['hand', 'shoulder']}
    pre_param_dicts = []
    for pre_params_set in itertools.product(*preprocess_dict.values()):
        pre_param_dicts.append({k:p for k,p in zip(preprocess_dict.keys(), pre_params_set)})

    # estimator_dict = {'SVR': (SVR(), svr_dict), 
    #               'Random Forest': (RandomForestRegressor(), rf_dict), 
    #               'Gradient Boosted Trees': (XGBRegressor(), xgb_dict)}

    estimator_dict = {'Random Forest': (RandomForestRegressor(), rf_dict), 
                    'Gradient Boosted Trees': (XGBRegressor(), xgb_dict)}

    estimator_dict = {'Random Forest': (RandomForestRegressor(), rf_dict)}

    # estimator_hyperparams = list(set([k for v in estimator_dict.values() for k in v[1].keys()]))
    # preprocess_hyperparams = list(preprocess_dict.keys())

    # hyperparams = preprocess_hyperparams + estimator_hyperparams
    # out_df = pd.DataFrame(columns=hyperparams)
    grid_results = []
    for dataset_name, dataset_dict in dataset_dicts.items():
        for lfads_params in dataset_dict['lfads_params']:
            file_root = dataset_dict['file_root']
            data_filename = '../data/intermediate/' + file_root + '.p'
            lfads_filename = '../data/model_output/' + \
                            '_'.join([file_root, lfads_params, 'all.h5'])
            inputInfo_filename = '../data/model_output/' + \
                                '_'.join([file_root, 'inputInfo.mat'])
            peak_filename = '../data/peaks/' + \
                            '_'.join([file_root, lfads_params, 'peaks_train.p'])
            
            df = data_filename = pd.read_pickle(data_filename)
            input_info = io.loadmat(inputInfo_filename)
            with h5py.File(lfads_filename, 'r+') as h5file:
                co = h5file['controller_outputs'][:]
                dt = utils.get_dt(h5file, input_info)
                trial_len = utils.get_trial_len(h5file, input_info)

            #peak_df = ta.get_peak_df(df, co, trial_len, min_heights, dt=0.01, win_start=win_start, win_stop=win_stop)
            peak_df = pd.read_pickle(peak_filename)
            used_inds = list(set(peak_df.index.get_level_values('trial')))
            used_inds = used_inds[:10]

            for pre_param_dict in pre_param_dicts:
                X, y = get_inputs_to_model(peak_df, co, trial_len, dt, 
                                            used_inds=used_inds, **pre_param_dict)            
            
                for estimator_name, (estimator, param_grid) in estimator_dict.items():
                    model = GridSearchCV(estimator, param_grid)
                    model.fit(X,y)
                    n_params = len(model.cv_results_['params']) #number of parameters in sklearn Grid Search
                    repeated_dataset_dict = {'dataset':[dataset_name]*n_params}
                    repeated_lfads_param_dict = {'lfads_params':[lfads_params]*n_params}
                    repeated_pre_param_dict = {k:[v]*n_params for k,v in pre_param_dict.items()}
                    grid_results.append(pd.DataFrame({**repeated_dataset_dict, **repeated_lfads_param_dict, 
                                                    **repeated_pre_param_dict, **model.cv_results_}))

    results = pd.concat(grid_results, ignore_index=True)
    results.to_csv(output_filename)
