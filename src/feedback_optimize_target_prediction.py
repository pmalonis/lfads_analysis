import numpy as np
import pandas as pd
import h5py
import new_predict_targets as pt
import timing_analysis as ta
import segment_submovements as ss
from multiprocessing import Pool
from optimize_target_prediction import get_inputs_to_model, get_endpoint, x_score_func, y_score_func, r_score_func
import custom_objective
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score,make_scorer
from sklearn.decomposition import PCA
from scipy import io
import utils
import importlib
import itertools
import os
import ray
import yaml
import multiprocessing
from scipy.signal import savgol_filter
from get_model_results import get_model_results
importlib.reload(utils)
importlib.reload(ta)
importlib.reload(pt)
importlib.reload(custom_objective)
importlib.reload(ss)

ray.init()

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

random_state = 1748
np.random.seed(random_state)
spike_dt = 0.001

train_test_random_state = 1027
train_test_ratio = 0.2

get_model_results = ray.remote(get_model_results)

if __name__=='__main__':
    output_filename = '../data/peaks/parallel_correction_search.csv'
    run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
    dataset_dicts = {}
    for dataset in run_info.keys():
        dset_dict = {}
        dset_dict['lfads_params'] = [open(os.path.dirname(__file__)+'/../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read()]
        dset_dict['file_root'] = dataset
        dataset_dicts[run_info[dataset]['name']] = dset_dict



    svr_dict = cfg['svr_parameters']
    random_forest_dict = cfg['svr_parameters']


    preprocess_dict = {'fit_direction':[True, False],
                'reference': ['hand', 'shoulder'],
                'poly_features': [True, False],
                'reduce_time': [True, False],
                'use_rates':[False, True],
                'filter_co': [True, False],
                'time_pcs':[5,10,15,20],
                'min_win_start':[-0.3],
                'max_win_stop':[0.2],
                'win_lim':[(-0.3,0), (-0.3,-0.1),(-0.25,-0.05),
                            (-0.2,0),(-0.15,0.05),(-0.1,0.1),(-0.05,0.15),(0,0.2)],
                'hand_time':[-0.1,-0.05, 0, 0.05, 0.1],
                'rate_pcs':[20]}
                        
    pre_param_dicts = []
    no_pcs_param_dicts = []
    for pre_params_set in itertools.product(*preprocess_dict.values()):
        no_pcs_param_dict = {k:p for k,p in zip(preprocess_dict.keys(), pre_params_set) if k !='rate_pcs'}
        
        #leaves out reduntant paramaters set for rate_pcs parameter only affecting sets with use_rates=True
        if ('use_rates' not in no_pcs_param_dict.keys() or no_pcs_param_dict['use_rates']==False) and no_pcs_param_dict in no_pcs_param_dicts:
            continue
        else:
            pre_param_dicts.append({k:p for k,p in zip(preprocess_dict.keys(), pre_params_set)})
    # estimator_dict = {'SVR': (MultiOutputRegressor(SVR()), svr_dict), 
    #               'Random Forest': (RandomForestRegressor(random_state=random_state), rf_dict), 
    #               'Gradient Boosted Trees': (MultiOutputRegressor(XGBRegressor(random_state=random_state)), xgb_dict)}
    #estimator_dict = {'Random Forest': (RandomForestRegressor(random_state=random_state), rf_dict)}
    estimator_dict = {'SVR': (MultiOutputRegressor(SVR()), svr_dict), 'Random Forest': (RandomForestRegressor(random_state=random_state), rf_dict)}

    inputs = {}
    grid_results = []
    scoring={'x_score':make_scorer(x_score_func),'y_score':make_scorer(y_score_func)}
    dir_scoring={'x_score':make_scorer(x_score_func),'y_score':make_scorer(y_score_func),'r_score':make_scorer(r_score_func)}
    for dataset_name, dataset_dict in dataset_dicts.items():
        for lfads_params in dataset_dict['lfads_params']:
            file_root = dataset_dict['file_root']
            data_filename = os.path.dirname(__file__)+'/../data/intermediate/' + file_root + '.p'
            lfads_filename = os.path.dirname(__file__)+'/../data/model_output/' + \
                            '_'.join([file_root, lfads_params, 'all.h5'])
            inputInfo_filename = os.path.dirname(__file__)+'/../data/model_output/' + \
                                '_'.join([file_root, 'inputInfo.mat'])
            peak_filename = os.path.dirname(__file__)+'/../data/peaks/' + \
                            '_'.join([file_root, 'firstmove_train.p'])
            
            df = pd.read_pickle(data_filename)
            input_info = io.loadmat(inputInfo_filename)
            with h5py.File(lfads_filename, 'r+') as h5file:
                co = h5file['controller_outputs'][:]
                dt = utils.get_dt(h5file, input_info)
                trial_len = utils.get_trial_len(h5file, input_info)

            #peak_df = ta.get_peak_df(df, co, trial_len, min_heights, dt=0.01, win_start=win_start, win_stop=win_stop)
            if os.path.exists(peak_filename):
                peak_df = pd.read_pickle(peak_filename)

            #peak_df = get_endpoint(peak_df, df, dt)
            
            # for pre_param_dict in pre_param_dicts:
            #     if pre_param_dict.get('align_peaks'):
            #         X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, win_start=0.05, win_stop=0.1, **pre_param_dict)            
            #     else:
            #         X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, **pre_param_dict)            
            #     for estimator_name, (estimator, param_grid) in estimator_dict.items():
            #         # if estimator_name == 'SVR' and ('reduce_time' not in pre_param_dict.keys() 
            #         #                                 or pre_param_dict['reduce_time']==False):
            #         #     continue
            #         if isinstance(estimator, MultiOutputRegressor):
            #             param_grid = {'estimator__' + k:v for k,v in param_grid.items()}

            #         if pre_param_dict.get('fit_direction'):
            #             model = GridSearchCV(estimator, param_grid, scoring=dir_scoring, refit=False, cv=cv, n_jobs=n_cores)
            #         else:
            #             model = GridSearchCV(estimator, param_grid, scoring=scoring, refit=False, cv=cv, n_jobs=n_cores)

            #         model.fit(X,y)
            #         n_params = len(model.cv_results_['params']) #number of parameters in sklearn Grid Search
            #         lfads_param_df = pd.DataFrame({'dataset':[dataset_name]*n_params, 'lfads_params':[lfads_params]*n_params})
            #         pre_param_df = pd.DataFrame({k:[v]*n_params for k,v in pre_param_dict.items()})
            #         model.cv_results_.pop('params')
            #         estimator_param_df = pd.DataFrame(model.cv_results_)
            #         estimator_param_df['estimator'] = [estimator_name] * n_params
                    
            #         #removing estimator__ prefix
            #         mapper = lambda s: s.replace('estimator__','')
            #         estimator_param_df.rename(mapper=mapper, axis='columns', inplace=True)
                    
            #         grid_results.append(pd.concat([lfads_param_df, pre_param_df, estimator_param_df], axis=1))
            #         print('param results saved')
            
            args_id = ray.put((peak_df, co, trial_len, dt, df, scoring, dir_scoring,
                                dataset_name, lfads_params, estimator_dict))
            grid_results += [get_model_results.remote(pre_param_dict, args_id) 
                            for pre_param_dict in pre_param_dicts]

    output = pd.concat(ray.get(grid_results), ignore_index=True)
    output['total_test_score'] = output[['mean_test_x_score', 'mean_test_y_score']].mean(1)
    output.to_csv(output_filename)
    
    ray.shutdown()
    # grid_results = []
    # scoring = {'x_score':make_scorer(x_score_func),'y_score':make_scorer(y_score_func)}
    # dir_scoring={'x_score':make_scorer(x_score_func),'y_score':make_scorer(y_score_func),'r_score':make_scorer(r_score_func)}
    # for dataset_name, dataset_dict in dataset_dicts.items():
    #     for lfads_params in dataset_dict['lfads_params']:
    #         file_root = dataset_dict['file_root']
    #         data_filename = os.path.dirname(__file__)+'/../data/intermediate/' + file_root + '.p'
    #         lfads_filename = os.path.dirname(__file__)+'/../data/model_output/' + \
    #                         '_'.join([file_root, lfads_params, 'all.h5'])
    #         inputInfo_filename = os.path.dirname(__file__)+'/../data/model_output/' + \
    #                             '_'.join([file_root, 'inputInfo.mat'])
    #         peak_filename = os.path.dirname(__file__)+'/../data/peaks/' + \
    #                         '_'.join([file_root, 'corrections_train.p'])
            
    #         df = data_filename = pd.read_pickle(data_filename)
    #         input_info = io.loadmat(inputInfo_filename)
    #         with h5py.File(lfads_filename, 'r+') as h5file:
    #             co = h5file['controller_outputs'][:]
    #             dt = utils.get_dt(h5file, input_info)
    #             trial_len = utils.get_trial_len(h5file, input_info)

    #         #peak_df = ta.get_peak_df(df, co, trial_len, min_heights, dt=0.01, win_start=win_start, win_stop=win_stop)
    #         peak_df = pd.read_pickle(peak_filename)
    #         #peak_df = get_endpoint(peak_df, df, dt)
    #         # if os.path.exists(peak_filename):
    #         #     peak_df = pd.read_pickle(peak_filename)
    #         # else:
    #         #     peak_df = pd.read_pickle('../data/peaks/%s_%s_peaks_relative_3sds.p'%(dataset, lfads_params))
    #         #     df_train, df_test = train_test_split(peak_df, test_size=train_test_ratio, random_state=train_test_random_state)
    #         #     df_train, df_test = (df_train.sort_index(), df_test.sort_index())
    #         #     df_train.to_pickle('../data/peaks/%s_%s_fb_peaks_train.p'%(dataset, lfads_params))
    #         #     df_test.to_pickle('../data/peaks/%s_%s_fb_peaks_test.p'%(dataset, lfads_params))
    #         # def fit_model(pre_param_dict):
    #         #     for estimator_name, (estimator, param_grid) in estimator_dict.items():
    #         #         if isinstance(estimator, MultiOutputRegressor):
    #         #             param_grid = {'estimator__' + k:v for k,v in param_grid.items()}

    #         #         if pre_param_dict.get('fit_direction'):
    #         #             model = GridSearchCV(estimator, param_grid, scoring=dir_scoring, refit=False, cv=cv)
    #         #         else:
    #         #             model = GridSearchCV(estimator, param_grid, scoring=scoring, refit=False, cv=cv)

    #         #         model.fit(X,y)
    #         #         n_params = len(model.cv_results_['params']) #number of parameters in sklearn Grid Search
    #         #         lfads_param_df = pd.DataFrame({'dataset':[dataset_name]*n_params, 'lfads_params':[lfads_params]*n_params})
    #         #         pre_param_df = pd.DataFrame({k:[v]*n_params for k,v in pre_param_dict.items()})
    #         #         model.cv_results_.pop('params')
    #         #         estimator_param_df = pd.DataFrame(model.cv_results_)
    #         #         estimator_param_df['estimator'] = [estimator_name] * n_params
                    
    #         #         #removing estimator__ prefix
    #         #         mapper = lambda s: s.replace('estimator__','')
    #         #         estimator_param_df.rename(mapper=mapper, axis='columns', inplace=True)
                    
    #         #         return pd.concat([lfads_param_df, pre_param_df, estimator_param_df], axis=1)
            
    #         # with Pool(n_cores) as job_pool:
    #         #     grid_results = job_pool.map(fit_model,pre_param_dicts)

    #         for pre_param_dict in pre_param_dicts:
    #             if pre_param_dict.get('align_peaks'):
    #                 X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, **pre_param_dict)            
    #             else:
    #                 X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, **pre_param_dict)
    #             for estimator_name, (estimator, param_grid) in estimator_dict.items():
    #                 if estimator_name == 'SVR' and ('reduce_time' not in pre_param_dict.keys() 
    #                                                 or pre_param_dict['reduce_time']==False):
    #                     continue
    #                 if isinstance(estimator, MultiOutputRegressor):
    #                     param_grid = {'estimator__' + k:v for k,v in param_grid.items()}

    #                 if pre_param_dict.get('fit_direction'):
    #                     model = GridSearchCV(estimator, param_grid, scoring=dir_scoring, refit=False, cv=cv, n_jobs=n_cores)
    #                 else:
    #                     model = GridSearchCV(estimator, param_grid, scoring=scoring, refit=False, cv=cv, n_jobs=n_cores)

    #                 model.fit(X,y)
    #                 n_params = len(model.cv_results_['params']) #number of parameters in sklearn Grid Search
    #                 lfads_param_df = pd.DataFrame({'dataset':[dataset_name]*n_params, 'lfads_params':[lfads_params]*n_params})
    #                 pre_param_df = pd.DataFrame({k:[v]*n_params for k,v in pre_param_dict.items()})
    #                 model.cv_results_.pop('params')
    #                 estimator_param_df = pd.DataFrame(model.cv_results_)
    #                 estimator_param_df['estimator'] = [estimator_name] * n_params
                    
    #                 #removing estimator__ prefix
    #                 mapper = lambda s: s.replace('estimator__','')
    #                 estimator_param_df.rename(mapper=mapper, axis='columns', inplace=True)

    #                 grid_results.append(pd.concat([lfads_param_df, pre_param_df, 
    #                                                 estimator_param_df], axis=1))
    #                 print('param results')

    # output = pd.concat(grid_results, ignore_index=True)
    # output.to_csv(output_filename)