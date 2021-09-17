import numpy as np
import pandas as pd
import h5py
import new_predict_targets as pt
import timing_analysis as ta
import segment_submovements as ss
import custom_objective
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score,make_scorer
from sklearn.decomposition import PCA
from scipy import io
from optimize_target_prediction import get_inputs_to_model, get_endpoint, x_score_func, y_score_func, r_score_func
import utils
import importlib
import itertools
import os
import yaml
import multiprocessing
from scipy.signal import savgol_filter
importlib.reload(utils)
importlib.reload(ta)
importlib.reload(pt)
importlib.reload(custom_objective)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

random_state = 1748
np.random.seed(random_state)

train_test_random_state = 1027
train_test_ratio = cfg['event_split_ratio']
spike_dt = 0.001

if __name__=='__main__':
    cv = 5 #cross validation factor
    n_cores = multiprocessing.cpu_count()
    output_filename = '../data/peaks/spectral_selected_use_rates_correction.csv'
    if not os.path.exists(output_filename):
        run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
        dataset_dicts = {}
        for dataset in run_info.keys():
            dset_dict = {}
            dset_dict['lfads_params'] = [open(os.path.dirname(__file__)+'/../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read()]
            dset_dict['file_root'] = dataset
            dataset_dicts[run_info[dataset]['name']] = dset_dict

        svr_dict = {'kernel':['rbf'], 
                    'gamma':['scale'], 
                    'C':[0.5]}
                    #'C':[0.2, 0.4, 0.5, 0.7, 0.9]}

        rf_dict = {'max_features':['auto', 'sqrt', 'log2'],
                'min_samples_leaf':[1,5,10,15]}
        preprocess_dict = {'fit_direction':[True],
                        'reference': ['hand', 'shoulder'],
                        'poly_features': [True, False],
                            'reduce_time': [True],
                            'filter_co': [True],
                            'time_pcs':[5, 7, 10, 15]}

        preprocess_dict = {'use_rates':[True, False], 'rate_pcs':np.arange(2,20),
                            'fit_direction':[True, False], 
                            'reference':['hand','shoulder'], #'time_pcs':[5, 7, 10, 15],'poly_features': [True],
                            'reduce_time': [False],
                            'use_dpca':[False],
                            'win_start':[-0.2],
                            'win_stop':[0]}
        #preprocess_dict = {'align_peaks':[True, False], 'fit_direction':[True, False]}
        #preprocess_dict = {'poly_features':[True, False], 'reduce_time':[True], 'filter_co':[True], 'time_pcs':[5], 'win_stop':[.16, 0.2]}
        pre_param_dicts = []
        for pre_params_set in itertools.product(*preprocess_dict.values()):
            pre_param_dicts.append({k:p for k,p in zip(preprocess_dict.keys(), pre_params_set)})

        # estimator_dict = {'SVR': (MultiOutputRegressor(SVR()), svr_dict), 
        #               'Random Forest': (RandomForestRegressor(random_state=random_state), rf_dict), 
        #               'Gradient Boosted Trees': (MultiOutputRegressor(XGBRegressor(random_state=random_state)), xgb_dict)}
        #estimator_dict = {'Random Forest': (RandomForestRegressor(random_state=random_state), rf_dict)}
        #poly_svr_pipeline = make_pipeline(MultiOutputRegressor(SVR()))
        #estimator_dict = {'Random Forest': (RandomForestRegressor(random_state=random_state), rf_dict)}

        # estimator_dict = {'SVR':(SVR(), svr_dict),
        #                 'Random Forest': (RandomForestRegressor(), rf_dict), 
        #                 'Gradient Boosted Trees': (XGBRegressor(), xgb_dict)}

        estimator_dict = {'SVR': (MultiOutputRegressor(SVR()), svr_dict),'Random Forest': (RandomForestRegressor(), rf_dict)}
                        #'Random Forest': (RandomForestRegressor(), rf_dict)}

        rf_dict = {}
        estimator_dict = {'Random Forest': (RandomForestRegressor(), rf_dict)}

        # estimator_hyperparams = list(set([k for v in estimator_dict.values() for k in v[1].keys()]))
        # preprocess_hyperparams = list(preprocess_dict.keys())

        # hyperparams = preprocess_hyperparams + estimator_hyperparams
        # out_df = pd.DataFrame(columns=hyperparams)

        trained = {}
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
                                '_'.join([file_root, 'corrections_train.p'])
                
                df = data_filename = pd.read_pickle(data_filename)
                input_info = io.loadmat(inputInfo_filename)
                with h5py.File(lfads_filename, 'r+') as h5file:
                    co = h5file['controller_outputs'][:]
                    dt = utils.get_dt(h5file, input_info)
                    trial_len = utils.get_trial_len(h5file, input_info)

                #peak_df = ta.get_peak_df(df, co, trial_len, min_heights, dt=0.01, win_start=win_start, win_stop=win_stop)
                peak_df = pd.read_pickle(peak_filename)
 
                #peak_df = get_endpoint(peak_df, df, dt)
                for pre_param_dict in pre_param_dicts:
                    if pre_param_dict.get('align_peaks'):
                        X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, win_start=0.05, win_stop=0.1, **pre_param_dict)            
                    else:
                        X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, **pre_param_dict)            
                    for estimator_name, (estimator, param_grid) in estimator_dict.items():
                        if isinstance(estimator, MultiOutputRegressor):
                            param_grid = {'estimator__' + k:v for k,v in param_grid.items()}

                        if pre_param_dict.get('fit_direction'):
                            model = GridSearchCV(estimator, param_grid, scoring=dir_scoring, refit=False, cv=cv, n_jobs=n_cores)
                        else:
                            model = GridSearchCV(estimator, param_grid, scoring=scoring, refit=False, cv=cv, n_jobs=n_cores)

                        model.fit(X,y)
                        n_params = len(model.cv_results_['params']) #number of parameters in sklearn Grid Search
                        lfads_param_df = pd.DataFrame({'dataset':[dataset_name]*n_params, 'lfads_params':[lfads_params]*n_params})
                        pre_param_df = pd.DataFrame({k:[v]*n_params for k,v in pre_param_dict.items()})
                        model.cv_results_.pop('params')
                        estimator_param_df = pd.DataFrame(model.cv_results_)
                        estimator_param_df['estimator'] = [estimator_name] * n_params
                        
                        #removing estimator__ prefix
                        mapper = lambda s: s.replace('estimator__','')
                        estimator_param_df.rename(mapper=mapper, axis='columns', inplace=True)

                        grid_results.append(pd.concat([lfads_param_df, pre_param_df, 
                                                        estimator_param_df], axis=1))
                        inputs[dataset_name] = (X, y)

        output = pd.concat(grid_results, ignore_index=True)
        output.to_csv(output_filename)

    else:
        output = pd.read_csv(output_filename)

    # output['total_test_score'] = output[['mean_test_x_score',
    #                                      'mean_test_y_score']].mean(1)
    # co_output = pd.read_csv('../data/peaks/target_prediction_search_initial.csv')
    # co_output['total_test_score'] = co_output[['mean_test_x_score',
    #                                          'mean_test_y_score']].mean(1)
    # output.groupby(('dataset','rate_pcs'))