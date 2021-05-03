import numpy as np
import pandas as pd
import h5py
import new_predict_targets as pt
import timing_analysis as ta
import segment_submovements as ss
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score,make_scorer
from sklearn.decomposition import PCA
from scipy import io
import utils
import importlib
import itertools
import os
import yaml
import multiprocessing
importlib.reload(utils)
importlib.reload(ta)
importlib.reload(pt)


config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

random_state = 1748
np.random.seed(random_state)
spike_dt = 0.001

def get_inputs_to_model(peak_df, df, co, trial_len, dt, win_start=0.05, win_stop=0.3, reference='hand', use_rates=False, 
                        rate_pcs=2, reduce_time=False, time_pcs=10, peaks_only=False,
                        align_peaks=False, reach_endpoint=False, fit_direction=False):
    #removing targets for which we don't have a full window of controller inputs

    peak_df = peak_df.iloc[np.where(peak_df.index.get_level_values('time') < trial_len - cfg['post_target_win_stop'])]
    used_inds = list(set(peak_df.index.get_level_values('trial')))

    k = 0 # target counter
    nneurons = sum('neural' in c for c in df.columns)
    win_size = int((win_stop-win_start)/dt)

    #TODO Create utils function for this
    if use_rates:
        std = cfg['target_decoding_smoothed_control_std']
        win = int(dt/spike_dt)
        midpoint_idx = int((win-1)/2)
        all_smoothed = np.zeros((len(used_inds), int(trial_len/dt), nneurons)) #holds firing rates for whole experiment (to be used for dimensionality reduction)
        for i in range(len(used_inds)):
            smoothed = df.loc[used_inds[i]].neural.rolling(window=std*4, min_periods=1, win_type='gaussian', center=True).mean(std=std)
            smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
            smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
            all_smoothed[i,:,:] = smoothed

        pca = PCA(n_components=rate_pcs)
        pca.fit(np.vstack(all_smoothed))
        X = np.zeros((peak_df.shape[0], win_size*rate_pcs))
    else:
        X = np.zeros((peak_df.shape[0], win_size*co.shape[2]))
    for i in range(len(used_inds)):
        trial_peak_df = peak_df.loc[used_inds[i]]
        target_times = trial_peak_df.index
        for target_time in target_times:
            if align_peaks:
                peak_win_start = int((target_time + win_start)/dt)
                peak_win_stop = int((target_time + win_stop)/dt)
                if use_rates:
                    peak_idx = np.argmax(all_smoothed[i, peak_win_start:peak_win_stop,:].sum(0)) + peak_win_start#peak of population response
                else:
                    peak_idx = np.argmax(np.abs(co[used_inds[i], peak_win_start:peak_win_stop,:]).sum(0)) + peak_win_start # peak of sum of absolute controller input
                    
                idx_start = peak_idx - int(win_size/2)
                idx_stop = peak_idx + int(win_size/2)
            else:
                idx_start = int((target_time + win_start)/dt)
                idx_stop = int((target_time + win_stop)/dt)
            try:    
                if use_rates:
                    X[k,:] = pca.transform(all_smoothed[i,idx_start:idx_stop,:]).T.flatten()
                else:
                    X[k,:] = co[used_inds[i],idx_start:idx_stop,:].T.flatten()
            except:
                import pdb;pdb.set_trace()
            k += 1

    if reduce_time:
        pca_time = PCA(n_components=time_pcs)
        X = pca_time.fit_transform(X)

    if reach_endpoint:
        target_pos = peak_df[['endpoint_x', 'endpoint_y']].values
    else:
        target_pos = peak_df[['target_x', 'target_y']].values

    if reference == 'hand':
        y = target_pos - peak_df[['x', 'y']].values
    elif reference == 'shoulder':
        y = target_pos

    if fit_direction:
        r = np.linalg.norm(y, axis=1)
        y = (y.T / r).T
        y = np.concatenate([y, r.reshape((-1,1))], axis=1)

    return X, y

def x_score_func(y, y_true):
    return r2_score(y[:,0], y_true[:,0])

def y_score_func(y, y_true):
    return r2_score(y[:,1], y_true[:,1])

def r_score_func(y, y_true):
    return r2_score(y[:,2], y_true[:,2])

def get_endpoint(peak_df, df, dt):
    used_inds = list(set(peak_df.index.get_level_values('trial')))
    endpoint_x = np.zeros(peak_df.shape[0])
    endpoint_y = np.zeros(peak_df.shape[0])
    k = 0
    for i in used_inds:
        trial_df = df.loc[0]
        minima = ss.trial_transitions(df.loc[i])
        t_move = df.loc[i].index[minima]
        for t_target in peak_df.loc[i].index:
            try:
                t_end = t_move[np.where(t_move > t_target + cfg['post_target_win_stop'])[0][0]]
            except:
                import pdb;pdb.set_trace()
            idx_end = trial_df.index.get_loc(t_end, method='nearest')
            endpoint_x[k] = trial_df.kinematic.iloc[idx_end]['x']
            endpoint_y[k] = trial_df.kinematic.iloc[idx_end]['y']
            k += 1

    peak_df['endpoint_x'] = endpoint_x
    peak_df['endpoint_y'] = endpoint_y
    
    return peak_df

if __name__=='__main__':
    cv = 5 #cross validation factor
    n_cores = multiprocessing.cpu_count()
    output_filename = '../data/peaks/target_prediction_grid_search.csv'
    datasets = ['rockstar', 'raju', 'mack']
    params = ['2OLS24', '2OLS24', '2OLS24']
    win_start = 0
    win_stop = 0.5

    rockstar_dict = {'lfads_params': ['final-fixed-2OLS24'], 
                     'file_root':'rockstar'}
    raju_dict = {'lfads_params': ['2OLS24'], 
                 'file_root':'raju'}
    mack_dict = {'lfads_params': ['2OLS24'], 
                 'file_root':'mack'}

    testing_dict =  {'lfads_params': ['2OLS24'], 
                     'file_root':'rockstar-testing'}
    
    
    # dataset_dicts = {'Rockstar': rockstar_dict,
    #                  'Raju': raju_dict,
    #                  'Mack': mack_dict}

    dataset_dicts = {'Rockstar': rockstar_dict}

    #dataset_dicts = {'testing':testing_dict}

    svr_dict = {'kernel':['linear', 'rbf', 'sigmoid'], 'gamma':np.logspace(-4,0,4), 'C':[0.5,1,1.5,2]}
    rf_dict = {'max_features':['auto', 'sqrt', 'log2'], 
               'min_samples_leaf':[1,5,10,15]}
    xgb_dict = {'max_depth':[1, 2, 3, 4, 5, 6], 
                'learning_rate':[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}

    preprocess_dict = {'reference':['hand', 'shoulder'],
                       'use_rates':[True, False],
                       'reduce_time':[True, False],
                       'reach_endpoint':[True, False],
                       'align_peaks':[False]}

    preprocess_dict = {'fit_direction': [True]}

    pre_param_dicts = []
    for pre_params_set in itertools.product(*preprocess_dict.values()):
        pre_param_dicts.append({k:p for k,p in zip(preprocess_dict.keys(), pre_params_set)})

    estimator_dict = {'SVR': (MultiOutputRegressor(SVR()), svr_dict), 
                  'Random Forest': (RandomForestRegressor(random_state=random_state), rf_dict), 
                  'Gradient Boosted Trees': (MultiOutputRegressor(XGBRegressor(random_state=random_state)), xgb_dict)}
    rf_dict = {'max_features':['auto']}
    estimator_dict = {'Random Forest': (RandomForestRegressor(random_state=random_state), rf_dict)}

    # estimator_dict = {'SVR':(SVR(), svr_dict),
    #                 'Random Forest': (RandomForestRegressor(), rf_dict), 
    #                 'Gradient Boosted Trees': (XGBRegressor(), xgb_dict)}

    # estimator_dict = {'Random Forest': (RandomForestRegressor(), rf_dict)}

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
            peak_df = get_endpoint(peak_df, df, dt)
            for pre_param_dict in pre_param_dicts:
                if 'fit_direction' in pre_param_dict and pre_param_dict['fit_direction']==True:
                    scoring={'x_score':make_scorer(x_score_func),
                                'y_score':make_scorer(y_score_func),
                                'r_score':make_scorer(r_score_func)}
                else:
                    scoring={'x_score':make_scorer(x_score_func),'y_score':make_scorer(y_score_func)}

                if 'align_peaks' in pre_param_dict and pre_param_dict['align_peaks']:
                    X, y = get_inputs_to_model(peak_df, df, co, trial_len, dt, win_start=0.05, win_stop=0.1, **pre_param_dict)            
                else:
                    X, y = get_inputs_to_model(peak_df, df, co, trial_len, dt, **pre_param_dict)            
                for estimator_name, (estimator, param_grid) in estimator_dict.items():
                    if isinstance(estimator, MultiOutputRegressor):
                        param_grid = {'estimator__' + k:v for k,v in param_grid.items()}

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

    output = pd.concat(grid_results, ignore_index=True)
    output.to_csv(output_filename)
