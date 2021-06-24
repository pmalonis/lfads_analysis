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
train_test_ratio = 0.2
spike_dt = 0.001


def get_inputs_to_model(peak_df, co, trial_len, dt, win_start=0.05, win_stop=0.3, reference='hand', use_rates=False, 
                        df=None, rate_pcs=2, reduce_time=False, time_pcs=10, peaks_only=False,
                        align_peaks=False, find_peak_win_size=0.2, reach_endpoint=False, fit_direction=True,
                        poly_features=False, poly_degree=2, filter_co=False, align_win_start=0.0, align_win_stop=0.2):
    #removing targets for which we don't have a full window of controller inputs

    peak_df = peak_df.iloc[np.where(peak_df.index.get_level_values('time') < trial_len - cfg['post_target_win_stop'])]
    used_inds = list(set(peak_df.index.get_level_values('trial')))

    k = 0 # target counter
    if use_rates and not isinstance(df, pd.DataFrame):
        raise ValueError('df argument must be given if use_rates is True')

    win_size = int((win_stop-win_start)/dt)

    #TODO Create utils function for this
    if use_rates:
        nneurons = sum('neural' in c for c in df.columns)
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

    idx_to_remove = []
    for i in range(len(used_inds)):
        trial_peak_df = peak_df.loc[used_inds[i]]
        target_times = trial_peak_df.index
        for target_time in target_times:
            if align_peaks:
                peak_win_start = int((target_time + align_win_start)/dt)
                peak_win_stop = int((target_time + align_win_stop)/dt)
                if use_rates:
                    peak_idx = peak_win_start + np.argmax(all_smoothed[i, peak_win_start:peak_win_stop,:].sum(1)) #peak of population response
                else:
                    try:
                        peak_idx = peak_win_start + np.argmax(np.abs(co[used_inds[i], peak_win_start:peak_win_stop,:]).sum(1)) # peak of sum of absolute controller input
                    except:
                        import pdb;pdb.set_trace()

                idx_start = peak_idx + int(win_start/dt)
                idx_stop = peak_idx + int(win_stop/dt)
            else:
                idx_start = int((target_time + win_start)/dt)
                idx_stop = int((target_time + win_stop)/dt)
            
            if (idx_start < 0) or (idx_stop >= trial_len//dt): #for align_peaks condition
                idx_to_remove.append(k)
            elif use_rates:
                if idx_start < 0 or idx_stop >= all_smoothed.shape[1]:
                    X[k,:] = np.nan
                else:
                    X[k,:] = pca.transform(all_smoothed[i,idx_start:idx_stop,:]).T.flatten()
            else:
                if idx_start < 0 or idx_stop >= co.shape[1]:
                    X[k,:] = np.nan
                else:
                    X[k,:] = co[used_inds[i],idx_start:idx_stop,:].T.flatten()

            k += 1

    X = np.delete(X, idx_to_remove, axis=0)

    X_index = ~np.all(np.isnan(X), axis=1)
    X = X[X_index]

    if filter_co:
        for i in range(co.shape[2]):
            X[:,i*win_size:(i+1)*win_size] = savgol_filter(X[:,i*win_size:(i+1)*win_size], 11, 2, axis=1)

    if reduce_time:
        X_reduced = np.zeros((X.shape[0], co.shape[2]*time_pcs))
        for i in range(co.shape[2]):
            pca_time = PCA(n_components=time_pcs)
            X_reduced[:,i*time_pcs:(i+1)*time_pcs] = pca_time.fit_transform(X[:,i*win_size:(i+1)*win_size])
        
        X = X_reduced

    if poly_features:
        poly = PolynomialFeatures(degree=poly_degree)
        X = poly.fit_transform(X)

    if reach_endpoint:
        target_pos = peak_df[['endpoint_x', 'endpoint_y']].values
    else:
        target_pos = peak_df[['target_x', 'target_y']].values

    if reference == 'hand':
        y = target_pos - peak_df[['x', 'y']].values
    elif reference == 'shoulder':
        y = target_pos

    if fit_direction:
        #y = np.arctan2(y[:,1], y[:,0])
        r = np.linalg.norm(y, axis=1)
        y = (y.T / np.linalg.norm(y, axis=1)).T
        y = np.concatenate([y, r.reshape(-1,1)], axis=1)

    y = np.delete(y, idx_to_remove, axis=0)
    y = y[X_index]

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

# def dir_scorer(estimator, X, y):
#     return custom_objective.cos_eval(y, estimator.predict(X))

if __name__=='__main__':
    cv = 5 #cross validation factor
    n_cores = multiprocessing.cpu_count()
    output_filename = '../data/peaks/gini_selected.csv'
    win_start = 0
    win_stop = 0.5
    run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
    dataset_dicts = {}
    for dataset in run_info.keys():
        dset_dict = {}
        dset_dict['lfads_params'] = [open('../data/peaks/%s_selected_param_gini.txt'%(dataset)).read()]
        dset_dict['file_root'] = dataset
        dataset_dicts[run_info[dataset]['name']] = dset_dict

    # rockstar_dict = {'lfads_params':  ['all-early-stop-kl-sweep-yKzIQf'],
    #                  'file_root':'rockstar'}
    # raju_dict = {'lfads_params': ['final-fixed-2OLS24'], 
    #              'file_root':'raju'}
    # nb_raju_dict = {'lfads_params': ['final-fixed-2OLS24'], 
    #              'file_root':'raju_no_bad_trials'}
    # mack_dict = {'lfads_params': ['all-early-stop-kl-sweep-bMGCVf'],
    #                               'file_root':'mack'}
    # raju_M1_dict = {'lfads_params': ['raju-split-updated-params-2OLS24'], 
    #              'file_root':'raju-M1'}
    # raju_PMd_dict = {'lfads_params': ['raju-split-updated-params-2OLS24'], 
    #                 'file_root':'raju-PMd'}

    # testing_dict =  {'lfads_params': ['2OLS24'], 
    #                  'file_root':'rockstar-testing'}
    
    # dataset_dicts = {'Rockstar': rockstar_dict,
    #                  'Raju': raju_dict,
    #                 'Mack': mack_dict}
    #dataset_dicts = {'Raju': raju_dict, 'NB Raju':nb_raju_dict}

    svr_dict = {'kernel':['rbf'], 
                'gamma':['scale'], 
                'C':[0.2, 0.4, 0.5, 0.7, 0.9]}

    rf_dict = {'max_features':['auto', 'sqrt', 'log2'],
               'min_samples_leaf':[1,5,10,15]}
    # xgb_dict = {'max_depth':[1, 2, 3, 4, 5, 6],
    #             'learning_rate':[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}
    
    # preprocess_dict = {'reference':['hand', 'shoulder'],
    #                    'use_rates':[True, False],
    #                    'reduce_time':[True, False],
    #                    'reach_endpoint':[True, False],
    #                    'align_peaks':[False]}

    preprocess_dict = {'fit_direction':[True],
                       'reference': ['hand', 'shoulder'],
                       'poly_features': [True, False],
                        'reduce_time': [True],
                        'filter_co': [True],
                        'time_pcs':[5, 7, 10, 15]}

    #preprocess_dict = {'fit_direction':[True, False]}
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

    estimator_dict = {'SVR': (MultiOutputRegressor(SVR()), svr_dict),}
                      #'Random Forest': (RandomForestRegressor(), rf_dict)}

    #estimator_dict = {'Random Forest': (RandomForestRegressor(), rf_dict)}

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
            data_filename = '../data/intermediate/' + file_root + '.p'
            lfads_filename = '../data/model_output/' + \
                            '_'.join([file_root, lfads_params, 'all.h5'])
            inputInfo_filename = '../data/model_output/' + \
                                '_'.join([file_root, 'inputInfo.mat'])
            peak_filename = '../data/peaks/' + \
                            '_'.join([file_root, 'peaks_train.p'])
            
            df = data_filename = pd.read_pickle(data_filename)
            input_info = io.loadmat(inputInfo_filename)
            with h5py.File(lfads_filename, 'r+') as h5file:
                co = h5file['controller_outputs'][:]
                dt = utils.get_dt(h5file, input_info)
                trial_len = utils.get_trial_len(h5file, input_info)

            #peak_df = ta.get_peak_df(df, co, trial_len, min_heights, dt=0.01, win_start=win_start, win_stop=win_stop)
            if os.path.exists(peak_filename):
                peak_df = pd.read_pickle(peak_filename)
            else:
                peak_df = pd.read_pickle('../data/peaks/%s_%s_peaks_relative_3sds.p'%(dataset, lfads_params))
                df_train, df_test = train_test_split(peak_df, test_size=train_test_ratio, random_state=train_test_random_state)
                df_train, df_test = (df_train.sort_index(), df_test.sort_index())
                df_train.to_pickle('../data/peaks/%s_%s_peaks_train.p'%(dataset, lfads_params))
                df_test.to_pickle('../data/peaks/%s_%s_peaks_test.p'%(dataset, lfads_params))

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