import ray
import numpy as np
import pandas as pd
import h5py
import new_predict_targets as pt
import timing_analysis as ta
import segment_submovements as ss
import custom_objective
from ast import literal_eval
from multiprocessing import Pool
import joblib
from itertools import product
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score,make_scorer
from sklearn.decomposition import PCA
from dPCA.dPCA import dPCA
from scipy import io
import utils
import importlib
import itertools
import os
import yaml
import multiprocessing
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
import sys
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
nbins = 12
cv = cfg['target_prediction_cv_splits']

kfold = KFold(n_splits=cv, random_state=train_test_random_state, shuffle=True)

def get_model_results(pre_param_dict, args):
    peak_df, co, trial_len, dt, df, scoring, dir_scoring, dataset_name, lfads_params, estimator_dict, k1, k2 = args
    if pre_param_dict.get('align_peaks'):
        X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, win_start=0.05, win_stop=0.1, k1=k1, k2=k2, **pre_param_dict)            
    else:
        X, y = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, k1=k1, k2=k2, **pre_param_dict)


    results = []
    for estimator_name, (estimator, param_grid) in estimator_dict.items():
        # if estimator_name == 'SVR' and ('reduce_time' not in pre_param_dict.keys() 
        #                                 or pre_param_dict['reduce_time']==False):
        #     continue
        if isinstance(estimator, MultiOutputRegressor):
            param_grid = {'estimator__' + k:v for k,v in param_grid.items()}

        if pre_param_dict.get('fit_direction') and not pre_param_dict.get('corr')=='ja':
            model = GridSearchCV(estimator, param_grid, scoring=dir_scoring, 
                                refit=False, cv=kfold)
        else:
            model = GridSearchCV(estimator, param_grid, scoring=scoring, 
                                refit=False, cv=kfold)

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

        results.append(pd.concat([lfads_param_df, pre_param_df, estimator_param_df], axis=1))
    
    return pd.concat(results, ignore_index=True)

def get_inputs_to_model(peak_df, co, trial_len, dt, df, win_start=0.05, win_stop=0.3, reference='hand', use_rates=False, 
                        reduce_neurons=True, rate_pcs=2, reduce_time=False, time_pcs=10, peaks_only=False, use_dpca=False,
                        align_peaks=False, find_peak_win_size=0.2, reach_endpoint=False, fit_direction=True,
                        poly_features=False, poly_degree=2, filter_co=False, align_win_start=0.0, align_win_stop=0.2,
                        hand_time=0, min_win_start=None, max_win_stop=None, win_lim=None, corr='ca', k1=None, k2=None):
    '''
    fit_direction: fit raw x-y coordinates or fit normalized x-y coordinates as well as
    length vector
    hand_time: time from event to use as reference for hand-centric coordinates
    '''
    #removing targets for which we don't have a full window of controller inputs
    if win_lim is not None:
        win_start, win_stop = win_lim
    
    if min_win_start is None:
        min_win_start = win_start

    if max_win_stop is None:
        max_win_stop = win_stop

    #removing events with a window that goes beyond the trial length
    peak_df = peak_df.iloc[np.where(peak_df.index.get_level_values('time') < trial_len - max_win_stop)]
    
    #removing events with a window that starts before the start of the trial
    peak_df = peak_df.iloc[np.where(peak_df.index.get_level_values('time') + min_win_start >= 0)]
    
    used_inds = np.sort(list(set(peak_df.index.get_level_values('trial'))))

    k = 0 # target counter
    if use_rates and not isinstance(df, pd.DataFrame):
        raise ValueError('df argument must be given if use_rates is True')

    win_size = int(np.round((win_stop-win_start)/dt))
    
    hand_pos = np.zeros((peak_df.shape[0],2))
    
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

        if use_dpca:
            pca = FunctionTransformer()
            X = np.zeros((peak_df.shape[0], win_size*nneurons))
        else:
            if reduce_neurons:
                pca = PCA(n_components=rate_pcs)
                pca.fit(np.vstack(all_smoothed))
                X = np.zeros((peak_df.shape[0], win_size*rate_pcs))
            else:
                pca = FunctionTransformer()
                X = np.zeros((peak_df.shape[0], win_size*nneurons))
    else:
        X = np.zeros((peak_df.shape[0], win_size*co.shape[2]))

    if filter_co:
        co = savgol_filter(co, 11, 2, axis=1)

    idx_to_remove = []
    for i in range(len(used_inds)):
        trial_peak_df = peak_df.loc[used_inds[i]]
        trial_df = df.loc[used_inds[i]]
        transition_times = trial_peak_df.index
        for transition_time in transition_times:
            if align_peaks:
                peak_win_start = int((transition_time + align_win_start)/dt)
                peak_win_stop = int((transition_time + align_win_stop)/dt)
                if use_rates:
                    peak_idx = peak_win_start + np.argmax(all_smoothed[i, peak_win_start:peak_win_stop,:].sum(1)) #peak of population response
                else:
                    peak_idx = peak_win_start + np.argmax(np.abs(co[used_inds[i], peak_win_start:peak_win_stop,:]).sum(1)) # peak of sum of absolute controller input

                idx_start = peak_idx + int(win_start/dt)
                idx_stop = peak_idx + int(win_stop/dt)
            else:
                idx_start = int((transition_time + win_start)/dt)
                idx_stop = int((transition_time + win_stop)/dt)
            
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

            df_idx = trial_df.index.get_loc(transition_time + hand_time, method='nearest')
            hand_pos[k,:] = trial_df.kinematic.iloc[df_idx][['x','y']].values
            k += 1
    
    X = np.delete(X, idx_to_remove, axis=0)

    X_index = ~np.all(np.isnan(X), axis=1)
    X = X[X_index]

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
    
    target_pos = np.delete(target_pos, idx_to_remove, axis=0)
    target_pos = target_pos[X_index]

    hand_pos = np.delete(hand_pos, idx_to_remove, axis=0)
    hand_pos = hand_pos[X_index]

    not_zero_rows = ~np.all((target_pos-hand_pos)==0, axis=1)
    
    if corr == 'sc':
        hand_pos = cartesian_to_shoulder(hand_pos, hand_pos)
        target_pos = cartesian_to_shoulder(target_pos, hand_pos)
    elif corr == 'ja':
        hand_pos = cartesian_to_joint_pos(hand_pos, k1, k2)
        target_pos = cartesian_to_joint_pos(target_pos, k1, k2)

    if reference == 'hand':
        y = target_pos - hand_pos
    elif reference == 'shoulder':
        y = target_pos
    
    #removing examples where hand_pos and target pos are the same. 
    #this occurs if transition_time + hand_time falls exactly on target appearance time

    y = y[not_zero_rows]
    X = X[not_zero_rows]

    if fit_direction:
        #y = np.arctan2(y[:,1], y[:,0])
        r = np.linalg.norm(y, axis=1)
        y = (y.T / np.linalg.norm(y, axis=1)).T
        y = np.concatenate([y, r.reshape(-1,1)], axis=1)

    if use_rates and use_dpca:
        X_av = np.zeros((nneurons, trial_len//dt, nbins))
        reshaped_X = X.reshape((-1, trial_len//dt, nneurons), order='F')
        for i in range(-nbins//2, nbins//2):
            min_theta = i * bin_theta
            max_theta = (i+1) * bin_theta
            X_av[:,:,i] = reshaped_X[(theta > min_theta) & (theta <= max_theta)].mean(0).T
        
        model = dPCA('ntp', n_components=rate_pcs)
        model.fit(X_av)
        transformed = model.transform(reshaped_X.T)['p']

    return X, y

def cartesian_to_joint_pos(pos, k1, k2):
    x, y = pos.T
    theta = np.pi - np.arctan(y/x) - np.arccos((x**2 + y**2 + k1**2 - k2**2)/(2*k1*np.sqrt(x**2+y**2)))
    phi = np.arccos((x**2 + y**2 - k1**2 - k2**2)/(2*k1*k2))

    return np.vstack([theta, phi]).T

def cartesian_to_shoulder(pos, hand_pos):
    r = np.linalg.norm(hand_pos, axis=1)
    x, y = hand_pos.T
    n = hand_pos.shape[0]
    R = np.array([[y, -x],
                  [x,  y]])/r
    R = np.moveaxis(R, -1, 0)

    return R.dot(pos.T)[range(n),:,range(n)]

def cartesian_to_joint_vel(pos, vel, k1, k2):
    x, y = pos.T
    n = pos.shape[0]
    theta, phi = cartesian_to_joint_pos(pos, k1, k2).T
    R = np.array([[k1*sin(theta) + k2*sin(theta+phi), k2*sin(theta+phi)],
                  [k1*cos(theta) + k2*cos(theta+phi), k2*cos(theta+phi)]])    
    R = np.linalg.inv(np.moveaxis(R, -1, 0))
        
    return R.dot(vel.T)[range(n),:,range(n)]

def x_score_func(y, y_pred):
    return r2_score(y[:,0], y_pred[:,0])

def y_score_func(y, y_pred):
    return r2_score(y[:,1], y_pred[:,1])

def r_score_func(y, y_pred):
    return r2_score(y[:,2], y_pred[:,2])

def var_weighted_score_func(y, y_pred):
    return r2_score(y, y_pred, multioutput='variance_weighted')

def mean_cosine_score_func(y, y_pred):
    return np.mean(np.diag(cosine_similarity(y[:,:2], y_pred[:,:2])))
    
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
            t_end = t_move[np.where(t_move > t_target + cfg['post_target_win_stop'])[0][0]]
            idx_end = trial_df.index.get_loc(t_end, method='nearest')
            endpoint_x[k] = trial_df.kinematic.iloc[idx_end]['x']
            endpoint_y[k] = trial_df.kinematic.iloc[idx_end]['y']
            k += 1

    peak_df['endpoint_x'] = endpoint_x
    peak_df['endpoint_y'] = endpoint_y
    
    return peak_df

if __name__=='__main__':
    event_type = snakemake.wildcards.event_type #reference event time to load
    output_filename = snakemake.output[0]

    ray.init(num_cpus=cfg['num_cpus'])
    get_model_results = ray.remote(get_model_results)
    buffer_size = multiprocessing.cpu_count()

    run_info = yaml.safe_load(open(os.path.dirname(__file__) + '/../lfads_file_locations.yml', 'r'))
    dataset_dicts = {}
    for dataset in run_info.keys():
        if 'raju' not in dataset: #only running on raju
            continue

        dset_dict = {}
        dset_dict['lfads_params'] = ["kl-co-dim-search-6p5E8J"] #[open(os.path.dirname(__file__)+'/../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read().strip()]
        dset_dict['file_root'] = dataset
        raw_data = io.loadmat(os.path.dirname(__file__) + '/../data/raw/%s.mat'%dataset)
        if 'monkey' in raw_data.keys():
            dset_dict['k1'] = float(raw_data['monkey']['upper_kinarm'])
            dset_dict['k2'] = float(raw_data['monkey']['lower_kinarm'])
        else:
            dset_dict['k1'] = None
            dset_dict['k2'] = None
        del raw_data
        dataset_dicts[run_info[dataset]['name']] = dset_dict

    svr_dict = cfg['svr_parameters']

    random_forest_dict = cfg['svr_parameters']

    preprocess_dict = cfg['target_preprocessing_search']
    if 'win_lim' in preprocess_dict:
        preprocess_dict['win_lim'] = [literal_eval(w) for w in preprocess_dict['win_lim']]
        preprocess_dict['min_win_start'] = [min(w[0] for w in preprocess_dict['win_lim'])]
        preprocess_dict['max_win_stop'] = [max(w[1] for w in preprocess_dict['win_lim'])]

    pre_param_dicts = []
    no_pcs_param_dicts = []
    for pre_params_set in product(*preprocess_dict.values()):
        no_pcs_param_dict = {k:p for k,p in zip(preprocess_dict.keys(), pre_params_set) if k !='rate_pcs'}
        
        #leaves out reduntant paramaters set for rate_pcs parameter only affecting sets with use_rates=True
        if ('use_rates' not in no_pcs_param_dict.keys() or no_pcs_param_dict['use_rates']==False) and no_pcs_param_dict in no_pcs_param_dicts:
            continue
        else:
            pre_param_dicts.append({k:p for k,p in zip(preprocess_dict.keys(), pre_params_set)})

    estimator_dict = {'SVR': (MultiOutputRegressor(SVR()), cfg['svr_parameters']),
                      'Random Forest': (RandomForestRegressor(), cfg['random_forest_parameters'])
                      }

    trained = {}
    inputs = {}
    grid_results = []
    scoring = {'x_score':make_scorer(x_score_func),'y_score':make_scorer(y_score_func), 
                'var_weighted_score':make_scorer(var_weighted_score_func)}
    dir_scoring = {'x_score':make_scorer(x_score_func),
                    'y_score':make_scorer(y_score_func),
                    'r_score':make_scorer(r_score_func), 
                    'cosine_score':make_scorer(mean_cosine_score_func)}
    
    for dataset_name, dataset_dict in dataset_dicts.items():
        for lfads_params in dataset_dict['lfads_params']:
            file_root = dataset_dict['file_root']
            data_filename = os.path.dirname(__file__)+'/../data/intermediate/' + file_root + '.p'
            lfads_filename = os.path.dirname(__file__)+'/../data/model_output/' + \
                            '_'.join([file_root, lfads_params, 'all.h5'])
            inputInfo_filename = os.path.dirname(__file__)+'/../data/model_output/' + \
                                '_'.join([file_root, 'inputInfo.mat'])
            peak_filename = os.path.dirname(__file__)+'/../data/peaks/' + \
                            '_'.join([file_root, '%s_train.p'%event_type])
            
            df = pd.read_pickle(data_filename)
            input_info = io.loadmat(inputInfo_filename)
            with h5py.File(lfads_filename, 'r+') as h5file:
                co = h5file['controller_outputs'][:]
                dt = utils.get_dt(h5file, input_info)
                trial_len = utils.get_trial_len(h5file, input_info)

            peak_df = pd.read_pickle(peak_filename)
            k1 = dataset_dict['k1']
            k2 = dataset_dict['k2']
            args_id = ray.put((peak_df, co, trial_len, dt, df, scoring, dir_scoring,
                                dataset_name, lfads_params, estimator_dict, k1, k2))
            
            if dataset_dict['k1'] is None and 'ja' in preprocess_dict['corr']: #removing joint angle if we don't have arm measurements
                used_pre_param_dicts = [p for p in pre_param_dicts if p['corr'] != 'ja']
            else:
                used_pre_param_dicts = pre_param_dicts

            grid_results += [get_model_results.remote(pre_param_dict, args_id) 
                            for pre_param_dict in used_pre_param_dicts]

    output = pd.concat(ray.get(grid_results), ignore_index=True)
    output['total_test_score'] = output[['mean_test_x_score', 'mean_test_y_score']].mean(1)
    output.to_csv(output_filename)
    ray.shutdown()