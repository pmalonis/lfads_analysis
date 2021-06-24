import numpy as np
import h5py
import pandas as pd
from sklearn.model_selection import cross_val_score, permutation_test_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
#from xgboost import XGBoostRegressor, XGBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '.')
import utils
import timing_analysis as ta
import segment_submovements as ss
import os
import yaml
from scipy import io
import pickle
from scipy.signal import find_peaks
from importlib import reload
reload(ss)
reload(ta)

random_state = 1027
train_test_ratio = 0.2
non_feedback_window = 0.2
non_corrective_window = 0.35
win_start = -0.25
win_stop = 0

def filter_peaks(peak_df):
    peak_df = peak_df[peak_df[['latency_0', 'latency_1']].notnull().any(axis=1)]
    return peak_df

def peak_speed(trial_df, exclude_pre_target=None, exclude_post_target=None):
    speed = np.linalg.norm(trial_df.kinematic[['x_vel', 'y_vel']].values, axis=1)
    idx_targets = np.where(trial_df.kinematic['hit_target'])[0]
    peaks = []
    for i in range(len(idx_targets)-1):
        target_peaks,_ = find_peaks(speed[idx_targets[i]:idx_targets[i+1]])
        if len(target_peaks) > 0:
            target_peaks += idx_targets[i]
            peaks.append(target_peaks[speed[target_peaks].argmax()])
        else:
            continue

    return np.array(peaks)

def pre_peak(trial_df, exclude_pre_target=None, exclude_post_target=None):
    x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].values.T
    minima = ss.speed_minima(x_vel, y_vel)
    peaks = peak_speed(trial_df)
    peaks = peaks[peaks>minima[0]]
    t = trial_df.index.values
    speed = np.sqrt(x_vel**2 + y_vel**2)
    accel = np.gradient(speed, t)
    
    movements = []
    for i in range(len(peaks)-1):
        move_start = minima[np.argmin(np.ma.MaskedArray(peaks[i] - minima, peaks[i] - minima < 0))]
        move_stop = find_peaks(-speed[peaks[i]:])[0][0]
        start_accel = accel[move_start]
        peak_accel = np.max(accel[move_start:peaks[i]])
        accel_amp = peak_accel-start_accel

        peak_accel_idx = move_start + np.argmax(accel[move_start:peaks[i]])

        up_peaks = move_start + find_peaks(accel[move_start:peak_accel_idx])[0]
        if len(up_peaks) > 0:
            up_troughs = np.zeros(len(up_peaks), dtype=int)
            for j,p in enumerate(up_peaks):
                try:
                    up_troughs[j] = p + find_peaks(-accel[p:peak_accel_idx])[0][0]
                except:
                    import pdb; pdb.set_trace()

            movements += list(up_peaks[(accel[up_peaks] - accel[up_troughs])>(0.1*accel_amp)])

        down_peaks = peak_accel_idx + find_peaks(accel[peak_accel_idx:move_stop])[0]
        if len(down_peaks) > 0:
            down_troughs = np.zeros(len(down_peaks), dtype=int)
            for j,p in enumerate(down_peaks):
                try:
                    down_troughs[j] = p - find_peaks(-accel[p:peak_accel_idx:-1])[0][0]
                except:
                    import pdb; pdb.set_trace()

            movements += list(down_peaks[(accel[down_peaks] - accel[down_troughs])>0.1 * accel_amp])

    return np.array(movements)

if __name__=='__main__':
    run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
    datasets = run_info.keys()
    params = []
    for dataset in run_info.keys():
        params.append(open('../data/peaks/%s_selected_param_spectral.txt'%(dataset)).read())

    min_height_list = [[0.2, 0.2]]*len(datasets)#[[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]
    reverse_scores = []
    monkey_labels = []
    for i, (dataset, param, min_heights) in enumerate(zip(datasets, params, min_height_list)):
        data_filename = '../data/intermediate/' + dataset + '.p'
        lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
        inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
        
        df = data_filename = pd.read_pickle(data_filename)
        input_info = io.loadmat(inputInfo_filename)
        with h5py.File(lfads_filename, 'r+') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        used_inds = range(df.index[-1][0] + 1)
        transitions = ss.dataset_events(df, ss.trial_transitions, 
                                        exclude_post_target=non_corrective_window)
    
        peaks = ta.get_peaks(co, dt, min_heights, exclude_post_target=non_feedback_window, df=df)
        peak_df,_ = ta.get_latencies(transitions, peaks, win_start=win_start, win_stop=win_stop, trial_len=trial_len)

        idx_train = np.load('../data/intermediate/train_test_split/%s_trials_train.npy'%dataset)
        idx_test = np.load('../data/intermediate/train_test_split/%s_trials_test.npy'%dataset)
        
        df_train = peak_df.loc[idx_train]
        df_test = peak_df.loc[idx_test]
        
        df_train = filter_peaks(df_train)
        df_test = filter_peaks(df_test)
        
        df_train, df_test = (df_train.sort_index(), df_test.sort_index())
        df_train.to_pickle('../data/peaks/%s_fb_peaks_train.p'%(dataset))
        peak_df.to_pickle('../data/peaks/%s_fb_peaks_train.p'%(dataset))
        df_test.to_pickle('../data/peaks/%s_fb_peaks_test.p'%(dataset))