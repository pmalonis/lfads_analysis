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
import os
import yaml
from scipy import io
import pickle
from importlib import reload
reload(ta)

random_state = 1027
train_test_ratio = 0.2
min_heights = 0.3
win_start = 0
win_stop = 0.3

def split_peak_df(df, co, trial_len, dt, dataset, param):
    peak_df = ta.get_peak_df(df, co, trial_len, min_heights, dt, win_start=win_start, win_stop=win_stop)
    df_train, df_test = train_test_split(peak_df, test_size=train_test_ratio, random_state=random_state)
    df_train, df_test = (df_train.sort_index(), df_test.sort_index())
    df_train.to_pickle('../data/peaks/%s_%s_peaks_train.p'%(dataset, param))
    df_test.to_pickle('../data/peaks/%s_%s_peaks_test.p'%(dataset, param))

    return df_train, df_test

if __name__=='__main__':
    #datasets = ['rockstar','raju', 'mack']
    #params = ['final-fixed-2OLS24', 'final-fixed-2OLS24', 'mack-kl-co-sweep-0Wo8i9']
    # datasets = ['rockstar', 'rockstar', 'rockstar', 'rockstar']#,'mack']
    # params = ['all-early-stop-kl-sweep-OvP3yt',
    #         'all-early-stop-kl-sweep-6kE5-V',
    #         'all-early-stop-kl-sweep-tLbfG6',
    #         'all-early-stop-kl-sweep-yKzIQf']

    run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
    datasets = list(run_info.keys())
    params = []
    for dataset in datasets:
        params.append(open('../data/peaks/%s_selected_param_spectral.txt'%(dataset)).read())

    min_height_list = [[0.3,0.3]]*len(datasets)#[[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]
    reverse_scores = []
    monkey_labels = []
    for i, (dataset, param) in enumerate(zip(datasets, params)):
        data_filename = '../data/intermediate/' + dataset + '.p'
        lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
        inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
        
        df = data_filename = pd.read_pickle(data_filename)
        input_info = io.loadmat(inputInfo_filename)
        with h5py.File(lfads_filename, 'r+') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        split_peak_df(df, co, trial_len, dt, dataset, param)