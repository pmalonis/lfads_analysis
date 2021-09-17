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

# train_filename = snakemake.output.train_data
# test_filename = snakemake.output.test_data
# all_filename = snakemake.output.all_data
# data_filename = snakemake.input[0]

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())

for dataset in datasets:
    train_filename = "../data/peaks/%s_new-corrections_train.p"%dataset
    test_filename = "../data/peaks/%s_new-corrections_test.p"%dataset
    all_filename = "../data/peaks/%s_new-corrections_all.p"%dataset
    firstmove_filename = "../data/peaks/%s_new-firstmove_all.p"%dataset
    data_filename = "../data/intermediate/%s.p"%dataset

    config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))
    random_state = 1027
    train_test_ratio = cfg['event_split_ratio']
    #non_corrective_window = 0.3

    # def get_next_target(_df, data):
    #     i = _df.index[0][0]
    #     if i not in firstmove_df.index.get_level_values('trial'):
    #         return

    #     firstmove_times = firstmove_df.loc[i].index.values
    #     targets = data.loc[i].kinematic.query('hit_target')
    #     transition_times = _df.loc[i].index.values
    #     target_idx = []
    #     to_drop = []
    #     for j,t in enumerate(transition_times): 
    #         #making sure transition occurs before the last target is acquired
    #         if np.any(targets.index > t): #and ~np.any(np.isclose(firstmove_times, t)):
    #             target_idx.append(targets.index.get_loc(t,method='bfill'))
    #         else:
    #             to_drop.append(j)

    #     x, y = targets.iloc[target_idx][['x','y']].values.T
    #     _df = _df.drop(index=_df.iloc[to_drop].index)
    #     _df['target_x'] = x
    #     _df['target_y'] = y
        
    #     if len(target_idx) > 0:
    #         return _df.loc[i]

    def get_next_target(_df, data):
        i = _df.index[0][0]
        targets = data.loc[i].kinematic.query('hit_target')
        transition_times = _df.loc[i].index.values
        target_idx = []
        to_drop = []
        for j,t in enumerate(transition_times): 
            #making sure transition occurs before the last target is acquired
    
            if np.any(targets.index > t): #and ~np.any(np.isclose(firstmove_times, t)):
                target_idx.append(targets.index.get_loc(t,method='bfill'))
            else:
                to_drop.append(j)

        x, y = targets.iloc[target_idx][['x','y']].values.T
        _df = _df.drop(index=_df.iloc[to_drop].index)
        _df['target_x'] = x
        _df['target_y'] = y
        
        if len(target_idx) > 0:
            return _df.loc[i]


    def split_correction_df(df):
        transition_df = ss.dataset_events(df, ss.trial_corrections)
        correction_df = transition_df.groupby('trial').apply(lambda trial_correct: get_next_target(trial_correct, df))
        df_train, df_test = train_test_split(correction_df, test_size=train_test_ratio, random_state=random_state)
        df_train, df_test = (df_train.sort_index(), df_test.sort_index())
        df_train.to_pickle(train_filename)
        df_test.to_pickle(test_filename)
        correction_df.to_pickle(all_filename)
        return df_train, df_test

    if __name__=='__main__':
        df = pd.read_pickle(data_filename)
        
        split_correction_df(df)