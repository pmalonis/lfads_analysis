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

random_state = 1027
train_test_ratio = 0.2

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())
datasets = [datasets[0]]
for dataset in datasets:
    train_filename = "../data/peaks/%s_new-firstmove_train.p"%dataset
    test_filename = "../data/peaks/%s_new-firstmove_test.p"%dataset
    all_filename = "../data/peaks/%s_new-firstmove_all.p"%dataset
    data_filename = "../data/intermediate/%s.p"%dataset

    config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))
    random_state = 1027
    train_test_ratio = cfg['event_split_ratio']

    # def get_next_target(_df, data):
    #     i = _df.index[0][0]
    #     targets = data.loc[i].kinematic.query('hit_target')
    #     transition_times = _df.loc[i].index.values
    #     target_idx = []
    #     to_drop = []
    #     for j,t in enumerate(transition_times):
    #         if np.any(targets.index > t):
    #             prev_target_idx = targets.index.get_loc(t, method='ffill')
    #             t_prev_target = targets.index[prev_target_idx]
    #             if any((t_prev_target < transition_times) & (transition_times < t)):
    #                 to_drop.append(j)
    #             else:
    #                 target_idx.append(targets.index.get_loc(t, method='bfill'))
    #         else:
    #             to_drop.append(j)

    #     x, y = targets.iloc[target_idx][['x','y']].values.T
    #     if len(to_drop) > 0:
    #         _df = _df.drop(index=_df.iloc[to_drop].index)

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
            if np.any(targets.index > t):
                # prev_target_idx = targets.index.get_loc(t, method='ffill')
                # t_prev_target = targets.index[prev_target_idx]
                # if any((t_prev_target < transition_times) & (transition_times < t)):
                #     to_drop.append(j)
                # else:
                target_idx.append(targets.index.get_loc(t, method='bfill'))
            else:
                to_drop.append(j)

        x, y = targets.iloc[target_idx][['x','y']].values.T
        if len(to_drop) > 0:
            _df = _df.drop(index=_df.iloc[to_drop].index)

        _df['target_x'] = x
        _df['target_y'] = y
        
        if len(target_idx) > 0:
            return _df.loc[i]

    # def get_firstmove(_df, target_df, data):
    #     i = _df.index[0][0]
    #     targets = data.loc[i].kinematic.query('hit_target')
    #     transition_times = _df.loc[i].index.values
    #     target_idx = []
    #     to_drop = []
    #     for j,t in enumerate(transition_times): 
    #         #making sure transition occurs before the last target is acquired
    #         if np.any(targets.index > t) and ~np.any(np.isclose(firstmove_times, t)):
    #             target_idx.append(targets.index.get_loc(t,method='bfill'))
    #         else:
    #             to_drop.append(j)

    #     x, y = targets.iloc[target_idx][['x','y']].values.T
    #     _df = _df.drop(index=_df.iloc[to_drop].index)
    #     _df['target_x'] = x
    #     _df['target_y'] = y
        
    #     if len(target_idx) > 0:
    #         return _df.loc[i]

    #     return firstmoves_trial_df

    def split_firstmove_df(df):
        
        transition_df = ss.dataset_events(df, ss.trial_firstmoves)#, exclude_post_target=0,exclude_pre_target=0)
        target_df = df.kinematic.query('hit_target')
        #firstmove_df = transition_df.groupby('trial').apply(lambda _df: get_firstmove(_df, target_df, df))
        #firstmove_df = firstmove_df.groupby('trial').apply(lambda trial: get_next_target(trial, df))
        firstmove_df = transition_df.groupby('trial').apply(lambda trial: get_next_target(trial, df))
        firstmove_df.groupby('trial').apply(lambda _df: _df.loc[_df.index[0][0]].iloc[1:])
        df_train, df_test = train_test_split(firstmove_df, test_size=train_test_ratio, random_state=random_state)
        df_train, df_test = (df_train.sort_index(), df_test.sort_index())
        df_train.to_pickle(train_filename)
        df_test.to_pickle(test_filename)
        firstmove_df.to_pickle(all_filename)

        return df_train, df_test

    if __name__=='__main__':
        df = pd.read_pickle(data_filename)
        split_firstmove_df(df)