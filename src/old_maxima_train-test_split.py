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

train_filename = snakemake.output.train_data
test_filename = snakemake.output.test_data
all_filename = snakemake.output.all_data
data_filename = snakemake.input[0]

random_state = 1027
train_test_ratio = 0.2
non_feedback_window = 0.2
non_corrective_window = 0.3

def get_next_target(_df, data):
    i = _df.index[0][0]
    targets = data.loc[i].kinematic.query('hit_target')
    transition_times = _df.loc[i].index.values
    target_idx = []
    to_drop = []
    for j,t in enumerate(transition_times):
        if np.any(targets.index > t):
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

def get_maxima(_df, target_df, data):
    i = _df.index[0][0]
    trial_data = data.loc[i].kinematic
    target_times = target_df.loc[i].index.values
    firstmoves = []
    speed = np.sqrt(trial_data['x_vel']**2+trial_data['y_vel']**2)
    for j,t in enumerate(target_times[:-1]):
        if t >= _df.loc[i].index[-1]: #continue if no transition after taget
            continue
        idx = _df.loc[i].index.get_loc(t, 'bfill')
        if _df.loc[i].index[idx] < target_times[j+1]:
            move = _df.loc[i].iloc[idx]
        else: #get minimum speed
            post_target_window = min(t+0.3, target_times[j+1])
            move = trial_data.loc[speed.loc[t:post_target_window].idxmin()]

        firstmoves.append(move)
        
    firstmoves_trial_df = pd.concat(firstmoves, axis=1).T
    firstmoves_trial_df.index.rename('time',inplace=True)

    return firstmoves_trial_df

def split_maxima_df(df):
    maxima_df = ss.dataset_events(df, ss.trial_maxima, exclude_post_target=0,exclude_pre_target=0)
    target_df = df.kinematic.query('hit_target')
    maxima_df = maxima_df.groupby('trial').apply(lambda trial: get_next_target(trial, df))
    df_train, df_test = train_test_split(maxima_df, test_size=train_test_ratio, random_state=random_state)
    df_train, df_test = (df_train.sort_index(), df_test.sort_index())
    df_train.to_pickle(train_filename)
    df_test.to_pickle(test_filename)
    maxima_df.to_pickle(all_filename)

    return df_train, df_test


if __name__=='__main__':
    df = pd.read_pickle(data_filename)
    split_maxima_df(df)