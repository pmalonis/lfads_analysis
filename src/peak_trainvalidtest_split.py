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

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

random_state = 1027
train_test_ratio = cfg['event_split_ratio']

def split_target_df(df, dataset):
    target_df = ta.get_targets(df)
    df_train, df_test = train_test_split(target_df, test_size=train_test_ratio, random_state=random_state)
    df_train, df_test = (df_train.sort_index(), df_test.sort_index())
    df_train.to_pickle(os.path + '/../data/peaks/%s_targets_train.p'%dataset)
    df_test.to_pickle(os.path + '/../data/peaks/%s_targets_test.p'%dataset)

    return df_train, df_test

if __name__=='__main__':
    run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
    datasets = list(run_info.keys())
    for i, dataset in enumerate(datasets):
        data_filename = '../data/intermediate/' + dataset + '.p'
        df = data_filename = pd.read_pickle(data_filename)
        split_peak_df(df, co, trial_len, dt, dataset)