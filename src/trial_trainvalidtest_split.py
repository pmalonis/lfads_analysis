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

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

random_state = 1027
train_test_ratio = cfg['event_split_ratio']

if __name__=='__main__':
    datasets = ['raju-M1-no-bad-trials']
    win_start = 0
    win_stop = 0.5
    min_height_list = [[0.3,0.3]]*len(datasets)#[[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]
    reverse_scores = []
    monkey_labels = []
    for i,dataset in enumerate(datasets):
        data_filename = '../data/intermediate/' + dataset + '.p'
        df = data_filename = pd.read_pickle(data_filename)

        idx = range(df.index[-1][0] + 1)
        idx_train, idx_test = train_test_split(idx, test_size=train_test_ratio, random_state=random_state)
        
        np.save('../data/intermediate/train_test_split/%s_trials_train.npy'%dataset, idx_train)
        np.save('../data/intermediate/train_test_split/%s_trials_test.npy'%dataset, idx_test)