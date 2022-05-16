import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from ast import literal_eval
import os
import sys
import yaml
sys.path.insert(0, '..')
import utils
import model_evaluation

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 20

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
datasets = run_info.keys()
co_dims = range(5)

for dataset in datasets:
    output = pd.read_csv('../data/peaks/param_search_targets-not-one.csv')
    best_idx = output.query('dataset == @dataset & ~fit_direction & ~use_rates')['mean_test_var_weighted_score'].idxmax()
    best_row = output.loc[best_idx]
    for co_dim in co_dims:
        param_dict = run_info[dataset]['params']
        for param in param_dict:
            kl_match = param_dict[param]['param_values']['kl_co_weight'] == cfg['selected_kl_weight']
            if kl_match and param_dict[param]['param_values']['co_dim'] == co_dim:
                break

        data_filename = '../../data/intermediate/' + dataset + '.p'
        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
        inputInfo_filename = '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
        test_peak_df = '../data/peaks/%s_targets-not-one_test.p'%dataset

        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)

        out