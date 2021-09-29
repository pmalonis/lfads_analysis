import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from ast import literal_eval
import os
import sys
import yaml
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
sys.path.insert(0, '.')
import utils
import model_evaluation as me
import optimize_target_prediction as otp

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 20

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = run_info.keys()
co_dims = range(1,5)
all_mean_scores = []
all_std_scores = []
for dataset in datasets:
    mean_scores = []
    std_scores = []
    for co_dim in co_dims:
        output = pd.read_csv('../data/peaks/params_search_targets-not-one.csv')
        dset_name = run_info[dataset]['name']
        best_idx = output.query('dataset == @dset_name & ~fit_direction & ~use_rates')['mean_test_var_weighted_score'].idxmax()
        best_row = output.loc[best_idx]
        param_dict = run_info[dataset]['params']
        for param in param_dict:
            selected_param = open('../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read()
            selected_kl = param_dict[selected_param]['param_values']['kl_co_weight'] 
            kl_match = param_dict[param]['param_values']['kl_co_weight'] == selected_kl
            if kl_match and param_dict[param]['param_values']['co_dim'] == co_dim:
                break

        data_filename = '../data/intermediate/' + dataset + '.p'
        lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
        inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])

        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)
        test_peak_df = pd.read_pickle('../data/peaks/%s_targets-not-one_test.p'%dataset)

        preprocess_dict, model = me.get_row_params(best_row)
        X, y = otp.get_inputs_to_model(test_peak_df, co, trial_len, dt, df, **preprocess_dict)
        scorer = make_scorer(otp.var_weighted_score_func)
        scores = cross_val_score(model, X, y, scoring=scorer, 
                                cv=cfg['target_prediction_cv_splits'])
        mean_scores.append(np.mean(scores))
        std_scores.append(np.var(scores))

    all_mean_scores.append(mean_scores)
    all_std_scores.append(std_scores)

plt.figure(figsize=(8,6))
all_mean_scores = np.array(all_mean_scores)
all_std_scores = np.array(all_std_scores)
for i in range(len(datasets)):
    plt.plot(co_dims, all_mean_scores[i])
    plt.fill_between(co_dims, all_mean_scores[i]-np.sqrt(all_std_scores[i]), 
                    all_mean_scores[i]+np.sqrt(all_std_scores[i]), alpha=0.2)
    plt.xlabel('Inferred Input Dimensions')
    plt.ylabel('Decoding Performance $\mathregular{r^2}$')
    plt.xticks(co_dims)

plt.savefig('../figures/final_figures/numbered/2f.pdf')