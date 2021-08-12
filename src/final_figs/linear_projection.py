from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import io
import h5py
import yaml
import os
import sys
sys.path.insert(0, '..')
import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__=='__main__':
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))
    
    datasets = run_info.keys()
    dataset_names = [run_info[dataset]['name'] for dataset in datasets]
    params = []
    for dataset in datasets:
        params.append(open('../../data/peaks/%s_selected_param_%s.txt'%(dataset,
                                                                    cfg['selection_metric'])).read())
        #params.append(open('../../data/peaks/%s_selected_param_gini.txt'%(dataset)).read())

    scores = []
    for dataset, param in zip(datasets, params):
        data_filename = '../../data/intermediate/' + dataset + '.p'
        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
        inputInfo_filename = '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])

        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)
        used_inds = range(df.index[-1][0] + 1)
        sigma = cfg['rate_sigma_example_plot'] # using dt to ensure high enough resolution

        midpoint_idx = 4
        spike_dt = 0.001
        dt = 0.01
        win = int(dt/spike_dt)
        nneurons = sum('neural' in c for c in df.columns)
        all_smoothed = np.zeros((len(used_inds), int(trial_len/dt), nneurons)) #holds firing rates for whole experiment (to be used for dimensionality reduction)
        for i in used_inds:
            smoothed = df.loc[i].neural.rolling(window=sigma*6, min_periods=1, win_type='gaussian', center=True).mean(std=sigma)
            smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
            smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
            all_smoothed[i,:,:] = smoothed

        y = co.reshape((co.shape[0]*co.shape[1], co.shape[2]))
        X = all_smoothed.reshape((all_smoothed.shape[0]*all_smoothed.shape[1], all_smoothed.shape[2]))
        model = LinearRegression()
        model.fit(X,y)
        scores.append(model.score(X,y))

    plot_df = pd.DataFrame(np.array([dataset_names, scores]).T,
                            columns=['Monkey', 'Variance Explained'])
    sns.barplot(x='Monkey', y='Variance Explained',
                data=plot_df)
    plt.ylim([0, 1])
    plt.savefig('../../figures/final_figures/linear_projection.png')