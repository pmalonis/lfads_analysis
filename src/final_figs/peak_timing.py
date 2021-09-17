import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import yaml
import sys
import os
from scipy.signal import savgol_filter
sys.path.insert(0, '..')
import utils
import timing_analysis as ta
import seaborn as sns
plt.rcParams['font.size'] = 20

win_start = 0
win_stop = 0.3

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

selection_metric = cfg['selection_metric']

if __name__=='__main__':
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    datasets = run_info.keys()
    datasets=[list(datasets)[0]]
    params = []
    n_inputs = cfg['selected_co_dim']
    thresholds = [0.1, 0.2, 0.3, 0.4]
    event = 'targets-not-one'
    #thresholds = [0.4, 0.5]
    peak_times = [[[] for thresh_idx in range(len(thresholds))] for i in range(n_inputs)]

    for dataset in datasets:
        params.append(open('../../data/peaks/%s_selected_param_%s.txt'%(dataset,selection_metric)).read())

    for dataset, param in zip(datasets, params):
        data_filename = '../../data/intermediate/' + dataset + '.p'
        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
        inputInfo_filename = '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])

        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)

        #plt.figure()
        #plt.suptitle(run_info[dataset]['name'])
        for thresh_idx,threshold in enumerate(thresholds):
            peak_path = '../../data/peaks/separated_%s_%s_%s_all_thresh=%0.1f.p'%(dataset, param, event, threshold)
            if os.path.exists(peak_path):
                peak_df = pd.read_pickle(peak_path)
            else:
                if 'target' in event:
                    win_start = 0.0
                    win_stop = 0.3
                else: 
                    win_start = -0.3
                    win_stop = 0

                event_df = pd.read_pickle('../../data/peaks/%s_%s_all.p'%(dataset,event))
                peak_df = ta.get_peak_df(df, co, trial_len, threshold, event_df, dt, win_start=win_start, win_stop=win_stop)
                peak_df.to_pickle(peak_path)

            for i in range(n_inputs):
                #plt.subplot(n_inputs, 1, i+1)
                label = 'threshold = %0.1f'%threshold
                t = peak_df['latency_%d'%i].values*1000
                #sns.distplot(t, hist=False, label=label)
                peak_times[i][thresh_idx].append(t)

    plt.figure(figsize=(9,13))
    colors = plt.cm.jet(np.linspace(0.5,1,len(thresholds)))
    for thresh_idx,threshold in enumerate(thresholds):
        for i in range(n_inputs):
            plt.subplot(n_inputs, 1, i+1)
            label = 'threshold = %0.1f'%threshold
            sns.distplot(np.concatenate(peak_times[i][thresh_idx]), 
                        hist=False, label=label, color=list(colors[thresh_idx]))
            # sns.distplot(np.concatenate(peak_times[i][thresh_idx]), 
            #              hist=True, label=label, color=list(colors[thresh_idx]), norm_hist=False)
            plt.xlabel('Time From Target of Peaks for Input %d (ms)'%(i+1))
            plt.ylabel('Probability Density')
            ymin, ymax  = plt.yticks()[0][[0,-1]]
            plt.yticks([0, 0.01])
    
    plt.tight_layout()
    plt.savefig('../../figures/final_figures/peak_timing.svg')
    plt.savefig('../../figures/final_figures/4b.svg')