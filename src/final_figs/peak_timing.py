import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import yaml
import sys
import os
from scipy.signal import savgol_filter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
import timing_analysis as ta
import seaborn as sns
plt.rcParams['font.size'] = 18
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

win_start = 0
win_stop = 0.3

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

selection_metric = cfg['selection_metric']

if __name__=='__main__':
    run_info_path = os.path.join(os.path.dirname(__file__),'../../lfads_file_locations.yml')
    run_info = yaml.safe_load(open(run_info_path, 'r'))
    datasets = list(run_info.keys())
    params = []

    raju_thresholds = [0.05, 0.1, 0.15, 0.2]
    other_thresholds = [0.1, 0.2, 0.3, 0.4]
    event = 'targets-not-one'
    all_peak_times = {}
    for dataset in datasets:
        param_filename = os.path.join(os.path.dirname(__file__),
                '../../data/peaks/%s_selected_param_%s.txt'%(dataset,selection_metric))
        params.append(open(param_filename).read())

    for dataset, param in zip(datasets, params):
        data_filename = os.path.join(os.path.dirname(__file__),'../../data/intermediate/' + dataset + '.p')
        lfads_filename = os.path.join(os.path.dirname(__file__),
                        '../../data/model_output/' + '_'.join([dataset, param, 'all.h5']))
        inputInfo_filename = os.path.join(os.path.dirname(__file__), 
                            '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat']))

        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)
        n_inputs = co.shape[2]
        #plt.figure()
        #plt.suptitle(run_info[dataset]['name'])
        if 'raju' in dataset:
            thresholds = raju_thresholds
        else:
            thresholds = other_thresholds

        peak_times = [[[] for thresh_idx in range(len(thresholds))] for i in range(n_inputs)]
        for thresh_idx,threshold in enumerate(thresholds):
            peak_path = os.path.join(os.path.dirname(__file__),
                        '../../data/peaks/separated_%s_%s_%s_all_thresh=%0.4f.p'%(dataset, param, event, threshold))
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
        all_peak_times[dataset] = peak_times

    for dataset in datasets:
        plt.figure(figsize=(9,13))
        dset_name = run_info[dataset]['name']
        if 'raju' in dataset:
            thresholds = raju_thresholds
        else:
            thresholds = other_thresholds

        colors = plt.cm.jet(np.linspace(0.5,1,len(thresholds)))
        peak_times = all_peak_times[dataset]
        for thresh_idx,threshold in enumerate(thresholds):
            n_inputs = len(peak_times)
            for i in range(n_inputs):
                plt.subplot(n_inputs, 1, i+1)
                if 'raju' in dataset:
                    label = 'threshold = %0.2f'%threshold
                else:
                    label = 'threshold = %0.1f'%threshold

                sns.distplot(np.concatenate(peak_times[i][thresh_idx]), 
                            hist=False, label=label, color=list(colors[thresh_idx]))
                # sns.distplot(np.concatenate(peak_times[i][thresh_idx]), 
                #              hist=True, label=label, color=list(colors[thresh_idx]), norm_hist=False)
                plt.xlabel('Time From Target of Peaks for Input %d (ms)'%(i+1))
                plt.ylabel('Probability Density')
                ymin, ymax  = plt.yticks()[0][[0,-1]]
                plt.yticks([0, 0.01])
                plt.xticks([0, 50, 150, 250, 350])
                plt.suptitle('Monkey %s'%dset_name)
                plt.tight_layout()
        
        #plt.savefig('../../figures/final_figures/peak_timing.svg')
        print("FIGNUMS: %s"%plt.get_fignums())
        for fignum in plt.get_fignums():
            plt.figure(fignum)
            fig_filename = os.path.join(os.path.dirname(__file__), 
                            '../../figures/final_figures/numbered/4b-%d.pdf'%fignum)        
            plt.savefig(fig_filename)