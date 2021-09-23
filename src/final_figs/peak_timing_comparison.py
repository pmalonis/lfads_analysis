import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import yaml
import sys
import os
from scipy.signal import savgol_filter, peak_widths
from scipy.stats import levene, gaussian_kde
import seaborn as sns
sys.path.insert(0, '..')
import utils
import timing_analysis as ta
plt.rcParams['font.size'] = 16
#plt.rcParams['axes.spines.top'] = False
#plt.rcParams['axes.spines.right'] = False

win_start = 0
win_stop = 0.3

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

selection_metric = cfg['selection_metric']

def permutation_spread_test(samples, func, n=10):
    '''Permutation test for whether value of function is 
    significantly different between two distributions
    
    samples: array of two samples
    func: function that will be applied to the samples, as
    well as random splits of all the data in order to test
    significance
    '''
    v = []
    #aligning samples by their mode
    aligned_samples = []
    for sample in samples:
        kde = gaussian_kde(sample)
        t = np.arange(np.min(sample), np.max(sample))
        p = kde.pdf(t)
        mode = t[np.argmax(p)]
        aligned_samples.append(sample-mode)

    for i in range(n):
        s = len(aligned_samples[0]) #size of first sample
        sample1 = np.random.permutation(np.concatenate(aligned_samples))[:s]
        sample2 = np.random.permutation(np.concatenate(aligned_samples))[s:]
        sample_diff = np.abs(func(sample1) - func(sample2))
        v.append(sample_diff)

    actual_diff = np.abs(func(aligned_samples[0]) - func(aligned_samples[1]))
    p = np.sum(actual_diff < v)/len(v)
    return p

def dist_width(sample):
    kde = gaussian_kde(sample)

    pad = 100 #pad when calculating pdf, to make sure it includes full possible sample window for each distribution
    dist = kde.pdf(np.arange(np.min(sample)-pad,np.max(sample)+pad))
    peak = [np.argmax(dist)]
    width,_,_,_ = peak_widths(dist, peak)
    
    return width[0]

if __name__=='__main__':
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    datasets = run_info.keys()
    #datasets = [list(datasets)[0]]
    params = []
    n_inputs = 1#cfg['selected_co_dim']
    thresholds = [0.1, 0.2, 0.3, 0.4]
    comp_thresh = [0.3, 0.3, 0.2]
    event_types = ['targets-not-one', 'new-firstmove']
    peak_times = {e:[[[] for thresh_idx in range(len(thresholds))] for i in range(n_inputs)] for e in event_types}

    colors = utils.contrasting_colors(**cfg['colors']['target_vs_firstmove'])
    colors[0] = tuple(colors[0] - np.array([0, 0.2, 0.2]))
    for dataset in datasets:
        params.append(open('../../data/peaks/%s_selected_param_%s.txt'%(dataset,selection_metric)).read())

    plt.figure(figsize=(15,6))
    for dset_idx, (dataset, param) in enumerate(zip(datasets, params)):
        data_filename = '../../data/intermediate/' + dataset + '.p'
        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
        inputInfo_filename = '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])

        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)
        co = np.abs(co).sum(2, keepdims=True)
        peak_df = {}
        for thresh_idx,threshold in enumerate(thresholds):
            for event in event_types:
                peak_path = '../../data/peaks/%s_%s_%s_all_thresh=%0.1f.p'%(dataset, param, event, threshold)
                if os.path.exists(peak_path):
                    peak_df[event] = pd.read_pickle(peak_path)
                else:
                    if 'target' in event:
                        win_start = 0.0
                        win_stop = 0.3
                    else: 
                        win_start = -0.3
                        win_stop = 0

                    event_df = pd.read_pickle('../../data/peaks/%s_%s_all.p'%(dataset,event))
                    peak_df[event] = ta.get_peak_df(df, co, trial_len, threshold, event_df, dt, win_start=win_start, win_stop=win_stop)
                    peak_df[event].to_pickle(peak_path)

                for i in range(n_inputs):
                    #plt.subplot(n_inputs, 1, i+1)
                    label = 'threshold = %0.1f'%threshold
                    t = peak_df[event]['latency_%d'%i].values*1000
                    peak_times[event][i][thresh_idx] = t
            
            if threshold == comp_thresh[dset_idx]:
                plt.subplot(1, len(datasets), dset_idx+1)
                for i in range(n_inputs):
                    samples = []
                    for event_idx, event in enumerate(event_types):
                        latencies = peak_times[event][i][thresh_idx]
                        latency_idx = ~np.isnan(latencies)
                        latencies_notnull = latencies[latency_idx]
                        print(dataset)
                        print(len(latencies_notnull))
                        print('\n')
                        samples.append(latencies_notnull)
                        if dset_idx == 0:
                            plt.ylabel(' Probability Density')
                        if event_idx > 0:
                            plt.twiny()

                        if 'target' in event:
                            sns.distplot(samples[event_idx], hist=False, color=colors[event_idx])
                            plt.xlabel('Time from Target of Total \n Input Magnitude Peaks (ms)', color=colors[event_idx])
                        else:
                            sns.distplot(samples[event_idx], hist=False, color=colors[event_idx])
                            xticks,_ = plt.xticks()
                            #plt.gca().set_xticklabels(-xticks.astype(int))
                            plt.xlabel('Time from First Movement of Total \n Input Magnitude Peaks (ms)', color=colors[event_idx])
                            

                    print(levene(*samples))  
                    # print(permutation_spread_test(samples, np.var))
                    # print(permutation_spread_test(samples, dist_width))

                    ymin, ymax  = plt.yticks()[0][[0,-1]]
                    ymin = 0
                    ymax = ymin + 0.9 * (ymax - ymin)
                    plt.yticks([ymin, ymax], fontsize=14)
                    plt.gca().set_yticklabels([str(ymin), str('%.4f'%ymax)])

                dset_name = run_info[dataset]['name']
                plt.title('Monkey %s'%dset_name) 
                if dset_idx == 0:
                    plt.text(-190, 0.015, 'Aligned to \n Target', color=colors[0], fontdict={'fontsize':16})
                    plt.text(-160, 0.009, 'Aligned to \n First Movement', color=colors[1], fontdict={'fontsize':16})
            #plt.legend(['Aligned to First Movement', 'Aligned to Target'])

    # 
    # colors = plt.cm.jet(np.linspace(0.5,1,len(thresholds)))
    # for thresh_idx,threshold in enumerate(thresholds):
    #     for i in range(n_inputs):
    #         plt.subplot(n_inputs, 1, i+1)
    #         label = 'threshold = %0.1f'%threshold
    #         sns.distplot(np.concatenate(peak_times['targets-not-one'][i][thresh_idx]), 
    #                     hist=False, label=label, color=list(colors[thresh_idx]))
    #         # sns.distplot(np.concatenate(peak_times[i][thresh_idx]), 
    #         #              hist=True, label=label, color=list(colors[thresh_idx]), norm_hist=False)
    #         plt.xlabel('Latency to peak for input %d (ms)'%(i+1))
    #         plt.ylabel('Probability Density')
    #         ymin, ymax  = plt.yticks()[0][[0,-1]]
    #         plt.yticks([ymin, ymax])
    plt.tight_layout()
    plt.savefig('../../figures/final_figures/target_vs_firstmove_peak_timing.svg')
    plt.savefig('../../figures/final_figures/numbered/4c.svg')