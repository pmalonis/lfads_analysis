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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
import timing_analysis as ta
from importlib import reload
reload(ta)
plt.rcParams['font.size'] = 16
#plt.rcParams['axes.spines.top'] = False
#plt.rcParams['axes.spines.right'] = False

win_start = 0
win_stop = 0.3

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

selection_metric = cfg['selection_metric']

def permutation_spread_test(samples, func, n=1000):
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
    params = []
    #cfg['selected_co_dim']
    thresholds = [0.1, 0.2, 0.3, 0.4]
    comp_thresh = [0.3, 0.3, 0.2]
    colors = utils.contrasting_colors(**cfg['colors']['target_vs_firstmove'])
    colors[0] = tuple(colors[0] - np.array([0, 0.2, 0.2]))
    for dataset in datasets:
        params.append(open('../../data/peaks/%s_selected_param_%s.txt'%(dataset,selection_metric)).read())

    plt.figure(figsize=(15,6))
    peak_df = {}
    for dset_idx, (dataset, param) in enumerate(zip(datasets, params)):
        data_filename = '../../data/intermediate/' + dataset + '.p'
        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
        inputInfo_filename = '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])

        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)
        co = np.abs(co).sum(2, keepdims=True)
        n_inputs = co.shape[2]
        peak_path = '../../data/peaks/%s_%s_post_target_maximum_accel_aligned.p'%(dataset, param)
        if os.path.exists(peak_path):
            peak_df[dataset] = pd.read_pickle(peak_path)
        else:
            firstmove_df = pd.read_pickle('../../data/peaks/%s_first-accel_all.p'%(dataset))
            target_df = df.kinematic.query('hit_target')
            peak_df[dataset] = ta.get_maximum_peaks(co, dt, df, firstmove_df)
            peak_df[dataset].to_pickle(peak_path)

    events = ['target','firstmove']
    for dset_idx, dataset in enumerate(datasets):
        samples = []
        plt.subplot(1, len(datasets), dset_idx+1)
        for event_idx, event in enumerate(events):
            latencies = [peak_df[dataset]['%s_latency_%d'%(event, i+1)].values for i in range(n_inputs)]
            latencies = np.concatenate(latencies) * 1000 # converting to ms
            latency_idx = ~np.isnan(latencies)
            latencies_notnull = latencies[latency_idx]
            samples.append(latencies_notnull)
            if dset_idx == 0:
                plt.ylabel(' Probability Density')
            if event_idx > 0:
                plt.twiny()

            if 'target' in event:
                #sns.distplot(latencies_notnull, hist=False, color=colors[event_idx])
                sns.distplot(latencies_notnull[latencies_notnull<310], hist=False, color=colors[event_idx])
                plt.xlabel('Time from Target of Total \n Input Magnitude Peaks (ms)', color=colors[event_idx])
                plt.xticks([0, 100, 200, 300])
                #plt.xlim([0,500]) 
            else:
                #sns.distplot(latencies_notnull, hist=False, color=colors[event_idx])
                sns.distplot(latencies_notnull[latencies_notnull > -310], hist=False, color=colors[event_idx])
                xticks,_ = plt.xticks()
                #plt.gca().set_xticklabels(-xticks.astype(int))
                plt.xticks([-300, -200, -100, 0])
                #plt.xlim([-300, 200])
                plt.xlabel('Time from First Movement of Total \n Input Magnitude Peaks (ms)', color=colors[event_idx])
                
        # print(levene(*samples))  
        print("%s p-value: %f"%(dataset, permutation_spread_test(samples, dist_width)))

        ymin, ymax  = plt.yticks()[0][[0,-1]]
        ymin = 0
        ymax = ymin + 0.9 * (ymax - ymin)
        plt.yticks([ymin, ymax], fontsize=14)
        plt.gca().set_yticklabels([str(ymin), str('%.4f'%ymax)])

        dset_name = run_info[dataset]['name']
        plt.title('Monkey %s'%dset_name)
        if dset_idx == 0:
            plt.text(-220, 0.014, 'Aligned to \n Target', color=colors[0], fontdict={'fontsize':16})
            plt.text(-170, 0.0065, 'Aligned to \n First Movement', color=colors[1], fontdict={'fontsize':16})
            #plt.legend(['Aligned to First Movement', 'Aligned to Target'])

    plt.tight_layout()
    plt.savefig('../../figures/final_figures/first-accel_peak_timing_comparison.svg')
    #plt.savefig('../../figures/final_figures/numbered/4c.svg')