import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
import optimize_target_prediction as opt
import feedback_optimize_target_prediction as fotp
import utils
from importlib import reload
import os
import yaml
from scipy import io
import h5py
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from matplotlib import rcParams
from scipy.stats import spearmanr
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 16
reload(opt)
reload(utils)                       

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r')) 

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())
params = []
for dataset in run_info.keys():
    params.append(open('../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric'])).read())

#datasets = ['rockstar', 'mack']#['rockstar','raju', 'mack']
#params = ['all-early-stop-kl-sweep-yKzIQf', 'all-early-stop-kl-sweep-bMGCVf']#['mack-kl-co-sweep-0Wo8i9']#['final-fixed-2OLS24', 'final-fixed-2OLS24', 'mack-kl-co-sweep-0Wo8i9']
nbins = 6
fb_win_start = -0.3#0.00#-0.1#cfg['post_target_win_start']
fb_win_stop = 0.0#0.3#0.1#cfg['post_target_win_stop']
win_start = -0.3
win_stop = 0.0

lfads_filename = '../data/model_output/' + '_'.join([datasets[0], params[0], 'all.h5'])
with h5py.File(lfads_filename, 'r+') as h5file:
    co = h5file['controller_outputs'][:]

co = savgol_filter(co, 11, 2, axis=1)
n_co = co.shape[2]

if __name__=='__main__':
    all_means = []
    fb_all_means = []
    for dset_idx, (dataset, param) in enumerate(zip(datasets, params)):
        data_filename = '../data/intermediate/' + dataset + '.p'
        lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
        inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
        
        df = data_filename = pd.read_pickle(data_filename)
        input_info = io.loadmat(inputInfo_filename)
        with h5py.File(lfads_filename, 'r+') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        # peak_df_train = pd.read_pickle('../data/peaks/%s_new-firstmove_train.p'%(dataset))
        # peak_df_test = pd.read_pickle('../data/peaks/%s_new-firstmove_test.p'%(dataset))

        # peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()
        peak_df = pd.read_pickle('../data/peaks/%s_new-firstmove_all.p'%(dataset))

        # fb_peak_df_train = pd.read_pickle('../data/peaks/%s_new-corrections_train.p'%(dataset))
        # fb_peak_df_test = pd.read_pickle('../data/peaks/%s_new-corrections_test.p'%(dataset))
        
        #pd.concat([fb_peak_df_train, fb_peak_df_test]).sort_index()
        fb_peak_df = pd.read_pickle('../data/peaks/%s_new-corrections_all.p'%(dataset))
        
        X,y = opt.get_inputs_to_model(peak_df, co, trial_len, dt, df=df, 
                                    win_start=win_start, 
                                    win_stop=win_stop)

        fb_X,fb_y = opt.get_inputs_to_model(fb_peak_df, co, trial_len, dt, df=df,
                                            win_start=fb_win_start, win_stop=fb_win_stop)

        win_size = int((win_stop - win_start)/dt)
        fb_win_size = int((fb_win_stop - fb_win_start)/dt)

        theta = np.arctan2(y[:,1], y[:,0])
        fb_theta = np.arctan2(fb_y[:,1], fb_y[:,0])
        assert(nbins%2==0)
        
        bin_theta = np.pi / (nbins/2)
        mean_zscores = np.zeros(n_co*nbins)
        max_zscores = np.zeros(n_co*nbins)
        fb_mean_zscores = np.zeros(n_co*nbins)
        fb_max_zscores = np.zeros(n_co*nbins)
        for j in range(n_co):
            ymin, ymax = (np.min(X[:,j*win_size:(j+1)*win_size]), np.max(X[:,j*win_size:(j+1)*win_size]))
            fb_co_mean = []
            co_mean = []
            fb_co_max = []
            co_max = []
            fb_co_rank = [] 
            co_rank = [] 
            for i in range(-nbins//2, nbins//2):
                min_theta = i * bin_theta
                max_theta = (i+1) * bin_theta
                co_av = X[:,j*win_size:(j+1)*win_size][(theta > min_theta) & (theta <= max_theta)].mean(0)
                fb_co_av = fb_X[:,j*fb_win_size:(j+1)*fb_win_size][(fb_theta > min_theta) & (fb_theta <= max_theta)].mean(0) 
                co_mean.append(co_av.mean())
                fb_co_mean.append(fb_co_av.mean())
                co_max.append(co_av.max())
                fb_co_max.append(fb_co_av.max())

            co_rank += list(np.argsort(co_max))
            fb_co_rank += list(np.argsort(fb_co_max))

            mean_zscores[j*nbins:(j+1)*nbins] = zscore(co_mean)
            max_zscores[j*nbins:(j+1)*nbins] = zscore(co_max)
            fb_mean_zscores[j*nbins:(j+1)*nbins] = zscore(fb_co_mean)
            fb_max_zscores[j*nbins:(j+1)*nbins] = zscore(fb_co_max)

        plt.figure(1)
        #plt.tight_layout(pad=2)
        #plt.tight_layout()
        plt.subplot(1, len(datasets), dset_idx+1)
        sns.regplot(mean_zscores, fb_mean_zscores)
        if dset_idx == 0:
            plt.ylabel('Corrective Movement Direction-Mean (z-scored)')
        if j==0:
            plt.title(run_info[dataset]['name'])

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        xpos = xmin + .1*(xmax-xmin)
        ypos = ymax - .1*(ymax-ymin)
        r_value, p_value = pearsonr(mean_zscores, fb_mean_zscores)
        if 0.01 <= p_value < 0.05:
            p_str = 'p = %0.2f'%p_value
        elif 0.001 <= p_value < 0.01:
            p_str = 'p < 0.01'
        elif p_value < 0.001:
            p_str = 'p < 0.001'
        else:
            p_str = 'p > 0.05'

        plt.text(xpos,ypos,
                'r = %0.2f, %s'%(r_value, p_str), fontsize=13)

        plt.yticks([-2, -1, 0, 1, 2])
        plt.xticks([-2, -1, 0, 1, 2])

        dset_name = run_info[dataset]['name']
        plt.title('Monkey %s'%dset_name) 
        # plt.figure(2)
        # plt.tight_layout(pad=2)
        # plt.suptitle('Max of Controller Direction-Averages')
        # #plt.tight_layout()
        # plt.subplot(1,len(datasets),dset_idx+1)
        # sns.regplot(max_zscores, fb_max_zscores)
        # plt.xlabel('Initial Movement')
        # plt.ylabel('Corretive Movement')
        # if j==0:
        #     plt.title(run_info[dataset]['name'])
        # xmin, xmax = plt.xlim()
        # ymin, ymax = plt.ylim()
        # xpos = xmin + .1*(xmax-xmin)
        # ypos = ymax - .1*(ymax-ymin)
        # plt.text(xpos,ypos,
        #         'r = %0.2f'%np.corrcoef(max_zscores, fb_max_zscores)[1,0])

        all_means.append(mean_zscores)
        fb_all_means.append(fb_mean_zscores)

    fig = plt.figure(1)
    fig.text(0.5, 0.01, 'Initial Movement Direction-Mean (z-scored)',ha='center')
    
    #fig.text(0.02, .75, 'Rate PC 1')
    #fig.text(00.02, .25, 'Rate PC 2')
    fig.set_size_inches(12,6)
    #plt.savefig('../figures/final_figures/co_averages_correlation-means.svg')
    #plt.savefig('../figures/final_figures/numbered/7b.svg')
    #fig = plt.figure(2)
    #fig.text(0.02, .75, 'Rate PC 1')
    #fig.text(0.02, .25, 'Rate PC 2')
    #fig.set_size_inches(12,6)
    #plt.savefig('../figures/final_figures/co_averages_correlation-maxima.png')

    plt.figure(figsize=(8,6))
    #plt.tight_layout(pad=2)
    #plt.tight_layout()
    sns.regplot(np.concatenate(all_means), np.concatenate(fb_all_means))
    plt.xlabel('Initial Movement Direction-Mean (z-scored)')
    plt.ylabel('Corrective Movement Direction-Mean (z-scored)')

    if j==0:
        plt.title(run_info[dataset]['name'])
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xpos = xmin + .1*(xmax-xmin)
    ypos = ymax - .1*(ymax-ymin)
    r_value, p_value = pearsonr(np.concatenate(all_means), np.concatenate(fb_all_means))
    if 0.01 <= p_value < 0.05:
        p_str = 'p = %0.3f'%p_value
    elif 0.001 <= p_value < 0.01:
        p_str = 'p < 0.01'
    elif p_value < 0.001:
        p_str = 'p < 0.001'
    else:
        p_str = 'p > 0.05'

    plt.text(xpos,ypos,
            'r = %0.2f, %s'%(r_value, p_str), fontsize=13)

    plt.yticks([-2, -1, 0, 1, 2])
    plt.xticks([-2, -1, 0, 1, 2])
    plt.savefig('../figures/final_figures/co_averages_correlation_with_corrections_means_all.svg')
    #plt.savefig('../figures/final_figures/numbered/7c.svg')