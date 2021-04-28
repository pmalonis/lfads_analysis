import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optimize_target_prediction as opt
import utils
from importlib import reload
import os
import yaml
from scipy import io
import h5py
from matplotlib import rcParams
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.family'] = 'FreeSarif'
reload(opt)
reload(utils)


config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

datasets = ['rockstar','raju', 'mack']
params = ['final-fixed-2OLS24', 'final-fixed-2OLS24', 'mack-kl-co-sweep-0Wo8i9']
nbins = 12
win_start = cfg['post_target_win_start']
win_stop = cfg['post_target_win_stop']

lfads_filename = '../data/model_output/' + '_'.join([datasets[0], params[0], 'all.h5'])
with h5py.File(lfads_filename, 'r+') as h5file:
    co = h5file['controller_outputs'][:]
    
n_co = co.shape[2]

if __name__=='__main__':
    fig1, axplot = plt.subplots(n_co, len(datasets))
    for i, (dataset, param) in enumerate(zip(datasets, params)):
        data_filename = '../data/intermediate/' + dataset + '.p'
        lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
        inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
        
        df = data_filename = pd.read_pickle(data_filename)
        input_info = io.loadmat(inputInfo_filename)
        with h5py.File(lfads_filename, 'r+') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        peak_df_train = pd.read_pickle('../data/peaks/%s_%s_peaks_train.p'%(dataset,param))
        peak_df_test = pd.read_pickle('../data/peaks/%s_%s_peaks_test.p'%(dataset,param))

        peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()

        X,y = opt.get_inputs_to_model(peak_df, co, trial_len, dt, 
                                    win_start=win_start, 
                                    win_stop=win_stop)

        theta = np.arctan2(y[:,1], y[:,0])
        assert(nbins%2==0)
        
        bin_theta = np.pi / (nbins/2)
        colors = sns.color_palette('hls', nbins)
        t_ms = np.arange(win_start, win_stop, dt) * 1000
        for j in range(n_co):
            fig, axplot = plt.subplots()
            axplot.set_title('Controller %s'%(j+1))
            ymin, ymax = (np.min(X[:,j::n_co]), np.max(X[:,j::n_co]))
            
            #setting dimensions for inset
            left = .7#win_start + .8 * (win_stop-win_start)
            bottom = .7#ymin + .8 * (ymax-ymin)
            width = .15 #* (win_stop - win_start)
            height = .15 #* (ymax - ymin)
            inset = fig.add_axes([left, bottom, width, height], polar=True)
            co_min = np.inf
            co_max = -np.inf
            for i in range(-nbins//2, nbins//2):
                min_theta = i * bin_theta
                max_theta = (i+1) * bin_theta
                co_av = X[:,j::n_co][(theta > min_theta) & (theta <= max_theta)].mean(0)
                
                color = colors[i+nbins//2]
                axplot.plot(t_ms, co_av, color=color)
                inset.hist([(i+0.5) * bin_theta], [min_theta, max_theta], color=color)
                if np.min(co_av) < co_min:
                    co_argmin = i
                    co_min = np.min(co_av)
                
                if np.max(co_av) > co_max:
                    co_argmax = i
                    co_max = np.max(co_av)

            fig2, extremeplot = plt.subplots()
            inset_ex = fig2.add_axes([left, bottom, width, height], polar=True)
            for i in [co_argmin, co_argmax]:        
                min_theta = i * bin_theta
                max_theta = (i+1) * bin_theta
                x_binned = X[:,j::n_co][(theta > min_theta) & (theta <= max_theta)]
                co_av = x_binned.mean(0)
                error = x_binned.std(0)#/x_binned.shape[0]
                color = colors[i+nbins//2]
                extremeplot.plot(t_ms, co_av, color=color)
                extremeplot.fill_between(t_ms, co_av-error,co_av+error, alpha=0.3, color=color)
                inset_ex.hist([(i+0.5) * bin_theta], [min_theta, max_theta], color=color)


            axplot.set_xlabel('Time from target appearance (ms)')
            axplot.set_ylabel('Controller Value (a.u.)')
            axplot.set_title('Monkey %s'%dataset)
            inset.set_rticks([])
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title('Target Direction', fontdict={'fontsize': 8})
            inset.spines['polar'].set_visible(False)
            inset_ex.set_rticks([])
            inset_ex.set_xticks([])
            inset_ex.set_yticks([])
            inset_ex.set_title('Target Direction', fontdict={'fontsize': 8})
            #axplot.set_ylim([ymin, ymax])

            
                