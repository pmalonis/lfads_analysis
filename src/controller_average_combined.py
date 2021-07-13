import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
reload(opt)
reload(utils)


config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())
params = []
for dataset in run_info.keys():
    params.append(open('../data/peaks/%s_selected_param_gini.txt'%(dataset)).read())


#datasets = ['rockstar', 'mack']#['rockstar','raju', 'mack']
#params = ['all-early-stop-kl-sweep-yKzIQf', 'all-early-stop-kl-sweep-bMGCVf']#['mack-kl-co-sweep-0Wo8i9']#['final-fixed-2OLS24', 'final-fixed-2OLS24', 'mack-kl-co-sweep-0Wo8i9']
nbins = 12
fb_win_start = -0.2#0.00#-0.1#cfg['post_target_win_start']
fb_win_stop = 0 #0.3#0.1#cfg['post_target_win_stop']
win_start = -0.2
win_stop = 0

lfads_filename = '../data/model_output/' + '_'.join([datasets[0], params[0], 'all.h5'])
with h5py.File(lfads_filename, 'r+') as h5file:
    co = h5file['controller_outputs'][:]
    
n_co = co.shape[2]

if __name__=='__main__':
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

        co = savgol_filter(co, 11, 2, axis=1)
        peak_df_train = pd.read_pickle('../data/peaks/%s_firstmove_train.p'%(dataset))
        peak_df_test = pd.read_pickle('../data/peaks/%s_firstmove_test.p'%(dataset))

        peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()

        fb_peak_df_train = pd.read_pickle('../data/peaks/%s_corrections_train.p'%(dataset))
        fb_peak_df_test = pd.read_pickle('../data/peaks/%s_corrections_test.p'%(dataset))

        fb_peak_df = pd.concat([fb_peak_df_train, fb_peak_df_test]).sort_index()

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
        colors = sns.color_palette('hls', nbins)
        t_ms = np.arange(win_start, win_stop, dt) * 1000
        fb_t_ms = np.arange(fb_win_start, fb_win_stop, dt) * 1000
        for j in range(n_co):
            fig = plt.figure(figsize=(15,5))
            axplot = fig.subplots(1,2)
            plt.suptitle('%s Controller %s'%(run_info[dataset]['name'],(j+1)))
            ymin, ymax = (np.min(X[:,j*win_size:(j+1)*win_size]), np.max(X[:,j*win_size:(j+1)*win_size]))

            #setting dimensions for inset
            left = .4#win_start + .8 * (win_stop-win_start)
            bottom = .7#ymin + .8 * (ymax-ymin)
            width = .05 #* (win_stop - win_start)
            height = .15 #* (ymax - ymin)
            inset = fig.add_axes([left, bottom, width, height], polar=True)
           
            fb_left = .8#win_start + .8 * (win_stop-win_start)
            fb_bottom = .7#ymin + .8 * (ymax-ymin)
            fb_width = .05 #* (win_stop - win_start)
            fb_height = .15 #* (ymax - ymin)
            fb_inset = fig.add_axes([fb_left, fb_bottom, fb_width, fb_height], polar=True)
            for i in range(-nbins//2, nbins//2):
                min_theta = i * bin_theta
                max_theta = (i+1) * bin_theta
                co_av = X[:,j*win_size:(j+1)*win_size][(theta > min_theta) & (theta <= max_theta)].mean(0)
                fb_co_av = fb_X[:,j*fb_win_size:(j+1)*fb_win_size][(fb_theta > min_theta) & (fb_theta <= max_theta)].mean(0) 

                color = colors[i+nbins//2]
                axplot[0].plot(t_ms, co_av, color=color)
                axplot[0].set_title('Initial')
                axplot[0].set_xlabel('Time From Movement (ms)')
                axplot[0].set_ylabel('Controller Value (a.u.)')
                inset.hist([(i+0.5) * bin_theta], [min_theta, max_theta], color=color)
                
                axplot[1].plot(fb_t_ms, fb_co_av, color=color)
                axplot[1].set_title('Corrective')
                axplot[1].set_xlabel('Time From Movement (ms)')
                axplot[1].set_ylabel('Controller Value (a.u.)')
                fb_inset.hist([(i+0.5) * bin_theta], [min_theta, max_theta], color=color)
                plt.savefig('../figures/final_figures/%s_controller_%s_averages.png'%(dataset,j+1))