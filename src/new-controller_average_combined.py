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
plt.rcParams['font.size'] = 14
reload(opt)
reload(utils)


config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())
datasets = [datasets[0]]
params = []
for dataset in run_info.keys():
    params.append(open('../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read())

nbins = 12
fb_win_start = -0.3#0.00#-0.1#cfg['post_target_win_start']
fb_win_stop = 0.05 #0.3#0.1#cfg['post_target_win_stop']
win_start = -0.25
win_stop = 0.0

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
        peak_df_train = pd.read_pickle('../data/peaks/%s_new-firstmove_train.p'%(dataset))
        peak_df_test = pd.read_pickle('../data/peaks/%s_new-firstmove_test.p'%(dataset))

        peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()

        fb_peak_df_train = pd.read_pickle('../data/peaks/%s_new-corrections_train.p'%(dataset))
        fb_peak_df_test = pd.read_pickle('../data/peaks/%s_new-corrections_test.p'%(dataset))

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

        fig = plt.figure(figsize=(12,12))
        axplot = fig.subplots(2,2) 
        bin_theta = np.pi / (nbins/2)
        colors = sns.color_palette('husl', nbins)
        t_ms = np.arange(win_start, win_stop, dt) * 1000
        fb_t_ms = np.arange(fb_win_start, fb_win_stop, dt) * 1000


        for j in range(n_co):
            if j==0:              
                st=plt.suptitle('%s Inferred Input %s'%(run_info[dataset]['name'],(j+1)),fontsize=14)
            else:
                #average of minimum of previous plot and maximum of next plot, weighted towards the latter
                text_y = np.sum([0.75*axplot[j,0].get_position().ymax,
                                 0.25*axplot[j-1,0].get_position().ymin])
                plt.figtext(0.5, text_y, '%s Inferred Input %s'%(run_info[dataset]['name'],(j+1)),ha='center', fontsize=14)

            ymin, ymax = (np.min(X[:,j*win_size:(j+1)*win_size]), np.max(X[:,j*win_size:(j+1)*win_size]))
            if j == 0:
                fb_left = .82 #win_start + .8 * (win_stop-win_start)
                fb_bottom = .82 #ymin + .8 * (ymax-ymin)
                fb_width = .05 #* (win_stop - win_start)
                fb_height = .15 #* (ymax - ymin)
                fb_inset = fig.add_axes([fb_left, fb_bottom, fb_width, fb_height], polar=True)
                fb_inset.set_rticks([])
                fb_inset.set_xticks([])
                fb_inset.set_yticks([])
                fb_inset.text(-2.3, 2, 'Direction', fontsize=14)

                axplot[j,0].set_title('Initial')
                axplot[j,1].set_title('Corrective')

                plt.gca().set_xticklabels([])


            for i in range(-nbins//2, nbins//2):
                min_theta = i * bin_theta
                max_theta = (i+1) * bin_theta
                co_av = X[:,j*win_size:(j+1)*win_size][(theta > min_theta) & (theta <= max_theta)].mean(0)
                fb_co_av = fb_X[:,j*fb_win_size:(j+1)*fb_win_size][(fb_theta > min_theta) & (fb_theta <= max_theta)].mean(0) 

                color = colors[i+nbins//2]
                fb_inset.hist([(i+0.5) * bin_theta], [min_theta, max_theta], color=color)
        
                axplot[j,0].plot(t_ms, co_av, color=color)
                axplot[j,1].plot(fb_t_ms, fb_co_av, color=color)

            # if j == 0:
            #     fig.text(0.9, 0.25, 'RS Input %d'%j)
            # else:
            #     fig.text(0.9, 0.75, 'RS Input %d'%j)

            # axplot[j,0].set_yticks([-0.2, 0, 0.2])
            # axplot[j,1].set_yticks([-0.2, 0, 0.2])

        fig.text(0.05,0.4, 'Inferred Input Value (a.u.)',ha='center',rotation='vertical')        
        fig.text(0.5, 0.03, 'Time from Movement (ms)',ha='center')
        #plt.tight_layout()     
        fig.subplots_adjust(right=.88)   
        plt.savefig('../figures/final_figures/%s_controller_averages_initial_corrections.svg'%(dataset))
        plt.savefig('../figures/final_figures/numbered/7a.svg')