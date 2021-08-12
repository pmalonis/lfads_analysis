import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optimize_target_prediction as opt
import feedback_optimize_target_prediction as fotp
import utils
from scipy.signal import peak_widths
from scipy.stats import zscore
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
plt.rcParams['font.size'] = 16
reload(opt)
reload(utils)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())
params = []
for dataset in run_info.keys():
    params.append(open('../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read())

#datasets = ['rockstar', 'mack']#['rockstar','raju', 'mack']
#params = ['all-early-stop-kl-sweep-yKzIQf', 'all-early-stop-kl-sweep-bMGCVf']#['mack-kl-co-sweep-0Wo8i9']#['final-fixed-2OLS24', 'final-fixed-2OLS24', 'mack-kl-co-sweep-0Wo8i9']
nbins = 12
fb_win_start = -0.3#0.00#-0.1#cfg['post_target_win_start']
fb_win_stop = 0.0 #0.3#0.1#cfg['post_target_win_stop']
win_start = -0.0
win_stop = 0.3

lfads_filename = '../data/model_output/' + '_'.join([datasets[0], params[0], 'all.h5'])
with h5py.File(lfads_filename, 'r+') as h5file:
    co = h5file['controller_outputs'][:]
    
n_co = co.shape[2]

if __name__=='__main__':
    widths = []
    for first_event in ["targets", "firstmove"]:
        fig1 = plt.figure(figsize=(15,6))
        fig2 = plt.figure(figsize=(15,6))
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
            co = zscore(co)
            peak_df_train = pd.read_pickle('../data/peaks/%s_%s_train.p'%(dataset, first_event))
            peak_df_test = pd.read_pickle('../data/peaks/%s_%s_test.p'%(dataset, first_event))

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
            abs_av = np.mean(np.abs(co).sum(2))
            for event_idx, (t, co_data) in enumerate(zip([t_ms, fb_t_ms], [X, fb_X])):
                co_mag_sum = np.zeros(win_size)
                for j in range(n_co):
                    co_mag_sum += np.sum(np.abs(co_data[:,j*win_size:(j+1)*win_size]), axis=0)

                co_mag_av = co_mag_sum/X.shape[0] - abs_av
                plt.figure(event_idx+1)
                plt.subplot(1, 3, dset_idx + 1)
                plt.plot(t, co_mag_av)
                if event_idx == 0:
                    width = peak_widths(co_mag_av, np.argmax(co_mag_av))
                    widths.append([width, first_event, dataset])
                if dset_idx == 0:
                    plt.ylabel('Controller Magnitude (Z-Score)')
                # if event_idx == 0:
                #     if dset_idx != 2:
                #         plt.ylim([-0.5, 1.5])
                #         plt.yticks([0, 0.5 ,1])
                # else:
                #     if dset_idx != 2:
                #         plt.ylim([-1.15, -1.05])
                #         plt.yticks([-1.15, -1.10, -1.05])
                #     else:
                #         plt.ylim([-0.55, -0.3])
                #         plt.yticks([-0.5, -0.4, -0.3])
        fig = plt.figure(1)
        fig.text(0.4, 0.01, "Time from Target Appearance (ms)")
        fig = plt.figure(2)
        fig.text(0.4, 0.01, "Time from Correction (ms)")
        #fig.text(0.0, 0.7, "First\nMovement")
        #fig.text(0.0, 0.3, "Corrections")

        fig1.savefig('../figures/final_figures/co_average_all_dir_firstmove.png')
        fig2.savefig('../figures/final_figures/co_average_all_dir_corrections.png')
        plt.figure()
        plot_df = pd.DataFrame(widths, columns=['Width', 'Reference', 'Dataset'])
        sns.pointplot(plot_df)