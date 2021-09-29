import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optimize_target_prediction as opt
import feedback_optimize_target_prediction as fotp
import utils
from scipy.stats import zscore
from scipy.signal import peak_widths
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
plt.rcParams['font.size'] = 20
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

lfads_filename = '../data/model_output/' + '_'.join([datasets[0], params[0], 'all.h5'])
with h5py.File(lfads_filename, 'r+') as h5file:
    co = h5file['controller_outputs'][:]
    
def plot_average(event_name, win_start, win_stop, event_label,
                title='', axes=None, label_plot=True):

    widths = []
    dset_names = []
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

        if dataset == "firstmove":
            plt.twiny()
        #co = savgol_filter(co, 11, 2, axis=1)
        co = zscore(co, axis=(0,1))
        #co = np.abs(co).sum(2, keepdims=True)
        n_co = co.shape[2]
        peak_df_train = pd.read_pickle('../data/peaks/%s_%s_train.p'%(dataset, event_name))
        peak_df_test = pd.read_pickle('../data/peaks/%s_%s_test.p'%(dataset, event_name))

        peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()

        X,y = opt.get_inputs_to_model(peak_df, co, trial_len, dt, df=df, 
                                    win_start=win_start, 
                                    win_stop=win_stop)

        win_size = int((win_stop - win_start)/dt)

        theta = np.arctan2(y[:,1], y[:,0])
        assert(nbins%2==0)
        
        bin_theta = np.pi / (nbins/2)
        colors = sns.color_palette('hls', nbins)
        t_ms = np.arange(win_start, win_stop, dt) * 1000
        abs_av = np.mean(np.abs(co).sum(2))
        
        co_mag_sum = np.zeros(win_size)
        for j in range(n_co):
            co_mag_sum += np.sum(np.abs(X[:,j*win_size:(j+1)*win_size]), axis=0)
            co_mag_av = co_mag_sum/X.shape[0] - abs_av

        if axes is None:
            plt.subplot(1, 3, dset_idx + 1)
        else:
            if isinstance(axes, list):
                plt.sca(axes[dset_idx])
            else:
                plt.sca(axes)
        
        plt.plot(t_ms, co_mag_av)
        width,_,_,_ = peak_widths(co_mag_av, [np.argmax(co_mag_av)])
        dset_name = run_info[dataset]['name']
        dset_names.append(dset_name)
        widths.append([int(width[0]*dt*1000), event_label, dset_name])
        #plt.title("Monkey %s"%dset_name)
        if label_plot == True:
            if dset_idx == 0:
                plt.ylabel('Controller Magnitude (Z-Score)')

        print("%s Peak: %f"%(dset_name,t_ms[np.argmax(co_mag_av)]))
        plt.xticks([0, 100, 200])
        # if 'raju' in dataset:
        #     plt.yticks([0, 0.2, 0.4])
        # else:
        #     plt.yticks([0, 1, 2, 3])

    if label_plot == True:
        #fig.text(0.4, 0.01, "Time from %s(ms)"%event_label, color=color)
        plt.xlabel("Time from %s(ms)"%event_label)

    plt.legend(["Monkey %s"%n for n in dset_names])
    width_df = pd.DataFrame(widths, columns=['Peak Width (ms)', 'Reference Event', 'Dataset'])
    plt.suptitle(title)
    
    return width_df

def label_figure(fig):
    return 

if __name__=='__main__':
    width_dfs = []
    # event_names = ['targets-one', "targets-not-one", "firstmove", "corrections"]
    # event_labels = ["Target Appearance", "Target Appearance",
    #                 "Initial Movement", "Corrective Movement"]
    event_names = ["corrections"]
    event_labels = ["Correction"]
    fig = plt.figure(figsize=(8,6))
    axes = fig.subplots(1,1)
    for event_idx, (event_name, event_label) in enumerate(zip(event_names, event_labels)):
        if "target" in event_name:
            win_start = 0.0
            win_stop = 0.3
        else:
            win_start = -0.4
            win_stop = 0

        width_df = plot_average(event_name, win_start, win_stop, event_label, axes=axes)
        if event_name in ["targets-not-one", "firstmove"]:
            width_dfs.append(width_df)

        #plt.text(-190, 0.015, 'Aligned to \n Target', color=colors[0], fontdict={'fontsize':16})
        #plt.text(-160, 0.009, 'Aligned to \n First Movement', color=colors[1], fontdict={'fontsize':16})
        #plt.locator_params(num_ticks=4)

    plt.savefig('../figures/final_figures/co_average_all_dir_%s.svg'%event_name)
    #plt.savefig('../figures/final_figures/numbered/4a.pdf')
    # plot_df = pd.concat(width_dfs)
    # sns.pointplot(data=plot_df, x='Reference Event', y='Peak Width (ms)', hue='Dataset')
    # plt.savefig('../figures/final_figures/co_av_widths.svg')