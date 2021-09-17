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
win_start = 0.0
win_stop = 0.3

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
        peak_df_train = pd.read_pickle('../data/peaks/%s_targets-not-one_train.p'%(dataset))
        peak_df_test = pd.read_pickle('../data/peaks/%s_targets-not-one_test.p'%(dataset))

        peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()

        X,y = opt.get_inputs_to_model(peak_df, co, trial_len, dt, df=df, 
                            win_start=win_start, 
                            win_stop=win_stop)
        # X = np.random.randn(1000,60)
        # y = np.random.randn(1000,2)

        win_size = int((win_stop - win_start)/dt)

        theta = np.arctan2(y[:,1], y[:,0])
        assert(nbins%2==0)
        
        bin_theta = np.pi / (nbins/2)
        colors = sns.color_palette('husl', nbins)
        t_ms = np.arange(win_start, win_stop, dt) * 1000
        fig, axes = plt.subplots(2,1)
        fig.set_size_inches((7,15))
        for j in range(n_co):
            plt.title('%s Inferred Input %s'%(run_info[dataset]['name'],2-j))
            ymin, ymax = (np.min(X[:,j*win_size:(j+1)*win_size]), np.max(X[:,j*win_size:(j+1)*win_size]))

            #setting dimensions for inset
            left = .85#win_start + .8 * (win_stop-win_start)
            bottom = .85#ymin + .8 * (ymax-ymin)
            width = .1 #* (win_stop - win_start)
            height = .2 #* (ymax - ymin)
            if j == 0:
                inset = fig.add_axes([left, bottom, width, height], polar=True)
                inset.set_rticks([])
                inset.set_xticks([])
                inset.set_yticks([])
                inset.text(-2.3, 2, 'Direction', fontsize=14)

            for i in range(-nbins//2, nbins//2):
                min_theta = i * bin_theta
                max_theta = (i+1) * bin_theta
                co_av = X[:,j*win_size:(j+1)*win_size][(theta > min_theta) & (theta <= max_theta)].mean(0)

                color = colors[i+nbins//2]
                plt.sca(axes[j])
                plt.plot(t_ms, co_av, color=color)

                if j == n_co - 1:
                    plt.xlabel('Time From Movement (ms)')
                elif j == 0:
                    plt.gca().set_xticklabels([])

                inset.hist([(i+0.5) * bin_theta], [min_theta, max_theta], color=color)

        fig.text(0.05,0.4, 'Inferred Input Value (a.u.)',ha='center',rotation='vertical')        
        plt.tight_layout(pad=3)
        plt.savefig('../figures/final_figures/%s_controller_averages_initial.svg'%(dataset)) 
        plt.savefig('../figures/final_figures/numbered/5a.svg')