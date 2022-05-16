from scipy.stats import norm
import h5py
import pandas as pd
import sys
from scipy import io
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
import segment_submovements as ss
import utils
from importlib import reload
reload(ss)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 16

#removing any old pngs
for png in glob.glob(os.path.join(os.path.dirname(__file__), '../figures/speed_with_corrections/*.png')):
    os.remove(png)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
spike_dt = 0.001

run_info = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../lfads_file_locations.yml'), 'r'))
datasets = list(run_info.keys())
params = []
for dataset in run_info.keys():
    params.append(open(os.path.join(os.path.dirname(__file__), '../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric']))).read())

example_trials = [72, 287, 114]
for dset_idx, (dataset, param) in enumerate(zip(datasets, params)):
    data_filename = os.path.join(os.path.dirname(__file__), '../data/intermediate/' + dataset + '.p')
    lfads_filename = os.path.join(os.path.dirname(__file__), '../data/model_output/' + '_'.join([dataset, param, 'all.h5']))
    inputInfo_filename = os.path.join(os.path.dirname(__file__), '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat']))

    example_filename = os.path.join(os.path.dirname(__file__), '../data/intermediate/%s.p'%dataset)
    df = pd.read_pickle(example_filename)
    param = open(os.path.join(os.path.dirname(__file__), '../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric']))).read().strip()
    lfads_filename = os.path.join(os.path.dirname(__file__), '../data/model_output/' + '_'.join([dataset, param, 'all.h5']))
    inputInfo_filename = os.path.join(os.path.dirname(__file__), '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat']))
    input_info = io.loadmat(inputInfo_filename)
    with h5py.File(lfads_filename,'r') as h5file:
        co = h5file['controller_outputs'][:]
        dt = utils.get_dt(h5file, input_info)
        trial_len = utils.get_trial_len(h5file, input_info)

    fm = pd.read_pickle(os.path.join(os.path.dirname(__file__), '../data/peaks/%s_firstmove_all.p'%dataset))
    c = pd.read_pickle(os.path.join(os.path.dirname(__file__), '../data/peaks/%s_corrections_all.p'%dataset))

    #for example_trial in range(n_trials):
    example_trial = example_trials[dset_idx]

    trial_df = df.loc[example_trial]
    t_targets = trial_df.kinematic.query('hit_target').index.values
    t = np.arange(co.shape[1]) * dt
    if 'raju' in dataset:
        ss.plot_trajectory_co(trial_df, co[example_trial,:, 0], dt, co_max=0.5, co_min=-0.5)
    else:
        ss.plot_trajectory_co(trial_df, co[example_trial,:, 0], dt)

        # for plot_time in plot_times:
        #     plt.plot(*trial_df.kinematic[['x','y']].loc[plot_time:plot_time+.001].values[0], 'g.')

    if example_trial in fm.index.get_level_values('trial'):
        for plot_time in fm.loc[example_trial].loc[:trial_len].index:
            plt.plot(*trial_df.kinematic[['x','y']].loc[plot_time:plot_time+.001].values[0], color='m', marker='.',markersize=15)
        
    if example_trial in c.index.get_level_values('trial'):
        for plot_time in c.loc[example_trial].loc[:trial_len].index:
            plt.plot(*trial_df.kinematic[['x','y']].loc[plot_time:plot_time+.001].values[0], color='g', marker='.',markersize=15)

    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')

    plt.savefig(os.path.join(os.path.dirname(__file__), '../figures/final_figures/numbered/6b-%d.pdf'%(dset_idx+1)))