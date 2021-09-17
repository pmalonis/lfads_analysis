from scipy.stats import norm
import h5py
import pandas as pd
import sys
from scipy import io
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '..')
from segment_submovements import plot_trajectory_co
import utils
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 16

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
spike_dt = 0.001
if __name__=='__main__':
    dataset = 'rockstar'
    example_filename = os.path.dirname(__file__) + '/../../data/intermediate/%s.p'%dataset
    example_trial = 72
    param = open(os.path.dirname(__file__)+'/../../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read().strip()
    lfads_filename = os.path.dirname(__file__)+'/../../data/model_output/' + \
                            '_'.join([dataset, param, 'all.h5'])
    inputInfo_filename = os.path.dirname(__file__)+'/../../data/model_output/' + \
                    '_'.join([dataset, 'inputInfo.mat'])
    input_info = io.loadmat(inputInfo_filename)
    with h5py.File(lfads_filename) as h5file:
        co = h5file['controller_outputs'][:]
        dt = utils.get_dt(h5file, input_info)
        trial_len = utils.get_trial_len(h5file, input_info)

    fm = pd.read_pickle('../../data/peaks/%s_new-firstmove_all.p'%dataset)
    c = pd.read_pickle('../../data/peaks/%s_new-corrections_all.p'%dataset)

    df = pd.read_pickle(example_filename)
    trial_df = df.loc[example_trial]
    t_targets = trial_df.kinematic.query('hit_target').index.values
    t = np.arange(co.shape[1]) * dt
    plot_trajectory_co(trial_df, co[example_trial,:, 0], dt)
    
    plot_times = []
    if example_trial in fm.index.get_level_values('trial'):
        plot_times = np.concatenate([plot_times,fm.loc[example_trial].loc[:trial_len].index.values])
    
    if example_trial in c.index.get_level_values('trial'):
        plot_times = np.concatenate([plot_times, c.loc[example_trial].loc[:trial_len].index.values])

    # for plot_time in plot_times:
    #     plt.plot(*trial_df.kinematic[['x','y']].loc[plot_time:plot_time+.001].values[0], 'g.')

    if example_trial in fm.index.get_level_values('trial'):
        for plot_time in fm.loc[example_trial].loc[:trial_len].index:
            plt.plot(*trial_df.kinematic[['x','y']].loc[plot_time:plot_time+.001].values[0], 'r.', markersize=10)
        
    if example_trial in c.index.get_level_values('trial'):
        for plot_time in c.loc[example_trial].loc[:trial_len].index:
            plt.plot(*trial_df.kinematic[['x','y']].loc[plot_time:plot_time+.001].values[0], 'b.', markersize=10)
    

    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    
    #plt.savefig('../../figures/final_figures/example_traj_co.svg')
    #plt.savefig('../../figures/final_figures/numbered/6b.svg')