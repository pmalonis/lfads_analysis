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
sys.path.insert(0, '..')
import segment_submovements as ss
import utils
from importlib import reload
reload(ss)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 16

#removing any old pngs
for png in glob.glob('../figures/speed_with_corrections/*.png'):
    os.remove(png)

config_path = '../config.yml'
cfg = yaml.safe_load(open(config_path, 'r'))
spike_dt = 0.001

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())
params = []
for dataset in run_info.keys():
    params.append(open('../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric'])).read())

for dset_idx, (dataset, param) in enumerate(zip(datasets, params)):
    data_filename = '../data/intermediate/' + dataset + '.p'
    lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
    inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])

    example_filename = os.path.dirname(__file__) + '../data/intermediate/%s.p'%dataset
    df = pd.read_pickle(example_filename)
    param = open(os.path.dirname(__file__)+'../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read().strip()
    lfads_filename = os.path.dirname(__file__)+'../data/model_output/' + \
                            '_'.join([dataset, param, 'all.h5'])
    inputInfo_filename = os.path.dirname(__file__)+ '../data/model_output/' + \
                    '_'.join([dataset, 'inputInfo.mat'])
    input_info = io.loadmat(inputInfo_filename)
    with h5py.File(lfads_filename,'r') as h5file:
        co = h5file['controller_outputs'][:]
        dt = utils.get_dt(h5file, input_info)
        trial_len = utils.get_trial_len(h5file, input_info)

    fm = pd.read_pickle('../data/peaks/%s_new-firstmove_all.p'%dataset)
    c = pd.read_pickle('../data/peaks/%s_new-corrections_all.p'%dataset)
    
    n_trials = df.index[-1][0] + 1
    #for example_trial in range(n_trials):
    for example_trial in range(50):
        trial_df = df.loc[example_trial]
        t_targets = trial_df.kinematic.query('hit_target').index.values
        t = np.arange(co.shape[1]) * dt
        ss.plot_trajectory_co(trial_df, co[example_trial,:, 0], dt)

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
        if example_trial%10 == 0:
            print('Plotted trajectory for %s trial %d'%(dataset,example_trial))
        plt.savefig('../figures/traj_co_with_corrections/%s_%03d.png'%(dataset,example_trial))
        plt.close()
    
    print('Creating PDF')
    os.system('convert ../figures/traj_co_with_corrections/%s_*.png ../figures/traj_co_with_corrections/%s_trajectory.pdf'%(dataset,dataset))
    for png in glob.glob('../figures/speed_with_corrections/*.png'):
        os.remove(png)