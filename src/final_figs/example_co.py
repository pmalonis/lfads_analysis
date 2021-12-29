from scipy.stats import norm
import h5py
import pandas as pd
import sys
from scipy import io
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 20

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
spike_dt = 0.001

if __name__=='__main__':
    #dataset = 'rockstar'
    #example_trial = 8
    datasets = ['rockstar', 'mack', 'raju-M1-no-bad-trials']
    example_trials = [8, 254, 142]
    yticks = [0.5, 0.5, 0.25]
    ymins = [-0.7, -0.7, -0.3]
    ymaxs = [0.7, 0.7, 0.3]
    label_ctrl_1 = [(0.7, 0.4), (0.57, -0.5), (0.64, 0.15)]
    label_ctrl_2 = [(0.7, -0.6), (0.52, 0.2), (1.1, -0.12)]
    panels = ['a', 'b', 'c']
    for i, (dataset, example_trial, lc1, lc2, ytick) in enumerate(zip(datasets, example_trials, label_ctrl_1, label_ctrl_2, yticks)):
        example_filename = os.path.join(os.path.dirname(__file__), '../../data/intermediate/%s.p'%dataset)
        param_filename = os.path.join(os.path.dirname(__file__), 
                    '../../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric']))
        param = open(param_filename).read().strip()
        lfads_filename = os.path.join(os.path.dirname(__file__), '../../data/model_output/' + \
                                '_'.join([dataset, param, 'all.h5']))
        inputInfo_filename = os.path.join(os.path.dirname(__file__),'../../data/model_output/' + \
                            '_'.join([dataset, 'inputInfo.mat']))
        input_info = io.loadmat(inputInfo_filename)
        with h5py.File(lfads_filename) as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        df = pd.read_pickle(example_filename)
        trial_df = df.loc[example_trial]
        t_targets = trial_df.loc[:trial_len].kinematic.query('hit_target').index.values
        t = np.arange(co.shape[1]) * dt
        plt.figure(figsize=(12,6))
        colors = utils.contrasting_colors(**cfg['colors']['controller_example'])
        plt.plot(t, co[example_trial,:, 0], color=colors[0])
        plt.text(*lc1, "Controller 1", color=colors[0], fontsize=16)
        plt.plot(t, co[example_trial,:, 1], color=colors[1])
        plt.text(*lc2, "Controller 2",  color=colors[1], fontsize=16)

        if co.shape[2] == 3:
            color = [0.5, 0.0, 0.5]
            plt.plot(t, co[example_trial,:, 2], color=color)
            plt.text(0.645, -0.25, 'Controller 3', color=color, fontsize=16)

        plt.xlabel('Time (s)')
        plt.ylabel('LFADS Controller Value (a.u.)')
        
        plt.xlim([0, trial_len])
        ymin, ymax = ymins[i], ymaxs[i]
        #plt.ylim([ymin, ymax])
        #ymin, ymax = plt.ylim() 
        plt.vlines(t_targets, ymin, ymax)
        plt.text(t_targets[1]+0.02, ymax*.85, "Target 1\nAcquired", fontsize=12)
        for j,t in enumerate(t_targets[2:]):
            plt.text(t+0.01, ymax*.95, "Target %d"%(j+2), fontsize=16)
        
        plt.yticks([-ytick, 0, ytick])
        fig_filename = os.path.join(os.path.dirname(__file__), 
                '../../figures/final_figures/numbered/3%s.pdf'%panels[i])
        plt.savefig(fig_filename)