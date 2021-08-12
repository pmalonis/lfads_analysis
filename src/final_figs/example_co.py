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
    example_trial = 8
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

    df = pd.read_pickle(example_filename)
    trial_df = df.loc[example_trial]
    t_targets = trial_df.kinematic.query('hit_target').index.values
    t = np.arange(co.shape[1]) * dt
    plt.figure(figsize=(12,6))
    plt.plot(t, co[example_trial,:, 0], 'r', alpha=.6)
    plt.text(0.7,0.4,"Controller 1", color=(1,0,0,0.6), fontsize=12)
    plt.plot(t, co[example_trial,:, 1], 'c', alpha=1)
    plt.text(0.7,-0.6,"Controller 2",  color=(0,.6,.6), fontsize=12)

    plt.xlabel('Time (s)')
    plt.ylabel('LFADS Controller Value (a.u.)')
    
    plt.xlim([0, trial_len])
    plt.ylim([-0.9, 0.6])
    ymin, ymax = plt.ylim() 
    plt.vlines(t_targets, ymin, ymax)
    plt.text(t_targets[1]+0.02, ymax*.85, "Target 1\nAcquired", fontsize=12)
    for i,t in enumerate(t_targets[2:-1]):
        plt.text(t+0.01, ymax*.95, "Target %d"%(i+2), fontsize=12)
    
    plt.yticks([-0.5, 0, 0.5])
    plt.savefig('../../figures/final_figures/example_co.png')