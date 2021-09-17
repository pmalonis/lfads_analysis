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
from matplotlib.patches import Rectangle,Circle
plt.rcParams['axes.spines.top'] = True

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
spike_dt = 0.001
target_size = 10 #target size in millimeters
cursor_size = 3
if __name__=='__main__':
    dataset = 'rockstar'
    example_filename = os.path.dirname(__file__) + '/../../data/intermediate/%s.p'%dataset
    example_trial = 22
    param = open(os.path.dirname(__file__)+'/../../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read().strip()
    
    inputInfo_filename = os.path.dirname(__file__)+'/../../data/model_output/' + \
                    '_'.join([dataset, 'inputInfo.mat'])
    input_info = io.loadmat(inputInfo_filename)

    df = pd.read_pickle(example_filename)
    trial_df = df.loc[example_trial]
    target_df = trial_df.kinematic.query('hit_target')
    fig, ax = plt.subplots()
    for idx, (t, target) in enumerate(target_df.iterrows()):
        if idx > 5: break # only some targets so plot isn't messy

        if idx == 0: #starting postition, draw cursor
            ax.add_patch(Circle((target['x'], target['y']), cursor_size, color='r', zorder=2))
        else:
            t = target_df.index[idx]
            t_prev = target_df.index[idx-1]
            ax.plot(*trial_df.kinematic[['x','y']].loc[t_prev:t].values.T, zorder=1)
            ax.add_patch(Rectangle((target['x']-target_size/2, target['y']-target_size/2), 
                        target_size, target_size, zorder=2))

    ax.set_aspect('equal', adjustable='box')
    fig.set_size_inches((7,6))
    all_target_df = df.kinematic.query('hit_target')
    plt.xlim([all_target_df['x'].min(), all_target_df['x'].max()])
    plt.ylim([all_target_df['y'].min(), all_target_df['y'].max()])
    plt.xticks([])
    plt.yticks([])
    #plt.xlabel('X (mm)')
    #plt.ylabel('Y (mm)')
    #plt.show()
    plt.savefig('../../figures/final_figures/example_trajectory.svg')
    plt.savefig('../../figures/final_figures/numbered/1b.svg')