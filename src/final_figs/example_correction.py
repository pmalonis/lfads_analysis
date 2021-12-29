from scipy.stats import norm
import h5py
import pandas as pd
import sys
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = True
plt.rcParams['font.size'] = 18

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
spike_dt = 0.001
if __name__=='__main__':
    example_filename = '../../data/intermediate/rockstar.p'
    example_trial = 234
    example_filename = '../../data/intermediate/mack.p'
    example_trial = 43
    example_filename = '../../data/intermediate/raju-M1-no-bad-trials.p'
    example_trial = 121

    run_info = yaml.safe_load(open(os.path.dirname(__file__) + '/../../lfads_file_locations.yml', 'r'))
    datasets = list(run_info.keys())
    example_filenames = ['../../data/intermediate/%s.p'%dataset for dataset in datasets]
    example_trials = [234, 43, 121]
    example_filenames = [example_filenames[0]]
    example_trials = [45]
    for idx, (example_filename, example_trial) in enumerate(zip(example_filenames, example_trials)):
        df = pd.read_pickle(example_filename)
        trial_df = df.loc[example_trial].loc[:4]
        x_vel, y_vel = trial_df.kinematic[['x_vel','y_vel']].values.T
        speed = utils.get_speed(x_vel, y_vel)
        sigma_bin = cfg['rate_sigma_example_plot']
        bin_cutoff = sigma_bin*5 #cutoff kernal at 5 sigmas
        neural_kernel = norm.pdf(np.arange(-bin_cutoff, bin_cutoff), scale=sigma_bin)
        neural_kernel /= np.sum(neural_kernel)*spike_dt
        fr = np.apply_along_axis(np.convolve, 0, trial_df.neural.values, neural_kernel, mode='same')
        trial_pop_rate = fr.sum(1)
        t = trial_df.index.values
        t_targets = trial_df.kinematic.query('hit_target').index.values
        lns = []
        plt.figure(figsize=(18,6))
        lns += plt.plot(t, speed,'g')
        plt.ylabel('Cursor Speed (mm/s)')
        plt.xlabel('Time (s)')
        # plt.twinx()
        # lns += plt.plot(t, trial_pop_rate)
        # plt.ylabel('Population Firing Rate (spikes/s)')
        ymin, ymax = plt.ylim()    
        plt.vlines(t_targets[1:], ymin, ymax)
        plt.text(t_targets[1]+0.02, ymax*.915, "Target 1\nAcquired", fontsize=12)
        for i,t in enumerate(t_targets[2:-1]):
            plt.text(t+0.01, ymax*.95, "Target %d"%(i+2), fontsize=12)
        
        if idx == 0:
            plt.arrow(2.02, 45, .10, 40, head_width=.02, head_length=5)
            plt.text(1.65, 30, "Corrective Movement", fontsize=12)
        # plt.legend(handles=lns, labels=['Cursor Speed', 'Population Firing Rate'],
        #            bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lowerleft', ncol=3)
            plt.savefig('../../figures/final_figures/example_trial.png')
        plt.savefig('../../figures/final_figures/numbered/6a%d.png'%(idx+1))
        plt.savefig('../../figures/final_figures/numbered/6a%d.png'%(idx+1))
        plt.savefig('../../figures/final_figures/numbered/6a%d.png'%(idx+1))
    plt.show()