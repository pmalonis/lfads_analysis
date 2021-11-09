from scipy.stats import norm
import h5py
import pandas as pd
import sys
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '..')
import utils
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 18

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
spike_dt = 0.001
if __name__=='__main__':
    example_filename = '../../data/intermediate/mack.p'
    example_trial = 43
    df = pd.read_pickle(example_filename)
    trial_df = df.loc[example_trial]
    offset_ms = np.round(cfg['neural_offset']/0.001).astype(int)
    kin = trial_df.kinematic.iloc[:-offset_ms]
    x_vel, y_vel = kin[['x_vel','y_vel']].values.T
    speed = utils.get_speed(x_vel, y_vel)
    sigma_bin = cfg['rate_sigma_example_plot']
    bin_cutoff = sigma_bin*5 #cutoff kernal at 5 sigmas
    neural_kernel = norm.pdf(np.arange(-bin_cutoff, bin_cutoff), scale=sigma_bin)
    neural_kernel /= np.sum(neural_kernel)*spike_dt
    spikes = trial_df.neural.iloc[offset_ms:].values
    fr = np.apply_along_axis(np.convolve, 0, spikes, neural_kernel, mode='same')
    trial_pop_rate = fr.sum(1)
    t = trial_df.iloc[:-offset_ms].index.values
    t_targets = trial_df.kinematic.query('hit_target').index.values
    lns = []
    plt.figure(figsize=(18,6))
    lns += plt.plot(t, speed,'g')
    plt.ylim([0,1000])
    plt.ylabel('Cursor Speed (mm/s)')
    plt.xlabel('Time (s)')
    plt.twinx()
    lns += plt.plot(t, trial_pop_rate)
    plt.ylabel('Population Firing Rate (spikes/s)')
    ymin, ymax = plt.ylim()    
    plt.vlines(t_targets[1:], ymin, ymax)
    plt.text(t_targets[1]+0.02, ymax*.915, "Target 1\nAcquired", fontsize=12)
    for i,t in enumerate(t_targets[2:-1]):
        plt.text(t+0.01, ymax*.95, "Target %d"%(i+2), fontsize=12)

    # plt.legend(handles=lns, labels=['Cursor Speed', 'Population Firing Rate'],
    #            bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lowerleft', ncol=3)
    
    plt.text(2.6, 450, 'Cursor Speed', color=lns[0].get_color(), fontsize=16)
    plt.text(2.5, 925, 'Population\nFiring Rate', color=lns[1].get_color(), fontsize=16)

    plt.savefig('../../figures/final_figures/example_trial.svg')
    plt.savefig('../../figures/final_figures/numbered/1c.pdf')
    plt.show()