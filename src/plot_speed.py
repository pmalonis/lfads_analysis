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
plt.rcParams['font.size'] = 20

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
spike_dt = 0.001

dataset = 'raju-M1-no-bad-trials'
example_filename = '../data/intermediate/%s.p'%dataset
example_trial = 3

run_info = yaml.safe_load(open(os.path.dirname(__file__) + '/../../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())

df = pd.read_pickle(example_filename)
trial_df = df.loc[example_trial]
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