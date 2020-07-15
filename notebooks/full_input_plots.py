# %%
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
from power_law import get_power_law_deviation
from utils import get_indices

def angle_from_target(trial_df, trial_len):
    target_rows = trial_df.kinematic.query('hit_target')
    n_targets = trial_df.loc[:trial_len].kinematic.query('hit_target').shape[0]
    target_rows = target_rows.iloc[:n_targets + 1] #including target that isn't reached during the trial
    t_i = 0 #time of target
    j = 0
    n_samples = trial_df.loc[:trial_len].shape[0] #number of samples in trial
    angle = np.zeros(n_samples) #angle from hand velocity to target
    target_gen = target_rows.iterrows()
    next(target_gen)
    for t_ip1, row in target_gen:
        target_df = trial_df.loc[t_i:min(t_ip1-.0001,trial_len)].kinematic
        # truncating target samples for target that is present beyond the trial length
        if j + target_df.shape[0] <= n_samples:
            n = target_df.shape[0]
        else:
            n = n_samples - j

        #position of target in hand-centric coordinates
        target_x = row.x - target_df['x']
        target_y = row.y - target_df['y']
        r = np.sqrt(target_x**2 + target_y**2)
        try:
            angle[j:j+n] = r * np.arctan2(target_x - target_df['x_vel'].values, target_y - target_df['y_vel'])
        except:
            import pdb;pdb.set_trace()
        j += n
        t_i = t_ip1

    return angle

trial_type = 'all'

lfads_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_8QTVEk_%s.h5"%trial_type
data_filename = "/home/pmalonis/lfads_analysis/data/intermediate/rockstar.p"
inputInfo_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_inputInfo.mat"

input_info = loadmat(inputInfo_filename)
used_inds = get_indices(input_info, trial_type)

df = pd.read_pickle(data_filename)

with h5py.File(lfads_filename) as h5file:
    trial_len_ms = input_info['seq_timeVector'][-1][-1]
    dt = np.round(trial_len_ms/h5file['controller_outputs'].shape[1])/1000 
    trial_len = trial_len_ms/1000
    trial_len = np.floor(trial_len/dt)*dt
    lfads_t = np.arange(0, trial_len, dt)
    for i, trial_idx in enumerate(used_inds[:30]):
        data_t = df.loc[trial_idx].loc[:trial_len].index.values
        x = df.loc[trial_idx].loc[:trial_len].kinematic['x'].values
        y = df.loc[trial_idx].loc[:trial_len].kinematic['y'].values
        dev = get_power_law_deviation(x, y, data_t)
        angle = angle_from_target(df.loc[trial_idx], trial_len)
        x_vel = df.loc[trial_idx].loc[:trial_len].kinematic['x_vel'].values
        y_vel = df.loc[trial_idx].loc[:trial_len].kinematic['y_vel'].values
        x_accel = np.gradient(x_vel, data_t)
        y_accel = np.gradient(y_vel, data_t)
        speed = np.sqrt(x_vel**2 + y_vel**2)
        tang_accel = np.gradient(speed)
        x_jerk = np.gradient(x_accel, data_t)
        y_jerk = np.gradient(y_accel, data_t)
        jerk = np.sqrt(x_accel**2, y_accel**2)
        x_tang_velocity = x_vel/speed
        y_tang_velocity = y_vel/speed
        normal_accel = np.sqrt(np.gradient(x_tang_velocity, data_t)**2 + np.gradient(y_tang_velocity, data_t)**2)
        plt.figure(figsize=(12,4))
        plt.plot(data_t, angle, 'g')
        plt.xlabel("time (s)")
        plt.legend(['deviation'])
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        plt.title('trial%03d'%trial_idx)
        n_inputs = h5file['controller_outputs'].shape[2]
        legend=[]
        targets = df.loc[trial_idx].kinematic.loc[:trial_len].query('hit_target').index.values
        plt.vlines(targets, -.8,.8)
        for input_idx in range(n_inputs):
            ax2.plot(lfads_t, h5file['controller_outputs'][i,:,input_idx])
            legend.append('input %d'%input_idx)

        plt.legend(legend)

# %%
