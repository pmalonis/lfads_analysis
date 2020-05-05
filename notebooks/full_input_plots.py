# %%
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt

lfads_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_kuGTbO_valid.h5"
data_filename = "/home/pmalonis/lfads_analysis/data/intermediate/rockstar.p"
inputInfo_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_kuGTbO_inputInfo.mat"

input_info = loadmat(inputInfo_filename)
used_inds = input_info['validInds'][0] - 1

df = pd.read_pickle(data_filename)

with h5py.File(lfads_filename) as h5file:
    trial_len_ms = input_info['seq_timeVector'][-1][-1]
    dt = np.round(trial_len_ms/h5file['controller_outputs'].shape[1])/1000 
    trial_len = trial_len_ms/1000
    trial_len = np.floor(trial_len/dt)*dt
    lfads_t = np.arange(0, trial_len, dt)
    for i, trial_idx in enumerate(used_inds):
        plt.figure(figsize=(10,6))
        data_t = df.loc[trial_idx].loc[:trial_len].index.values
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
        plt.plot(data_t, normal_accel, 'g')
        plt.legend(['normal acceleration'])
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        n_inputs = h5file['controller_outputs'].shape[2]
        legend=[]
        targets = df.loc[trial_idx].kinematic.loc[:trial_len].query('hit_target').index.values
        plt.vlines(targets, -.8,.8)
        for input_idx in range(n_inputs):
            ax2.plot(lfads_t, h5file['controller_outputs'][i,:,input_idx])
            legend.append('input %d'%input_idx)

        plt.legend(legend)

# %%