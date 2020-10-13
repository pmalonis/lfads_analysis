import numpy as np
import pandas as pd
from scipy import io
import argparse
import time
import gc
import os
import yaml
import scipy.signal as sg

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def filter_kinematics(x_raw, y_raw):
    #TODO: try sosfilt, see if it obviates need to filter both velocity and position
    b,a = sg.butter(cfg['preprocess']['filter_order'], cfg['preprocess']['cutoff'], fs=500) #500 kinematic frequency of sampling
    x_smooth = sg.filtfilt(b, a, x_raw)
    y_smooth = sg.filtfilt(b, a, y_raw)

    return x_smooth, y_smooth

def upsample_2x(s):
    '''Upsample a time series by a factor of 2'''
    out = np.zeros(len(s) * 2 - 1)
    out[::2] = s
    out[1::2] = (out[:-2:2] + out[2::2])/2

    return out

def bin_trial(s, t):
    '''Bins trial data in time. Returns boolean array (assumes only 1 event per bin)'''
    
    trial_idx = np.logical_and(s >= t[0], s < t[-1])
    trial_spikes = s[trial_idx]
    out,_ = np.histogram(trial_spikes, bins=t)

    return out.astype(bool)

def get_area(mat_data):
    '''returns list of area for each neuron. neurons are '''
    areas = ['MI','PMd']
    neurons = [k for k in mat_data.keys() if 'Chan' in k]
    neuron_area = []

    for neuron in neurons:
        this_neuron_area = ''
        for area in areas:
            if np.any(['Chan%03d'%c in neuron for c in mat_data['%schans'%area].flatten()]):
                this_neuron_area = area
                break

        neuron_area.append(this_neuron_area)

    neuron_area[neuron_area=='MI'] = 'M1'
    return neuron_area

def raw_to_dataframe(data, input_info):
    '''Converts raw data from .mat file to pandas dataframe with processed kinematics. 
    
    *Only includes trials used in LFADS run*'''

    win_size = cfg['preprocess']['win_size'] #samples to average over for moving average filter

    channels = [k for k in data.keys() if 'Chan' in k]
    
    x_norm = data['x'][:,1] - np.mean(data['x'][:,1])
    y_norm = data['y'][:,1] - np.mean(data['y'][:,1])

    trial_len = input_info['seq_timeVector'][-1][-1]/1000

    ntrials = data['cpl_st_trial_rew'].shape[0]
    trial_dfs = []
    used_trial_counter = 0
    for i in range(ntrials):
        start = data['cpl_st_trial_rew'][i,0].real
        stop = data['cpl_st_trial_rew'][i,1].real 
        if stop - start < trial_len:
            continue

        stop += cfg['preprocess']['post_trial_pad']

        data_idx = np.logical_and(data['x'][:,0] >= start, data['x'][:,0] < stop)
        
        t = upsample_2x(data['x'][data_idx,0])

        neural = dict()
        for channel in channels:
            neural[channel] = bin_trial(data[channel].real, t)

        hit_target = bin_trial(data['hit_target'].real, t)
        hit_target[0] = True

        x_raw = x_norm[data_idx]
        y_raw = y_norm[data_idx]

        x_smooth, y_smooth = filter_kinematics(x_raw, y_raw)

        x = upsample_2x(x_smooth)
        y = upsample_2x(y_smooth)

        #removing last element because binning reduces length of psth
        t = t[:-1]
        x = x[:-1]
        y = y[:-1]

        x_vel = np.gradient(x, t)
        y_vel = np.gradient(y, t)

        # filtering velocity, again
        x_vel, y_vel = filter_kinematics(x_vel, y_vel)

        trial_index = np.ones(len(t), dtype=int) * used_trial_counter

        t -= data['cpl_st_trial_rew'][i,0].real #making time relative to trial start

        index = pd.MultiIndex.from_arrays([trial_index, t], names=['trial','time'])

        kinematic_df = pd.DataFrame({'x':x, 'y':y, 
                                    'x_vel':x_vel, 'y_vel':y_vel, 
                                    'hit_target':hit_target}, index=index)
        
        neural_df = pd.DataFrame(neural, index=index)

        trial_df = pd.concat([kinematic_df, neural_df], axis=1, 
                            keys=['kinematic', 'neural'])

        trial_dfs.append(trial_df)

        used_trial_counter += 1
        if i%10==0:
            print("Processed trial %d of %d"%(i,ntrials))

    df = pd.concat(trial_dfs)

    return df


if __name__=='__main__':
    data = io.loadmat(snakemake.input[0])
    input_info = io.loadmat(snakemake.input[1])
    df = raw_to_dataframe(data, input_info)
    df.to_pickle(snakemake.output[0])