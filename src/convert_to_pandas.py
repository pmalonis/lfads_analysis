import numpy as np
import pandas as pd
from scipy import io
import argparse
import time
import gc
import os
import yaml

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

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

def raw_to_dataframe(data):
    '''Converts raw data from .mat file to pandas dataframe with processed kinematics'''

    win_size = cfg['preprocess']['win_size'] #samples to average over for moving average filter

    channels = [k for k in data.keys() if 'Chan' in k]
    
    x_norm = data['x'][:,1] - np.mean(data['x'][:,1])
    y_norm = data['y'][:,1] - np.mean(data['y'][:,1])

    ntrials = data['cpl_st_trial_rew'].shape[0]
    trial_dfs = []
    for i in range(ntrials):
        start = data['cpl_st_trial_rew'][i,0].real
        stop = data['cpl_st_trial_rew'][i,1].real

        data_idx = np.logical_and(data['x'][:,0] >= start, data['x'][:,0] < stop)
        
        t = upsample_2x(data['x'][data_idx,0])

        neural = dict()
        for channel in channels:
            neural[channel] = bin_trial(data[channel].real, t)

        hit_target = bin_trial(data['hit_target'].real, t)
        hit_target[0] = True
        
        #padding kinematics index for moving average filter
        pad = np.floor(win_size/2).astype(int)
        start_idx = np.where(data_idx)[0][0] - pad
        stop_idx = np.where(data_idx)[0][-1] + pad + 1

        x_raw = x_norm[start_idx:stop_idx]
        y_raw = y_norm[start_idx:stop_idx]

        win = np.ones(win_size)/win_size
        x_smooth = np.convolve(x_raw, win, 'valid')
        y_smooth = np.convolve(y_raw, win, 'valid')

        x = upsample_2x(x_smooth)
        y = upsample_2x(y_smooth)

        #removing last element because binning reduces length of psth
        t = t[:-1]
        x = x[:-1]
        y = y[:-1]

        x_vel = np.gradient(x, t)
        y_vel = np.gradient(y, t)

        trial_index = np.ones(len(t), dtype=int) * i

        t -= data['cpl_st_trial_rew'][i,0].real #making time relative to trial start

        index = pd.MultiIndex.from_arrays([trial_index, t], names=['trial','time'])

        kinematic_df = pd.DataFrame({'x':x, 'y':y, 
                                    'x_vel':x_vel, 'y_vel':y_vel, 
                                    'hit_target':hit_target}, index=index)
        
        neural_df = pd.DataFrame(neural, index=index)

        trial_df = pd.concat([kinematic_df, neural_df], axis=1, 
                            keys=['kinematic', 'neural'])

        trial_dfs.append(trial_df)
        if i%10==0:
            print("Processed trial %d of %d"%(i,ntrials))

    df = pd.concat(trial_dfs)

    return df

if __name__=='__main__':
    data = io.loadmat(snakemake.input[0])
    df = raw_to_dataframe(data)
    df.to_pickle(snakemake.output[0])