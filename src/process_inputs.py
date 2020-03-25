import numpy as np
import pandas as pd
import h5py
import os
import yaml
from scipy import io

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def get_targets(_df):
    n_targets = 6
    target_df = _df.loc[_df.index[0][0]].kinematic.query('hit_target')

    target_df['target_x'] = np.append(target_df['x'].iloc[1:].values, np.nan)
    target_df['target_y'] = np.append(target_df['y'].iloc[1:].values, np.nan)

    return target_df.iloc[1:n_targets] #leaving out first target

def get_input_peaks(data_filename, valid_filename, inputInfo_filename):
    input_info = io.loadmat(inputInfo_filename)

    #subtracting to convert to 0-based indexing
    train_inds = input_info['trainInds'][0] - 1
    valid_inds = input_info['validInds'][0] - 1

    df = pd.read_pickle(data_filename)
    dt = 0.010 #TODO read from lfads file
    
    with h5py.File(valid_filename,'r') as h5_file:
        trial_len = h5_file['controller_outputs'].shape[1] * dt
        processed_df = df.loc[valid_inds].groupby('trial').apply(lambda _df: get_targets(_df).loc[:trial_len])
        processed_df['target_dist'] = np.sqrt((processed_df['x']-processed_df['target_x'])**2 + (processed_df['y']-processed_df['target_y'])**2)
        input_peaks = np.zeros((processed_df.shape[0], h5_file['controller_outputs'].shape[2]))
        input_integral = np.zeros((processed_df.shape[0], h5_file['controller_outputs'].shape[2]))
        k = 0
        for i in range(h5_file['controller_outputs'].shape[0]):
            inputs = h5_file['controller_outputs'][i,:,:]
            target_times = processed_df.loc[valid_inds[i]].index.values
            t = np.arange(0, trial_len, dt)
            for target in target_times:
                idx = np.logical_and(t >= target + cfg['post_target_win_start'], t < target + cfg['post_target_win_stop'])
                for j in range(inputs.shape[1]):
                    peak = np.argmax(np.ma.masked_array(np.abs(inputs[:,j]), mask=~idx))
                    input_peaks[k,j] = inputs[peak,j]
                    integral_idx = np.arange(max(0, peak - cfg['integration_win_size']/2/dt), 
                                             min(trial_len/dt, peak + cfg['integration_win_size']/2/dt), dtype=int)
                    input_integral[k,j] = np.sum(inputs[integral_idx,j])*dt
                k += 1
    
    processed_df['peak_input_1'] = input_peaks[:,0]
    processed_df['peak_input_2'] = input_peaks[:,1]
    processed_df['integral_input_1'] = input_integral[:,0]
    processed_df['integral_input_2'] = input_integral[:,1]

    return processed_df


if __name__=='__main__':
    
    data_filename = "data/intermediate/rockstar.p"
    inputInfo_filename = "data/model_output/rockstar_inputInfo.mat"
    valid_filename = "data/model_output/rockstar_valid.h5"
    output_filename = "data/model_output/rockstar_input_pulses.p"
