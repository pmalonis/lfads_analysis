import numpy as np
import pandas as pd
import h5py
import os
import yaml
from scipy import io
from utils import get_indices

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def get_targets(_df, trial_len):
    n_targets = 6
    target_df = _df.loc[_df.index[0][0]].kinematic.query('hit_target')
    target_df['target_x'] = np.append(target_df['x'].iloc[1:].values, np.nan)
    target_df['target_y'] = np.append(target_df['y'].iloc[1:].values, np.nan)

    return target_df.iloc[1:n_targets].loc[:trial_len] #leaving out first target

if __name__=='__main__':
    # data_filename = snakemake.input[0]
    # lfads_filename = snakemake.input[1]
    # inputInfo_filename = snakemake.input[2]
    # output_filename = snakemake.output[0]
    #trial_type = snakemake.wildcards.trial_type

    data_filename = "../data/intermediate/mk08011M1m-mack-RTP.p"
    inputInfo_filename = "../data/model_output/mk08011M1m-mack-RTP_inputInfo.mat"
    lfads_filename = "../data/model_output/mk08011M1m-mack-RTP_2OLS24_all.h5"
    output_filename = "~/process_test.p"
    trial_type = "all"

    input_info = io.loadmat(inputInfo_filename)

    used_inds = get_indices(input_info, trial_type)

    df = pd.read_pickle(data_filename)
    n_neurons = df.loc[0].neural.shape[1]
    dt = 0.010 #TODO read from lfads file
    kin_dt = 0.001
    if cfg['align_peaks']:
        n_win = int((cfg['post_peak_win']*2)/dt)
    else:
        n_win = int((cfg['post_target_win_stop'] - cfg['post_target_win_start'])/dt)
        
    n_spikes = n_win * int(dt/kin_dt) #length of window of spikes to extract
    kinematic_vars = ['x', 'y']
    with h5py.File(lfads_filename,'r') as h5_file:
        trial_len = h5_file['controller_outputs'].shape[1] * dt
        
        # getting target locations
        
        processed_df['target_dist'] = np.sqrt((processed_df['x']-processed_df['target_x'])**2 + (processed_df['y']-processed_df['target_y'])**2)

        # removing targets that are closer to the end than the extent of the window
        processed_df.drop(processed_df[processed_df.index.get_level_values('time') > trial_len - cfg['post_target_win_stop']].index, inplace=True)

        input_peaks = np.zeros((processed_df.shape[0], h5_file['controller_outputs'].shape[2]))
        input_integral = np.zeros((processed_df.shape[0], h5_file['controller_outputs'].shape[2]))

        n_kinematic = int((cfg['peri_target_kinematics_stop'] - cfg['peri_target_kinematics_start'])/kin_dt) #length of window of kinematics to extract

        n_inputs = h5_file['controller_outputs'].shape[2]
        kinematics = np.zeros((processed_df.shape[0], len(kinematic_vars)*n_kinematic))

        spikes = np.zeros((processed_df.shape[0], n_spikes, n_neurons))

        input_sig = np.zeros((processed_df.shape[0], n_win*n_inputs))
        k = 0
        for i in range(h5_file['controller_outputs'].shape[0]):
            inputs = h5_file['controller_outputs'][i,:,:]
            target_times = processed_df.loc[used_inds[i]].index.values

            t = np.arange(0, trial_len, dt)
            for target in target_times:
                # dropping targets that are followed too closely by another target
                if k < len(processed_df.index) - 1:
                    next_target = processed_df.index[k+1][1]
                    if next_target - target < cfg['post_target_win_stop']:
                        processed_df.drop(processed_df.index[k], axis=0, inplace=True)
                        input_sig = np.delete(input_sig, k, axis=0)
                        input_integral = np.delete(input_integral, k, axis=0)
                        input_peaks = np.delete(input_peaks, k, axis=0)
                        kinematics = np.delete(kinematics, k, axis=0)
                        spikes = np.delete(spikes, k, axis=0)
                        continue

                idx = np.logical_and(t >= target + cfg['post_target_win_start'], t < target + cfg['post_target_win_stop'])
                for j in range(n_inputs):
                    peak = np.argmax(np.ma.masked_array(np.abs(inputs[:,j]), mask=~idx))
                    input_peaks[k,j] = inputs[peak,j]
                    integral_idx = np.arange(max(0, peak - cfg['integration_win_size']/2/dt), 
                                             min(trial_len/dt, peak + cfg['integration_win_size']/2/dt), dtype=int)
                    input_integral[k,j] = np.sum(inputs[integral_idx,j])*dt
                    try:
                        if cfg['align_peaks']:
                            peak = peak/(1/dt)
                            idx = np.logical_and(t >= np.round((peak - cfg['post_peak_win'])/dt)*dt, 
                                                       t < np.round((peak + cfg['post_peak_win'])/dt)*dt)
                            if peak - cfg['post_peak_win'] < 0:
                                input_sig[k,n_win*j:n_win*(j+1)] = np.concatenate((np.zeros(n_win - np.sum(idx)), inputs[idx, j]))
                            elif peak + cfg['post_peak_win'] >= trial_len:
                                input_sig[k,n_win*j:n_win*(j+1)] = np.concatenate((inputs[idx, j], np.zeros(n_win - np.sum(idx))))
                            else:
                                input_sig[k,n_win*j:n_win*(j+1)] = inputs[idx, j]                          
                        else:
                            input_sig[k,n_win*j:n_win*(j+1)] = inputs[idx, j]
                    except:
                        import pdb;pdb.set_trace()

                for i_var,v in enumerate(kinematic_vars):
                    kinematics[k,i_var*n_kinematic:(i_var+1)*n_kinematic] = df.loc[used_inds[i]].kinematic[v].loc[target+cfg['peri_target_kinematics_start']:].iloc[:n_kinematic]
                
                # getting spikes in same window as inputs            
                #spikes[k,:,:] = df.loc[k].neural.loc[t[idx][0]:].iloc[:n_spikes].values

                k += 1

    processed_df['peak_input_1'] = input_peaks[:,0]
    processed_df['peak_input_2'] = input_peaks[:,1]
    processed_df['integral_input_1'] = input_integral[:,0]
    processed_df['integral_input_2'] = input_integral[:,1]
    processed_df['input'] = input_sig.tolist()
    processed_df['kinematics'] = kinematics.tolist()
    #processed_df['spikes'] = spikes.tolist()

    processed_df.to_pickle(output_filename)