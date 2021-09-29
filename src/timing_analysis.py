import os
import yaml
import numpy as np
import h5py
import pandas as pd
from scipy import io
import utils
from scipy import signal

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def get_targets(df, used_inds=None):
    '''
    Gets rows of data corresponding to time of
    hitting target

    Parameters
    df: pandas dataframe containing experiemnt data
    used_inds: (default: None) If given, will only use targets from
    given indices

    Returns
    targets: pandas dataframe containing rows of df corresponding to 
    target appearanes
    '''
    if used_inds is not None:
        targets = df.loc[used_inds].kinematic.query('hit_target')
    else:
        targets = df.kinematic.query('hit_target')

    return targets

def get_peaks(co, dt, min_height, relative=False, min_distance=cfg['min_peak_spacing'], 
                exclude_post_target=None, df=None):
    '''
    Returns times of peaks of controller outputs

    Parameters
    co: LFADS controller outputs. 3D numpy array with dimensions
    trials X time X controller output

    dt: time in between samples of controller outputs 

    min_height: minimum height of peak to consider, in number of standard deviations of absolute value above the mean absolute value. 
    Either scalar or vector with length equal to co.shape[2]. The later sets a separate 
    height treshold for each controller output

    min_distance: minimum distance to neighboring peak, as given to 
    scipy.signal.find_peaks as the "distance" argument. As with min_height, 
    can be a float or a vector. the unit of the distance is the index ste
    (1ms for the processed dataframes)

    Returns:
    peaks: object array of shape (trial X outputs). Each entry is a 
    lists of floats representing time in trial, in seconds, of each peak 
    in controller output that meet specifications given by min_height
    and min_distance
    peak_vals: 
    '''
    peaks = np.empty(shape=(co.shape[0], co.shape[2]), dtype='object')
    # filling array with empty lists. The default object is None, but empty list is simpler 
    # to work with since it can be pass to dataframe as slice with no error
    peaks.fill([])
    t_lfads = np.arange(co.shape[1]) * dt #time labels of lfads input
    abs_co = np.abs(co-co.mean((0,1)))
    if isinstance(min_height, (int, float)):
        min_height = np.ones(co.shape[2]) * min_height

    for trial_idx in range(co.shape[0]):
        for input_idx in range(co.shape[2]):
            if isinstance(min_distance, int):
                distance_arg = min_distance
            elif isinstance(min_distance, list):
                distance_arg = min_distance[input_idx]

            if relative:
                height_arg = abs_co[:,:,input_idx].mean() + min_height[input_idx]*abs_co[:,:,input_idx].std()
                p, _ = signal.find_peaks(abs_co[trial_idx, :, input_idx], 
                                height=height_arg, distance=distance_arg)
            else:
                p, _ = signal.find_peaks(abs_co[trial_idx, :, input_idx], 
                                height=min_height[input_idx], distance=distance_arg)


            if (exclude_post_target is not None) and (df is not None): #TODO used_indx
                times = []
                for i in range(len(p)):
                    t = t_lfads[p[i]]
                    t_targets = df.loc[trial_idx].kinematic.query('hit_target').index
                    if np.any((t - t_targets < exclude_post_target) & (t - t_targets > 0)):
                        continue
                    else:
                        times.append(t)
                times = np.array(times)
            else:
                times = t_lfads[p]

            peaks[trial_idx, input_idx] = times
 
    return peaks

def get_maximum_peaks(co, dt, df, firstmove_df, min_peak_time=0.05, min_firstmove_time=0.1):
    '''
    Returns times of maximum in controller output prior to first movement

    Parameters
    co: LFADS controller outputs. 3D numpy array with dimensions
    trials X time X controller output

    dt: time in between samples of controller outputs 

    df: dataframe containing preprocessed trial data

    post_target: window after target to look for maximum

    Returns:
    peaks: object array of shape (trial X outputs). Each entry is a 
    lists of floats representing time in trial, in seconds, of each peak 
    in controller output that meet specifications given by min_height
    and min_distance
    peak_vals: 
    '''

    t_lfads = np.arange(co.shape[1]) * dt #time labels of lfads input
    co = signal.savgol_filter(co, 11,2, axis=1)
    abs_co = np.abs(co)
    
    trial_len = co.shape[1] * dt
    output_df = get_targets(df)
    n_inputs = co.shape[2]
    n_trials = co.shape[0]
    target_latency = []
    firstmove_latency = []
    k = 0
    for trial_idx in range(n_trials):
        t_firstmoves = firstmove_df.loc[trial_idx].index.values

        trial_target_latency = [] # list of lists of latencies, one list for each input
        trial_firstmove_latency = [] # list of lists of latencies, one list for each input
        for input_idx in range(n_inputs):
            t_targets = output_df.loc[trial_idx].loc[:trial_len].index.values
            t_targets = np.append(t_targets, t_lfads[-1])
            input_target_latency = []
            input_firstmove_latency = []
            for t_target, t_next_target in zip(t_targets[:-1], t_targets[1:]):
                t_firstmove = t_firstmoves[np.argmax(t_firstmoves > t_target)] # getting initial movement after target
                if t_firstmove >= t_next_target or t_firstmove - t_target < min_firstmove_time: #if no first movement before next target continue
                    output_df.drop(index=(trial_idx, t_target), inplace=True)
                    continue

                assert np.any(t_lfads > t_target)
                assert np.any(t_lfads > t_firstmove)
                idx_target = np.argmax(t_lfads > t_target)# first index after target
                idx_firstmove = np.argmax(t_lfads > t_firstmove) # first index after target

                #p = np.argmax(abs_co[trial_idx,idx_target:idx_firstmove+10,input_idx])
                p,_ = signal.find_peaks(abs_co[trial_idx,idx_target:idx_firstmove,input_idx])
                if len(p) == 0:
                    output_df.drop(index=(trial_idx, t_target), inplace=True)
                    continue

                p = p[np.argmax(abs_co[trial_idx, idx_target+p, input_idx])]
                if p == idx_firstmove - idx_target - 1:
                    k += 1

                if p * dt < min_peak_time:
                    output_df.drop(index=(trial_idx, t_target), inplace=True)
                    continue

                t_max = t_lfads[idx_target + p]
                input_target_latency.append(t_max - t_target)
                input_firstmove_latency.append(t_max - t_firstmove)
            
            trial_target_latency.append(input_target_latency)
            trial_firstmove_latency.append(input_firstmove_latency)

        target_latency += list(zip(*trial_target_latency))
        firstmove_latency += list(zip(*trial_firstmove_latency))

    target_columns = ['target_latency_%d'%(i+1) for i in range(n_inputs)]
    firstmove_columns = ['firstmove_latency_%d'%(i+1) for i in range(n_inputs)]
    output_df = output_df.groupby('trial').apply(lambda _df: _df.loc[_df.index[0][0]].loc[:trial_len])
    output_df[target_columns] = np.array(target_latency)
    output_df[firstmove_columns] = np.array(firstmove_latency)

    return output_df

def assign_target_column(_df):
    trial_df = _df.loc[_df.index[0][0]]
    #import pdb;pdb.set_trace()
    trial_df['target_x'] = np.append(trial_df['x'].iloc[1:].values, np.nan)
    trial_df['target_y'] = np.append(trial_df['y'].iloc[1:].values, np.nan)

    return trial_df

def get_target_direction(peaks, df):
    n_trials = len(peaks)
    directions = []
    for i in range(n_trials):
        for p in peaks[i]: 
            target_x, target_y = df.loc[i].kinematic.loc[p:].query('hit_target').iloc[0][['x','y']]
            theta = np.arctan2(target_y, target_x)
            directions.append(theta)

    return np.array(directions)

def get_target_pos(peaks, df):
    n_trials = len(peaks)
    directions = []
    tx = []
    ty = []
    for i in range(n_trials):
        for p in peaks[i]: 
            target_x, target_y = df.loc[i].kinematic.loc[p:].query('hit_target').iloc[0][['x','y']]
            tx.append(target_x)
            ty.append(target_y)

    return np.array(tx), np.array(ty)

def get_co_around_peak(peaks, co, dt, win_start=5, win_stop=5):
    n_trials = len(peaks)
    peak_co = []
    for i in range(n_trials):
        for p in peaks[i]:
            j = int(p/dt)
            interval = co[i, j-win_start:j+win_stop]
            if win_start + win_stop > len(interval):
                continue
            peak_co.append(interval)

    return np.array(peak_co)

def get_latencies(targets, peaks, co, dt, win_start, win_stop, trial_len):
    '''
    Calculates latencies between target appearance and nearest peak of each
    input

    Parameters:
    targets: pandas dataframe containing rows of df corresponding to 
    target appearanes. Must only include trials corresponding to 
    trials used to generate peaks parameter(below)

    peaks: Return of get_peaks, object array of shape (trial X outputs). 
    Each entry is a lists of floats representing time in trial, in seconds, 
    of each peak in controller output that mean specifications given by 
    and min_distance

    win_start: start of window after target to include controller peak
    
    win_stop: end of window after target to include controller peak

    Returns:
    targets_peaks: targets dataframe that was inputed, but with added columns
    next target location, and latency to controller peak if included in win_start
    and win_stop. kinematic and controller columns are split into two groups with a 
    multindex
    '''
    #making sure targets and peaks consider same number of trials
    n_trials_targets = targets.index[-1][0] + 1
    n_trials_peaks = peaks.shape[0]
    n_inputs = peaks.shape[1]
    n_trials = n_trials_targets
    latency = [[] for i in range(n_inputs)]
    target_peaks = targets
    target_peaks = target_peaks.groupby('trial').apply(assign_target_column)

    # counts of peaks for each input which are in window around target appearance
    peak_count = np.zeros(n_inputs)
    t_lfads = np.arange(co.shape[1]) * dt
    for input_idx in range(n_inputs):
        target_peaks['latency_%d'%input_idx] = np.nan
    for trial_idx in list(set(targets.index.get_level_values('trial'))):
        #if no events in trial, continue
        try:
            t_targets = targets.loc[trial_idx].index
        except:
            continue
        
        for input_idx in range(n_inputs):
            prev_ti = -1
            t_peaks = peaks[trial_idx, input_idx]
            for target_idx,t_target in enumerate(t_targets):
                candidate_peaks = t_peaks[(t_peaks - t_target >= win_start) & 
                                            (t_peaks - t_target < win_stop)]
                if np.any(candidate_peaks):
                    idx_peaks = [np.round(tp/dt).astype(int) for tp in candidate_peaks]
                    selected_peak = candidate_peaks[np.argmax(co[trial_idx, idx_peaks, input_idx])] # finding largest peak
                    latency = selected_peak -t_target
                    target_peaks.loc[trial_idx]['latency_%d'%input_idx].iloc[target_idx] = latency

    #removing targets with another target following too close (inside the examination window)
    drop_bool = ((target_peaks.index.get_level_values('trial')[1:]-target_peaks.index.get_level_values('trial')[:-1])==0) & ((target_peaks.index.get_level_values('time')[1:]-target_peaks.index.get_level_values('time')[:-1]) < win_stop)
    drop_bool = np.append(drop_bool, False)
    drop_idx = target_peaks.index[drop_bool]
    if len(drop_idx) > 0:
        target_peaks.drop(drop_idx, inplace=True)

    #removing targets that occur when lfads not run
    target_peaks = target_peaks.iloc[np.where(target_peaks.index.get_level_values('time')< trial_len)]

    return target_peaks, peak_count

# def get_latencies(targets, peaks, win_start, win_stop, trial_len):
#     '''
#     Calculates latencies between target appearance and nearest peak of each
#     input

#     Parameters:
#     targets: pandas dataframe containing rows of df corresponding to 
#     target appearanes. Must only include trials corresponding to 
#     trials used to generate peaks parameter(below)

#     peaks: Return of get_peaks, object array of shape (trial X outputs). 
#     Each entry is a lists of floats representing time in trial, in seconds, 
#     of each peak in controller output that mean specifications given by 
#     and min_distance

#     win_start: start of window after target to include controller peak
    
#     win_stop: end of window after target to include controller peak

#     Returns:
#     targets_peaks: targets dataframe that was inputed, but with added columns
#     next target location, and latency to controller peak if included in win_start
#     and win_stop. kinematic and controller columns are split into two groups with a 
#     multindex
#     '''
#     #making sure targets and peaks consider same number of trials
#     n_trials_targets = targets.index[-1][0] + 1
#     n_trials_peaks = peaks.shape[0]
#     n_inputs = peaks.shape[1]
#     n_trials = n_trials_targets
#     latency = [[] for i in range(n_inputs)]
#     target_peaks = targets
#     target_peaks = target_peaks.groupby('trial').apply(assign_target_column)

#     # counts of peaks for each input which are in window around target appearance
#     peak_count = np.zeros(n_inputs)
#     for input_idx in range(n_inputs):
#         target_peaks['latency_%d'%input_idx] = np.nan
#     for trial_idx in list(set(targets.index.get_level_values('trial'))):
#         #if no events in trial, continue
#         try:
#             t_targets = targets.loc[trial_idx].index
#         except:
#             continue
        
#         for input_idx in range(n_inputs):
#             prev_ti = -1
#             t_peaks = peaks[trial_idx, input_idx]
#             for tp in t_peaks:
#                 if any((tp - t_targets >= win_start) & (tp - t_targets < win_stop)):
#                     peak_count[input_idx] += 1
#                     diff_targets = tp - t_targets
#                     diff_targets = np.ma.MaskedArray(diff_targets, 
#                     (diff_targets < win_start) | (diff_targets >= win_stop)) #masking values outside window
#                     target_idx = np.argmin(np.abs(diff_targets))
#                     if target_idx == prev_ti: #continue if there is already a peak for that target
#                         continue
#                     else:
#                         target_peaks.loc[trial_idx]['latency_%d'%input_idx].iloc[target_idx] = diff_targets[target_idx]
                    
#                     prev_ti = target_idx

#     #removing targets with another target following too close (inside the examination window)
#     drop_bool = ((target_peaks.index.get_level_values('trial')[1:]-target_peaks.index.get_level_values('trial')[:-1])==0) & ((target_peaks.index.get_level_values('time')[1:]-target_peaks.index.get_level_values('time')[:-1]) < win_stop)
#     drop_bool = np.append(drop_bool, False)
#     drop_idx = target_peaks.index[drop_bool]
#     if len(drop_idx) > 0:
#         target_peaks.drop(drop_idx, inplace=True)

#     #removing targets that occur when lfads not run
#     target_peaks = target_peaks.iloc[np.where(target_peaks.index.get_level_values('time')< trial_len)]

#     return target_peaks, peak_count

def get_target_peak_counts(target_peaks, input_idx, all_inputs=False):
    '''
    Counts of targets with and without peak for each lfads input

    Parameters:
    target_peaks: dataframe as returned by get_latencies

    input_idx: controller index or indices to count. Can be int, or if multiple 
    controller inputs are to be counted, a list

    all_inputs: if True, all inputs must have a peak to be counted (default: False)

    Return:
    targets_with_peak
    '''
    if not isinstance(input_idx, list):
        input_idx = [input_idx]

    #making sure all inputs exist
    inputs_exist = np.all(['latency_%d'%idx in target_peaks.columns for idx in input_idx])
    assert(inputs_exist, 'input_idx argument does not match columns in target_peaks argument')

    if all_inputs:
        targets_with_peak = target_peaks[['latency_%d'%i for i in input_idx]].isnull().all(axis=1)
    else:
        targets_with_peak = target_peaks[['latency_%d'%i for i in input_idx]].isnull().any(axis=1)

    return targets_with_peak

# def get_peak_latencies(df, co, min_heights, dt, used_inds,
#                         min_height, win_start=0, win_stop=0.5):
#     '''
#     Parameters:
#     df: pandas dataframe containing experiment data
    
#     Returns
#     latencies: list of lists of latency from nearest target appearance in window.
#     Each list gives the latencies for a different controller output
#     all_peak_counts: count of all controller peaks, for each controller output
#     target_peak_count: count of all peaks within window of target, for each controller ouput
#     '''
#     dt = utils.get_dt(h5file, input_info)

#     used_inds = utils.get_indices(input_info, trial_type)
#     targets = df.loc[used_inds].kinematic.query('hit_target')
#     trial_len = co.shape[1] * dt
#     t_lfads = np.arange(co.shape[1]) * dt #time labels of lfads input
#     all_peak_count = 0 # count of all controller peaks
#     target_peak_count = 0 # count of all peaks within window of target
#     latencies = [] # latency
#     for i in used_inds:
#         peaks = []
#         for input_idx in range(1):#range(co.shape[2]):
#             input_peaks, _ = signal.find_peaks(np.abs(co[i, :, input_idx]), 
#                             height=min_heights[input_idx])
#             peaks.append(input_peaks)

#         peaks = np.concatenate(peaks)
#         t_peaks = t_lfads[peaks]
#         t_targets = targets.loc[i].index
#         all_peak_count += len(t_peaks)
#         for tp in t_peaks:
#             if any((tp - t_targets >= win_start) & (tp - t_targets < win_stop)):
#                 diff_targets = tp - t_targets
#                 latency = np.min(diff_targets[diff_targets>0]) #latency to closest target
#                 latencies.append(latency)
#                 target_peak_count += 1


def get_peak_df(df, co, trial_len, min_heights, event_df, dt=0.01, relative=False, win_start=0, win_stop=0.5):
    '''Chaining above function above to get useful dataframe'''
    peaks = get_peaks(co, dt, min_heights, relative, min_distance=cfg['min_peak_spacing'])
    peak_df, _ = get_latencies(event_df, peaks, co, dt, win_start, win_stop, trial_len=trial_len)

    return peak_df
        
if __name__=='__main__':
    trial_type = 'all'

    lfads_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_2OLS24_%s.h5"%trial_type
    data_filename = "/home/pmalonis/lfads_analysis/data/intermediate/rockstar.p"
    inputInfo_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_inputInfo.mat"

    df = pd.read_pickle(data_filename)
    input_info = io.loadmat(inputInfo_filename)
    with h5py.File(lfads_filename) as h5file:
        co = h5file['controller_outputs'][:]
        
    used_inds = utils.get_indices(input_info, trial_type)