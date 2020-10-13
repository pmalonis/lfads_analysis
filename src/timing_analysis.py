import numpy as np
import h5py
import pandas as pd
from scipy import signal

# trial_type = 'all'

# lfads_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_8QTVEk_%s.h5"%trial_type
# data_filename = "/home/pmalonis/lfads_analysis/data/intermediate/rockstar.p"
# inputInfo_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_inputInfo.mat"

# df = pd.read_pickle(data_filename)
# input_info = io.loadmat(inputInfo_filename)
# with h5py.File(lfads_filename) as h5file:
#     co = h5file['controller_outputs'].value
#     

# used_inds = utils.get_indices(input_info, trial_type)


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
        targets = df.loc.kinematic.query('hit_target')

    return targets

def get_peaks(co, dt, min_height, min_distance):
    '''
    Returns times of peaks of contorller outputs

    Parameters
    co: LFADS controller outputs. 3D numpy array with dimensions
    trials X time X controller output

    dt: time in between samples of controller outputs 

    min_height: minimum height of peak to consider. Either scalar or 
    vector with length equal to co.shape[2]. The later sets a separate 
    height treshold for each controller output

    min_distance: minimum distance to neighboring peak, as given to 
    scipy.signal.find_peaks as the "distance" argument. As with min_height, 
    can be a float or a vector.

    Returns:
    peaks: object array of shape (trial X outputs). Each entry is a 
    lists of floats representing time in trial, in seconds, of each peak 
    in controller output that mean specifications given by min_height
    and min_distance
    '''
    peaks = np.empty(shape=(co.shape[0], co.shape[2]), dtype='object')
    # filling array with empty lists. The default object is None, but empty list is simpler 
    # to work with since it can be pass to dataframe as slice with no error
    peaks.fill([])
    t_lfads = np.arange(co.shape[1]) * dt #time labels of lfads input
    for trial_idx in range(co.shape[0]):
        for input_idx in range(co.shape[2]):
            p, _ = signal.find_peaks(np.abs(co[i, :, input_idx]), 
                            height=min_heights[input_idx])
            peaks[trial_idx, input_idx] = t_lfads[p]
 
    return peaks

def assign_target_column(_df):
    _df['target_x'] = np.append(_df['x'].iloc[1:].values, np.nan)
    _df['target_y'] = np.append(_df['y'].iloc[1:].values, np.nan)

    return _df

def get_latencies(targets, peaks, win_start, win_stop):
    '''
    Calculates latencies between target appearance and nearest peak of each
    input

    Parameters:
    targets: pandas dataframe containing rows of df corresponding to 
    target appearanes. Must only include trials corresponding to 
    trials used to generate peaks parameter(below)

    peaks: Return of get_peaks, object array of shape (trial X outputs). 
    Each entry is a lists of floats representing time in trial, in seconds, 
    of each peak in controller output that mean specifications given by min_height
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
    assert(n_trials_targets == n_trials_peaks)
    
    n_trials = n_trials_targets
    for trial_idx in range(n_trials):
        t_targets = targets.loc[trial_idx].index
        for input_idx in range(peaks.shape[1]):
            t_peaks = peaks[trial_idx, input_idx]
            for tp in t_peaks:
                if any((tp - t_targets >= win_start) & (tp - t_targets < win_stop)):
                    diff_targets = tp - t_targets
                    latency = np.min(diff_targets[diff_targets>0])

            
    
def get_peak_latencies(df, co, min_heights, dt, used_inds,
                        min_height, win_start=0, win_stop=0.5):
    '''
    Parameters:
    df: pandas dataframe containing experiment data
    

    Returns
    latencies: list of lists of latency from nearest target appearance in window.
    Each list gives the latencies for a different controller output
    all_peak_counts: count of all controller peaks, for each controller output
    target_peak_count: count of all peaks within window of target, for each controller ouput
    '''
    dt = utils.get_dt(h5file, input_info)

    used_inds = utils.get_indices(input_info, trial_type)
    targets = df.loc[used_inds].kinematic.query('hit_target')
    trial_len = co.shape[1] * dt
    t_lfads = np.arange(co.shape[1]) * dt #time labels of lfads input
    all_peak_count = 0 # count of all controller peaks
    target_peak_count = 0 # count of all peaks within window of target
    latencies = [] # latency
    for i in used_inds:
        peaks = []
        for input_idx in range(1):#range(co.shape[2]):
            input_peaks, _ = signal.find_peaks(np.abs(co[i, :, input_idx]), 
                            height=min_heights[input_idx])
            peaks.append(input_peaks)

        peaks = np.concatenate(peaks)
        t_peaks = t_lfads[peaks]
        t_targets = targets.loc[i].index
        all_peak_count += len(t_peaks)
        for tp in t_peaks:
            if any((tp - t_targets >= win_start) & (tp - t_targets < win_stop)):
                diff_targets = tp - t_targets
                latency = np.min(diff_targets[diff_targets>0]) #latency to closest target
                latencies.append(latency)
                target_peak_count += 1