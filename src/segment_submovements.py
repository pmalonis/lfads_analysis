import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import yaml
import os
from scipy.signal import savgol_filter
import sys
sys.path.insert(0,'.')
import utils
from submovement_core import speed_minima
from importlib import reload
reload(utils)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def trial_intervals(trial_df):
    '''
    Calculates submovement transitions from single trial, based on 
    speed profile
    
    Parameters
    trial_df: single-trial dataframe
    
    Returns
    intervals: list of tuples representing start/stop times for each 
    submovement
    '''
    trans = trial_transitions(trial_df)
    intervals = [(trans[i], trans[i+1]) for i in range(len(trans)-1)]
    intervals = [(0, trans[0])] + intervals

    return intervals

def add_target_position_column(trial_df):
    targets = trial_df.kinematic.query('hit_target')
    idx_target = np.where(trial_df.kinematic['hit_target'].values)[0]
    target_x = np.zeros(trial_df.shape[0])
    target_y = np.zeros(trial_df.shape[0])
    for i in range(targets.shape[0]-1):
        target_x[idx_target[i]:idx_target[i+1]] = targets['x'].iloc[i+1]
        target_y[idx_target[i]:idx_target[i+1]] = targets['y'].iloc[i+1]

    trial_df['target_x'] = target_x
    trial_df['target_y'] = target_y

    return trial_df

def convert_angle(angle_diff):
    angle_diff -= 2*np.pi
    angle_diff[angle_diff<=-np.pi] = angle_diff[angle_diff<=-np.pi]%np.pi
    return angle_diff


def angle_to_xy(theta):
    '''angle in radians to unit vector xy'''
    return np.array([np.cos(theta), np.sin(theta)])

def angle_difference(a, b):
    '''Difference between two angles, which are
    given in randians between -pi and pi'''
    #this ensures that precision error doesn't give an invalid arccos
    dotprod = angle_to_xy(a).dot(angle_to_xy(b))
    dotprod = min(1, dotprod)
    dotprod = max(-1, dotprod)
    
    return np.arccos(dotprod)
    
def peak_error(trial_df, exclude_post_target=None, exclude_pre_target=None):
    targets = trial_df.kinematic.query('hit_target')
    next_target = targets[['x', 'y']].iloc[1:]

    trial_df = add_target_position_column(trial_df)
    target_vect = trial_df[['target_x','target_y']].values - trial_df.kinematic[['x', 'y']].values
    target_vect = (target_vect.T / np.linalg.norm(target_vect, axis=1)).T
    arm_vect = trial_df.kinematic[['x_vel', 'y_vel']].values
    arm_vect = (arm_vect.T / np.linalg.norm(arm_vect, axis=1)).T
    target_dir = np.arctan2(target_vect[:,1], target_vect[:,0])
    arm_dir = np.arctan2(arm_vect[:,1], arm_vect[:,0])
    angle_diff = angle_difference(arm_dir, target_dir)

    return angle_diff

def trial_transitions(trial_df, exclude_post_target=None, exclude_pre_target=None):
    '''Calculates submovement transitions from single trial, based on 
    speed profile
    
    Parameters
    trial_df: single-trial dataframe
    
    Returns
    trial_transitions: List of minima defining starts/ends of submovements, in index
    '''
    x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].values.T
    minima = speed_minima(x_vel, y_vel)

    targets = trial_df.kinematic.query('hit_target')
    if exclude_post_target is not None:
        #determine whether event time is in window post target
        event_filter = lambda x: not np.any((x - targets.index < exclude_post_target) & (x - targets.index >= 0))
        non_post_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
        minima = minima[non_post_target_events]
    
    if exclude_pre_target is not None:
        #determine whether event time is in window post target
        event_filter = lambda x: not np.any((targets.index - x < exclude_pre_target) & (targets.index - x >= 0))
        non_pre_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
        minima = minima[non_pre_target_events]

    return minima

# def trial_firstmoves(trial_df, exclude_post_target=None, exclude_pre_target=None):
#     '''Calculates submovement transitions from single trial, based on 
#     speed profile
    
#     Parameters
#     trial_df: single-trial dataframe
    
#     Returns
#     trial_transitions: List of minima defining starts/ends of submovements, in index
#     '''
#     threshold = cfg['firstmove_speed_transition_thresh']
#     prominence = cfg['firstmove_min_speed_prominence']
#     distance = cfg['firstmove_min_distance']

#     x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].values.T
#     minima = speed_minima(x_vel, y_vel, threshold, prominence, distance)

#     targets = trial_df.kinematic.query('hit_target')
#     t_targets = targets.index.values
#     targets
#     t = trial_df.index.values
#     firstmove = []
#     for i in range(len(targets)-1):
#         target_moves = [m for m in minima if t[m] > t_targets[i] and t[m] < t_targets[i+1]]
#         if len(target_moves) > 0:
#             target_firstmove = target_moves[0]
#             firstmove.append(target_firstmove)

#     minima = np.array(firstmove)
#     if exclude_post_target is not None:
#         #determine whether event time is in window post target
#         event_filter = lambda x: not np.any((x - targets.index < exclude_post_target) & (x - targets.index >= 0))
#         non_post_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
#         minima = minima[non_post_target_events]
    
#     if exclude_pre_target is not None:
#         #determine whether event time is in window post target
#         event_filter = lambda x: not np.any((targets.index - x < exclude_pre_target) & (targets.index - x >= 0))
#         non_pre_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
#         minima = minima[non_pre_target_events]

#     return minima

def trial_firstmoves(trial_df, exclude_post_target=None, exclude_pre_target=None):
    '''Calculates submovement transitions from single trial, based on 
    speed profile
    
    Parameters
    trial_df: single-trial dataframe
    
    Returns
    trial_transitions: List of minima defining starts/ends of submovements, in index
    '''
    threshold = cfg['firstmove_speed_transition_thresh']
    prominence = cfg['firstmove_min_speed_prominence']
    distance = cfg['firstmove_min_distance']

    x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].values.T
    targets = trial_df.kinematic.query('hit_target')
    t_targets = targets.index.values
    idx_targets = [trial_df.index.get_loc(t_targets[i]) for i in range(len(t_targets))]
    minima = speed_minima(x_vel, y_vel, threshold, prominence, distance, idx_targets)

    targets = trial_df.kinematic.query('hit_target')
    t_targets = targets.index.values
    t = trial_df.index.values
    firstmove = []
    for i in range(len(targets)-1):
        target_moves = [m for m in minima if t[m] > t_targets[i] and t[m] < t_targets[i+1]]
        if len(target_moves) > 0:
            target_firstmove = target_moves[0]
            firstmove.append(target_firstmove)

    minima = np.array(firstmove)
    if exclude_post_target is not None:
        #determine whether event time is in window post target
        event_filter = lambda x: not np.any((x - targets.index < exclude_post_target) & (x - targets.index >= 0))
        non_post_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
        minima = minima[non_post_target_events]
    
    if exclude_pre_target is not None:
        #determine whether event time is in window post target
        event_filter = lambda x: not np.any((targets.index - x < exclude_pre_target) & (targets.index - x >= 0))
        non_pre_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
        minima = minima[non_pre_target_events]

    return minima

def trial_first_accel(trial_df, exclude_post_target=None, exclude_pre_target=None):
    '''Calculates submovement transitions from single trial, based on 
    speed profile
    
    Parameters
    trial_df: single-trial dataframe
    
    Returns
    trial_transitions: List of minima defining starts/ends of submovements, in index
    '''
    threshold = cfg['firstmove_speed_transition_thresh']
    prominence = cfg['firstmove_min_speed_prominence']
    distance = cfg['firstmove_min_distance']

    x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].values.T
    speed = utils.get_speed(x_vel, y_vel)
    targets = trial_df.kinematic.query('hit_target')
    t_targets = targets.index.values
    idx_targets = [trial_df.index.get_loc(t_targets[i]) for i in range(len(t_targets))]
    minima = speed_minima(x_vel, y_vel, threshold, prominence, distance, idx_targets)

    targets = trial_df.kinematic.query('hit_target')
    t_targets = targets.index.values
    t = trial_df.index.values
    firstmove = []
    for i in range(len(targets)-1):
        target_moves = [m for m in minima if t[m] > t_targets[i] and t[m] < t_targets[i+1]]
        if len(target_moves) > 0:
            target_firstmove = target_moves[0]
            p,_ = signal.find_peaks(speed[target_firstmove:]) #get peaks after firstmove 
            if len(p) < 1:                
                continue
                
            next_peak = target_firstmove + p[0]
            #find threshold to take crossing of
            thresh = speed[target_firstmove] + cfg['accel_thresh']*(speed[next_peak] - speed[target_firstmove])
            # get point where threshold is reached (relative to firstmove)
            cross_thresh = np.argmax(speed[target_firstmove:] > thresh)
            # make relative to stat of trial
            target_firstmove += cross_thresh
            firstmove.append(target_firstmove)

    minima = np.array(firstmove)
    if exclude_post_target is not None:
        #determine whether event time is in window post target
        event_filter = lambda x: not np.any((x - targets.index < exclude_post_target) & (x - targets.index >= 0))
        non_post_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
        minima = minima[non_post_target_events]
    
    if exclude_pre_target is not None:
        #determine whether event time is in window post target
        event_filter = lambda x: not np.any((targets.index - x < exclude_pre_target) & (targets.index - x >= 0))
        non_pre_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
        minima = minima[non_pre_target_events]

    return minima


def trial_corrections(trial_df, exclude_post_target=None, exclude_pre_target=None):
    '''Calculates submovement transitions from single trial, based on 
    speed profile
    
    Parameters
    trial_df: single-trial dataframe
    
    Returns
    trial_transitions: List of minima defining starts/ends of submovements, in index
    '''
    firstmove_threshold = cfg['firstmove_speed_transition_thresh']
    firstmove_prominence = cfg['firstmove_min_speed_prominence']
    firstmove_distance = cfg['firstmove_min_distance']

    threshold = cfg['correction_speed_transition_thresh']
    prominence = cfg['correction_min_speed_prominence']
    distance = cfg['correction_min_distance']

    x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].values.T
    targets = trial_df.kinematic.query('hit_target')
    t_targets = targets.index.values
    idx_targets = [trial_df.index.get_loc(t_targets[i]) for i in range(len(t_targets))]
    minima = speed_minima(x_vel, y_vel, threshold, prominence, distance, idx_targets)
    firstmove_params_minima = speed_minima(x_vel, y_vel, firstmove_threshold, firstmove_prominence, firstmove_distance, idx_targets)    
    t = trial_df.index.values
    corrections= []
    for i in range(len(targets)-1):
        target_moves = np.array([m for m in minima if t[m] > t_targets[i] and t[m] < t_targets[i+1]])
        if len(target_moves) > 1: #must be more that 1 movement, if only one then not a correction
            target_firstmoves = [m for m in firstmove_params_minima 
                                 if t[m] > t_targets[i] and t[m] < t_targets[i+1]]
            if len(target_firstmoves) > 0: #must be a first movement according to firstmove parameters
                target_corrections = target_moves[target_moves > target_firstmoves[0]]
                corrections.append(target_corrections)

    if len(corrections) > 0:
        minima = np.concatenate(corrections)
    else:
        minima = np.array([])
    if exclude_post_target is not None:
        #determine whether event time is in window post target
        event_filter = lambda x: not np.any((x - targets.index < exclude_post_target) & (x - targets.index >= 0))
        non_post_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
        minima = minima[non_post_target_events]
    
    if exclude_pre_target is not None:
        #determine whether event time is in window post target
        event_filter = lambda x: not np.any((targets.index - x < exclude_pre_target) & (targets.index - x >= 0))
        non_pre_target_events = [event_filter(ev) for ev in trial_df.iloc[minima].index]
        minima = minima[non_pre_target_events]

    return minima

def dataset_events(df, func, column_label='transition', exclude_post_target=None, exclude_pre_target=None):
    '''
    return rows of dataset corresponding to event, based on a function that
    computes events for each trial

    Parameters
    df: dataframe of dataset
    func: function for calculating events from trial dataframe. function should take 
    one argument, a trial dataframe, and return index of data in the trial to label
    column_label: name to assign

    Returns
    events: dataframe rows corresponding to transitions
    '''
    
    event_idx = df.groupby('trial').apply(lambda _df: func(_df.loc[_df.index[0][0]], exclude_post_target))
    event_idx = np.concatenate([event_idx.loc[i] + (df.loc[:i-1].shape[0] if i > 0 else 0) for i in range(len(event_idx))])

    events = df.kinematic.iloc[event_idx]

    return events

def trial_control_transitions(trial_df):    
    return

# def speed_minima(x_vel, y_vel):
#     '''Calculates submovements from velocity coordinates, based on 
#     speed profile
    
#     Parameters
#     x_vel: velocity in x-axis
#     y_vel: velocity in y-axis
    
#     Returns
#     intervals: List of minima defining starts/ends of submovements
#     '''
#     speed = np.sqrt(x_vel**2 + y_vel**2)
#     minima,_ = signal.find_peaks(-speed,
#                                 height=-cfg['speed_transition_thresh'],
#                                 prominence=cfg['min_speed_prominence'])
    
#     return minima

# def speed_minima(x_vel, y_vel):
#     '''Calculates submovements from velocity coordinates, based on 
#     speed profile
    
#     Parameters
#     x_vel: velocity in x-axis
#     y_vel: velocity in y-axis
    
#     Returns
#     intervals: List of minima defining starts/ends of submovements
#     '''
#     speed = utils.get_speed(x_vel, y_vel)
#     minima, _  = signal.find_peaks(-speed,
#                                   height=-cfg['speed_transition_thresh'])
    
#     prominence = cfg['min_speed_prominence']
#     bounds = np.append(minima, len(speed) - 1)
#     minima = [minima[i] for i in range(len(minima))
#               if speed[bounds[i]:bounds[i+1]].max() - speed[minima[i]] > prominence]
    
#     return np.array(minima)

# def speed_minima(x_vel, y_vel, threshold, prominence, distance):
#     '''Calculates submovements from velocity coordinates, based on 
#     speed profile
    
#     Parameters
#     x_vel: velocity in x-axis
#     y_vel: velocity in y-axis
#     threshold: threshold above which speed minima will not be returned
#     prominence: minimum size of the following movement

#     Returns
#     intervals: List of minima defining starts/ends of submovements
#     '''

#     speed = utils.get_speed(x_vel, y_vel)
#     minima, _  = signal.find_peaks(-speed,
#                                     height=-threshold,
#                                     distance=distance)
    
#     bounds = np.append(minima, len(speed) - 1)
#     minima = np.array([minima[i] for i in range(len(minima)) if 
#                         speed[bounds[i]:bounds[i+1]].max() - speed[minima[i]] > prominence])

#     # while any(np.diff(minima)<distance):
#     #     idx = np.argmax(np.diff(minima)<distance)
#     #     remove_candidates = [idx, idx + 1]
#     #     # deleting greater minimum
#     #     to_remove = remove_candidates[np.argmax(speed[minima[remove_candidates]])]
#     #     minima = np.delete(minima, to_remove)

#     return np.array(minima)

def right_prominences(minima, speed):
    maxima,_ = signal.find_peaks(speed)
    prominences = np.zeros(len(minima))
    for i, mi in enumerate(minima):
        next_peak = maxima[maxima > mi][0]
        prominences[i] = speed[next_peak] - mi

    return prominences

# def speed_minima(x_vel, y_vel, threshold, prominence, distance, idx_targets):
#     '''Calculates submovements from velocity coordinates, based on 
#     speed profile
    
#     Parameters
#     x_vel: velocity in x-axis
#     y_vel: velocity in y-axis
#     threshold: threshold above which speed minima will not be returned
#     prominence: minimum size of the following movement

#     Returns
#     intervals: List of minima defining starts/ends of submovements
#     '''

#     speed = utils.get_speed(x_vel, y_vel)
#     minima, _  = signal.find_peaks(-speed,
#                                     height=-threshold)
    
#     bounds = np.append(minima, len(speed) - 1)
#     minima = np.array([minima[i] for i in range(len(minima)) if 
#                         speed[bounds[i]:bounds[i+1]].max() - speed[minima[i]] > prominence])
        
#     target_spaced_minima = [m + distance*np.sum(idx_targets<m) for m in minima]
#     all_to_remove = []
#     while any(np.diff(target_spaced_minima)<distance):
#         idx = np.argmax(np.diff(target_spaced_minima)<distance)
#         remove_candidates = [idx, idx + 1]
#         # deleting lesser prominence
#         bounds = np.append(minima, len(speed) - 1)
#         prominences = np.array([speed[bounds[i]:bounds[i+1]].max() - speed[minima[i]] 
#                         for i in range(len(minima))])
#         #removing peak with smaller prominence
#         to_remove = remove_candidates[np.argmin(prominences[remove_candidates])]
#         target_spaced_minima.pop(to_remove)
#         minima = np.delete(minima, to_remove)

#     return np.array(minima)


# def trial_maxima(trial_df, exclude_post_target=None, exclude_pre_target=None):
#     '''Calculates submovement transitions from single trial, based on 
#     speed profile
    
#     Parameters
#     trial_df: single-trial dataframe
    
#     Returns
#     trial_transitions: List of minima defining starts/ends of submovements, in index
#     '''
#     x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].values.T
#     maxima = speed_maxima(x_vel, y_vel)

#     targets = trial_df.kinematic.query('hit_target')
#     if exclude_post_target is not None:
#         #determine whether event time is in window post target
#         event_filter = lambda x: not np.any((x - targets.index < exclude_post_target) & (x - targets.index >= 0))
#         non_post_target_events = [event_filter(ev) for ev in trial_df.iloc[maxima].index]
#         maxima = maxima[non_post_target_events]
    
#     if exclude_pre_target is not None:
#         #determine whether event time is in window post target
#         event_filter = lambda x: not np.any((targets.index - x < exclude_pre_target) & (targets.index - x >= 0))
#         non_pre_target_events = [event_filter(ev) for ev in trial_df.iloc[maxima].index]
#         maxima = maxima[non_pre_target_events]

#     return maxima

# def impulse_corrections(trial_df):
#     '''Finds points of impulse control corrections, based on speed'''
    
#     t = trial_df.index.values
#     speed = np.linalg.norm(trial_df.kinematic[['x_vel', 'y_vel']].values)
#     speed = savgol_filter(speed, cfg['speed_filter_win'], cfg['speed_filter_order'])
    
#     accel = np.gradient(speed, t)
#     jerk = np.gradient(accel, t)

#     return accel

# def filter_peaks(peak_df):
#     peak_df = peak_df[peak_df[['latency_0', 'latency_1']].notnull().any(axis=1)]
#     return peak_df

# def peak_speed(trial_df, exclude_pre_target=None, exclude_post_target=None):
#     speed = np.linalg.norm(trial_df.kinematic[['x_vel', 'y_vel']].values, axis=1)
#     idx_targets = np.where(trial_df.kinematic['hit_target'])[0]
#     peaks = []
#     for i in range(len(idx_targets)-1):
#         target_peaks,_ = find_peaks(speed[idx_targets[i]:idx_targets[i+1]])
#         if len(target_peaks) > 0:
#             target_peaks += idx_targets[i]
#             peaks.append(target_peaks[speed[target_peaks].argmax()])
#         else:
#             continue

#     return np.array(peaks)

def pre_peak(trial_df, exclude_pre_target=None, exclude_post_target=None):
    x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].values.T
    minima = ss.speed_minima(x_vel, y_vel)
    peaks = peak_speed(trial_df)
    peaks = peaks[peaks>minima[0]]
    t = trial_df.index.values
    speed = np.sqrt(x_vel**2 + y_vel**2)
    accel = np.gradient(speed, t)
    
    movements = []
    for i in range(len(peaks)-1):
        move_start = minima[np.argmin(np.ma.MaskedArray(peaks[i] - minima, peaks[i] - minima < 0))]
        move_stop = find_peaks(-speed[peaks[i]:])[0][0]
        start_accel = accel[move_start]
        peak_accel = np.max(accel[move_start:peaks[i]])
        accel_amp = peak_accel-start_accel

        peak_accel_idx = move_start + np.argmax(accel[move_start:peaks[i]])

        up_peaks = move_start + find_peaks(accel[move_start:peak_accel_idx])[0]
        if len(up_peaks) > 0:
            up_troughs = np.zeros(len(up_peaks), dtype=int)
            for j,p in enumerate(up_peaks):
                try:
                    up_troughs[j] = p + find_peaks(-accel[p:peak_accel_idx])[0][0]
                except:
                    import pdb; pdb.set_trace()

            movements += list(up_peaks[(accel[up_peaks] - accel[up_troughs])>(0.1*accel_amp)])

        down_peaks = peak_accel_idx + find_peaks(accel[peak_accel_idx:move_stop])[0]
        if len(down_peaks) > 0:
            down_troughs = np.zeros(len(down_peaks), dtype=int)
            for j,p in enumerate(down_peaks):
                try:
                    down_troughs[j] = p - find_peaks(-accel[p:peak_accel_idx:-1])[0][0]
                except:
                    import pdb; pdb.set_trace()

            movements += list(down_peaks[(accel[down_peaks] - accel[down_troughs])>0.1 * accel_amp])

    return np.array(movements)

def plot_submovements(trial_df):
    '''
    Plots targets with arrows showing target order, along with hand trajectory,
    using colors to distinguish segments of the trajectory

    Parameters
    trial_df: single-trial dataframe corresponding to trial to plot

    Returns
    ax: axis object containing plot
    '''
    targets = trial_df.kinematic.query('hit_target')[['x', 'y']].values.T
    f, ax = plt.subplots(figsize=(8,6))
    ax.plot(*targets[:,0], 'ro')
    ax.plot(*targets[:,1:], 'bs')
    xs, ys = targets[:,:-1]
    dxs, dys = np.diff(targets, axis=1)
    arrow_shorten = 1
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        arrow_len = np.linalg.norm([dx, dy])
        c = (arrow_len - arrow_shorten)/arrow_len
        #plt.arrow(x, y, dx*c, dy*c, width=.00001, head_width=5, length_includes_head=True, alpha=.2)

    intervals = trial_intervals(trial_df)
    x, y = trial_df.kinematic[['x', 'y']].values.T
    ax.plot(x, y, 'k')
    cmap = matplotlib.cm.get_cmap('Paired')
    for i, interval in enumerate(intervals):
        pos = trial_df.kinematic[['x', 'y']].iloc[interval[0]:interval[1]].values.T
        c = cmap(((i + 0.5)/10)%1)
        ax.plot(*pos, color=c)

    return ax, intervals

def plot_trajectory_co(trial_df, trial_co, dt, co_min=-1, co_max=1):
    '''
    Plots trajectory of trial, sampled at points where controller output is recorded

    Parameters
    trial_df: single-trial dataframe corresponding to trial to plot
    trial_co: controller outputs for one trial (time X n_outputs)
    dt: sampling period of lfads
    co_min: minimum of colorbar scale for controller output
    co_max: maximum of colorbar scale for controller output
    '''
    trial_len = len(trial_co) * dt
    targets = trial_df.loc[:trial_len].kinematic.query('hit_target')[['x', 'y']].values.T
    f, ax = plt.subplots(figsize=(10,8))
    ax.plot(*targets[:,0], 'ks', markersize=10)
    ax.plot(*targets[:,1:], 'ks', markersize=10)
    xs, ys = targets[:,:-1]
    dxs, dys = np.diff(targets, axis=1)
    arrow_shorten = 1
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        arrow_len = np.linalg.norm([dx, dy])
        c = (arrow_len - arrow_shorten)/arrow_len
        plt.arrow(x, y, dx*c, dy*c, width=.00001, head_width=2, length_includes_head=True, alpha=.2)
    
    t = np.arange(len(trial_co)) * dt
    idx = [trial_df.index.get_loc(time, method='nearest') for time in t]
    x, y = trial_df.iloc[idx].kinematic[['x', 'y']].values.T
    cm = plt.cm.get_cmap('coolwarm')
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=co_min, vmax=co_max)
    plt.scatter(x, y, s=20, c=trial_co, cmap=cm, norm=norm)
    cb = plt.colorbar()
    cb.set_label('Inferred Input 1 Value')

def plot_abs_trajectory_co(trial_df, trial_co, dt, co_max=3):
    '''
    Plots trajectory of trial, sampled at points where controller output is recorded

    Parameters
    trial_df: single-trial dataframe corresponding to trial to plot
    trial_co: controller outputs for one trial (time X n_outputs)
    co_norm: normalizing factor for controller output.
    dt: sampling period of lfads
    co_max: maximum of colorbar scale for controller output (in units of co_norm)
    '''
    trial_len = len(trial_co) * dt
    targets = trial_df.loc[:trial_len].kinematic.query('hit_target')[['x', 'y']].values.T
    f, ax = plt.subplots(figsize=(10,8))
    ax.plot(*targets[:,0], 'ro')
    ax.plot(*targets[:,1:], 'bs')
    xs, ys = targets[:,:-1]
    dxs, dys = np.diff(targets, axis=1)
    arrow_shorten = 1
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        arrow_len = np.linalg.norm([dx, dy])
        c = (arrow_len - arrow_shorten)/arrow_len
        plt.arrow(x, y, dx*c, dy*c, width=.00001, head_width=2, length_includes_head=True, alpha=.2)
    
    t = np.arange(trial_co.shape[0]) * dt
    trial_co = np.copy(trial_co)
    trial_co = np.abs(trial_co)
    trial_co = trial_co.sum(1)
    mean_co = np.mean(trial_co)
    idx = [trial_df.index.get_loc(time, method='nearest') for time in t]
    x, y = trial_df.iloc[idx].kinematic[['x', 'y']].values.T
    cm = plt.cm.get_cmap('YlOrRd')
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=mean_co, vmin=mean_co-0.001, vmax=co_max)
    plt.scatter(x, y, s=10, c=trial_co, cmap=cm, norm=norm, edgecolors='k', linewidth=0.1)
    cb = plt.colorbar()
    cb.set_label('Controller Value')

    return trial_co, mean_co

def plot_trial(trial_df, trial_co, dt, trial_firstmove_df, trial_correction_df):
    '''
    Plots the speed profile, controller outputs, and target times
    
    Parameters
    trial_df: single-trial dataframe corresponding to trial to plot
    trial_co: controller outputs for one trial (time X n_outputs)
    '''
    trial_len = trial_co.shape[0] * dt
    trial_df = trial_df.loc[:trial_len]
    t_targets = trial_df.kinematic.query('hit_target').index.values
    #x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].loc[:trial_len].T.values
    x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].T.values
    speed = np.sqrt(x_vel**2 + y_vel**2)
    filt_speed = utils.get_speed(x_vel, y_vel)
    t = trial_df.loc[:trial_len].index.values
    t = trial_df.index.values
    lns = [] #for legend
    #transitions = trial_transitions(trial_df)
    fm = trial_firstmove_df.loc[:trial_len].index.values
    nc = trial_correction_df.loc[:trial_len].index.values
    
    fm = [np.argmin(np.abs(t-fm[i])) for i in range(len(fm))]
    nc = [np.argmin(np.abs(t-nc[i])) for i in range(len(nc))]

    trial_len_ms = trial_len*1000
    #transitions = transitions[transitions < trial_len_ms]
    plt.figure(figsize=(12,6))
    #filt_speed = get_normal_accel(x_vel, y_vel, t)
    #lns.append(plt.plot(t, speed,'g'))
    lns.append(plt.plot(t, filt_speed, 'k'))
    #plt.twinx()
    
    #fm_line = plt.plot(t[fm], filt_speed[fm],'r.',markersize=15, label='First Movement')
    #c_line = plt.plot(t[nc], filt_speed[nc],'b.',markersize=15, label='Correction')
    fm_line = plt.plot(t[fm], filt_speed[fm],'m.',markersize=12, label='First Movement')
    c_line = plt.plot(t[nc], filt_speed[nc],'g.',markersize=12, label='Correction')

    
    ymin, ymax = plt.ylim() 
    plt.vlines(t_targets[1:], ymin, ymax)   
    plt.text(t_targets[1]+0.02, ymax*.915, "Target 1\nAcquired", fontsize=12)
    for i,t in enumerate(t_targets[2:]):
        plt.text(t+0.01, ymax*.95, "Target %d"%(i+2), fontsize=12)
    plt.legend(handles=[fm_line[0], c_line[0]], fontsize=12, loc=(.15,.9))
    plt.ylabel('Speed (mm/s)')
    plt.xlabel('Time (s)')
    # plt.twinx()
    # t_co = np.arange(0, trial_len, dt) 
    # lns.append(plt.plot(t_co, trial_co[:,0]))
    # lns.append(plt.plot(t_co, trial_co[:,1]))
    # plt.legend(lns, ['Cursor Speed', 'Controller 1', 'Controller 2'])
    # plt.ylabel('Controller Value')
    ymin, ymax = plt.ylim()
    # plt.xlim([0, trial_len])
    # plt.ylim([ymin, ymax])

    #return t, filt_speed, transitions

def get_normal_accel(x_vel, y_vel, t):
    """
    computes normal acceleration at each time point
    """
    x_vel = savgol_filter(x_vel, 101, 2)
    y_vel = savgol_filter(y_vel, 101, 2)
    x_accel = np.gradient(x_vel, t)
    y_accel = np.gradient(y_vel, t)

    speed = np.sqrt(x_vel**2 + y_vel**2)
    x_tang_velocity = x_vel/speed
    y_tang_velocity = y_vel/speed
    normal_accel = np.sqrt(np.gradient(x_tang_velocity, t)**2 + np.gradient(y_tang_velocity, t)**2)

    return normal_accel

def speed_maxima(x_vel, y_vel):
    '''Calculates submovements from velocity coordinates, based on 
    speed profile
    
    Parameters
    x_vel: velocity in x-axis
    y_vel: velocity in y-axis
    
    Returns
    intervals: List of minima defining starts/ends of submovements
    '''
    speed = utils.get_speed(x_vel, y_vel)
    minima, _  = signal.find_peaks(speed,
                                  height=200, prominence=100)
    
    # prominence = 100
    # bounds = np.append(minima, len(speed) - 1)
    # minima = [minima[i] for i in range(len(minima))
    #           if speed[bounds[i]:bounds[i+1]].max() - speed[minima[i]] > prominence]
    
    return np.array(minima)

if __name__=='__main__':
    pass 