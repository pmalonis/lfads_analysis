import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

min_submovement_ms = 25

def trial_intervals(trial_df):
    '''Calculates submovement transitions from single trial, based on 
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

def angle_difference(a, b):
    '''Difference between two angles, which are
    given in randients between -pi and pi a is reference'''

    #converting to 0 to 2pi
    a += np.pi
    b += np.pi
    angle_diff = (b - a)%(2*np.pi)
    angle_diff -= 2*np.pi
    angle_diff[angle_diff<=-np.pi] = angle_diff[angle_diff<=-np.pi]%np.pi
    return angle_diff

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

def dataset_events(df, func, column_label='transition', exclude_post_target=0.2, exclude_pre_target=0.2):
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

def speed_minima(x_vel, y_vel):
    '''Calculates submovements from velocity coordinates, based on 
    speed profile
    
    Parameters
    x_vel: velocity in x-axis
    y_vel: velocity in y-axis
    
    Returns
    intervals: List of minima defining starts/ends of submovements
    '''
    speed = np.sqrt(x_vel**2, y_vel**2)
    minima, _ = signal.find_peaks(-speed, width=min_submovement_ms)
    
    return minima

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
        plt.arrow(x, y, dx*c, dy*c, width=.00001, head_width=5, length_includes_head=True, alpha=.2)

    intervals = trial_intervals(trial_df)
    x, y = trial_df.kinematic[['x', 'y']].values.T
    ax.plot(x, y, 'k')
    for interval in intervals:
        pos = trial_df.kinematic[['x', 'y']].iloc[interval[0]:interval[1]].values.T
        ax.plot(*pos)

    return ax

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
    ax.plot(*targets[:,0], 'ro')
    ax.plot(*targets[:,1:], 'bs')
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
    plt.scatter(x, y, s=10, c=trial_co, cmap=cm, norm=norm)
    cb = plt.colorbar()
    cb.set_label('Controller Value')

def plot_trial(trial_df, trial_co, dt):
    '''
    Plots the speed profile, controller outputs, and target times
    
    Parameters
    trial_df: single-trial dataframe corresponding to trial to plot
    trial_co: controller outputs for one trial (time X n_outputs)
    '''
    trial_len = trial_co.shape[0] * dt
    x_vel, y_vel = trial_df.kinematic[['x_vel', 'y_vel']].loc[:trial_len].T.values
    speed = np.sqrt(x_vel**2, y_vel**2)
    t_targets = trial_df.kinematic.query('hit_target').index.values
    t = trial_df.loc[:trial_len].index.values
    lns = [] #for legend
    transitions = trial_transitions(trial_df)
    trial_len_ms = trial_len*1000
    transitions = transitions[transitions < trial_len_ms]
    plt.figure(figsize=(12,6))
    lns.append(plt.plot(t, speed,'g'))
    plt.plot(t[transitions], speed[transitions],'r.')
    plt.ylabel('Speed (mm/s)')
    plt.xlabel('Time (s)')
    plt.twinx()
    t_co = np.arange(0, trial_len, dt) 
    lns.append(plt.plot(t_co, trial_co[:,0]))
    lns.append(plt.plot(t_co, trial_co[:,1]))
    plt.legend(lns, ['Cursor Speed', 'Controller 1', 'Controller 2'])
    plt.ylabel('Controller Value')
    ymin, ymax = plt.ylim()
    plt.vlines(t_targets, ymin, ymax)

if __name__=='__main__':
    pass 