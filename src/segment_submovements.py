import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

min_submovement_ms = 50

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

def trial_transitions(trial_df, exclude_post_target=None):
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

    return minima[non_post_target_events]

def dataset_events(df, func, column_label='transition', exclude_post_target=0.3):
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
    using colors to distinguish segments of the tragjectory

    Parameters
    trial_df: single-trial dataframe corresponding to trial to plot
    segments: list of tuples containing segments corresponding to the start and stop
    times of different submovements. The different sub

    Returns
    ax: axis object containing plot
    '''
    targets = trial_df.kinematic.query('hit_target')
    targets = targets.iloc[1:][['x', 'y']].values.T
    f, ax = plt.subplots(figsize=(8,6))
    ax.plot(*targets[:,0], 'ro')
    ax.plot(*targets[:,1:], 'bs')
    xs, ys = targets[:,:-1]
    dxs, dys = np.diff(targets, axis=1)
    arrow_shorten = 1
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        arrow_len = np.linalg.norm([dx, dy])
        c = (arrow_len - arrow_shorten)/arrow_len
        plt.arrow(x, y, dx*c, dy*c, width=.00001, head_width=1, length_includes_head=True, alpha=.2)

    intervals = trial_submovements(trial_df)
    x, y = trial_df.kinematic[['x', 'y']].values.T
    ax.plot(x, y, 'k')
    for interval in intervals:
        pos = trial_df.kinematic[['x', 'y']].loc[interval[0]:interval[1]].values.T
        ax.plot(*pos)

    return ax

if __name__=='__main__':
    pass