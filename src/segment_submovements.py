import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

min_submovement_ms = 50


def trial_submovements(x_vel, y_vel):
    '''Calculates submovements from single trial, based on 
    speed profile
    
    Parameters
    trial_df: single-trial dataframe
    
    Returns
    
    '''
    

    return 

def get_submovements(x_vel, y_vel):
    '''Calculates submovements from velocity coordinates, based on 
    speed profile
    
    x_vel: velocity in x-axis
    y_vel: velocity in y-axis'''

    speed = np.sqrt(x_vel**2, y_vel**2)
    minima = signal.find_peaks(-speed, width=min_submovement_ms)
    
    return minima

def plot_submovements(trial_df, segments=None):
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
    ax.plot(*targets[:,0], 'k.')
    ax.plot(*targets[:,1:], 'bs')
    xs, ys = targets[:,:-1]
    dxs, dys = np.diff(targets, axis=1)
    arrow_shorten = 1
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        arrow_len = np.linalg.norm([dx, dy])
        c = (arrow_len - arrow_shorten)/arrow_len
        plt.arrow(x, y, dx*c, dy*c, width=.00001, head_width=1, length_includes_head=True, alpha=.2)

    #TODO plot segments

if __name__=='__main__':
    pass