import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
from matplotlib import animation
from scipy import io
import h5py
import os
from glob import glob
from utils import get_indices


def animate_trial(x, y, target_pos, fs=5):
    '''Creates matplotlib animation of data for single trial. The animation
    shows the cursor position over time as a dot, and the target positions as a square. 
    in addition, a scrolling plot of any variable is also plotted
    
    Parameters
    mat_data:  data dictionary originating from raw data .mat file
    trial_index: index of trial in mat_data['cpl_st_trial_rew'] to use
    scroll_data: Data to include in scrolling plot. Each row represents a different time series
    that will be included in the scrolling plot
    variable_dt: Sampling time of the scrolled data
    title: Title to display on animation from kinematics
    '''

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    cursor_ln, = ax.plot([], [], 'r.', markersize=8)
    target_ln, = ax.plot([], [], marker='s', color='b', markersize=8)
    
    ax.set_xlim(np.min(x)-np.min(x)*0.05, np.max(x)+np.max(x)*0.05)
    ax.set_ylim(np.min(y)-np.min(y)*0.05, np.max(y)+np.max(y)*0.05)

    def init():
        cursor_ln.set_data([], [])
        target_ln.set_data([], [])
        return cursor_ln, target_ln

    def animate(t):
        '''Animates kinematic frame at time t by plotting cursor position and, if t
        is a time during a trial, the target position'''

        # plotting cursor
        cursor_ln.set_data(x[t], y[t])

        target_ln.set_data(*target_pos[t])


        return cursor_ln, target_ln

    assert(len(x)==len(y)==len(target_pos))
    frames = len(x)
    interval = 1/fs * 1000
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                    init_func=init, blit=True, interval=interval)

    return anim

