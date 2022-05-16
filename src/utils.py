from scipy.signal import savgol_filter
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
from scipy import io
import h5py
from matplotlib import colors
import colorsys
import seaborn as sns

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def load_data(data_filename, lfads_filename, inputInfo_filename):
    '''loads data from filenames'''
    df = pd.read_pickle(data_filename)
    input_info = io.loadmat(inputInfo_filename)
    with h5py.File(lfads_filename, 'r') as h5file:
        co = h5file['controller_outputs'][:]
        dt = get_dt(h5file, input_info)
        trial_len = get_trial_len(h5file, input_info)
    
    return df, co, trial_len, dt

def inputs_to_model(dataset, event_type, split='train'):
    file_root = dataset
    lfads_params = open(os.path.dirname(__file__) + '../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric'])).read().strip()
    data_filename = os.path.dirname(__file__)+'../data/intermediate/' + file_root + '.p'
    lfads_filename = os.path.dirname(__file__)+'../data/model_output/' + \
                    '_'.join([file_root, lfads_params, 'all.h5'])
    inputInfo_filename = os.path.dirname(__file__)+'../data/model_output/' + \
                        '_'.join([file_root, 'inputInfo.mat'])
    peak_filename = os.path.dirname(__file__)+'../data/peaks/' + \
                    '_'.join([file_root, '%s_%s.p'%(event_type,split)])

    peak_df = pd.read_pickle(peak_filename)
    df, co, trial_len, dt = load_data(data_filename, lfads_filename, inputInfo_filename)

    return peak_df, co, trial_len, dt, df

def print_commit():
    '''saves matplotlib figure with hash of current git commit as metadata'''
    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    print(commit)

def git_savefig(fig, filename):
    '''saves matplotlib figure with hash of current git commit as metadata'''
    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    fig.savefig(filename, metadata={'commit':commit})

def get_indices(input_info, trial_type):
    if trial_type == 'train':
        used_inds = input_info['trainInds'][0] - 1
    elif trial_type == 'valid':
        used_inds = input_info['validInds'][0] - 1
    elif trial_type == 'all':
        used_inds = np.sort(np.concatenate([input_info['trainInds'][0] - 1, input_info['validInds'][0] - 1]))

    return used_inds

def get_dt(lfads_h5file, input_info):
    '''Gets LFADS time bin size'''
    if input_info.get('autolfads'):
        return cfg['autolfads_time_bin']
    else:
        trial_len_ms = input_info['seq_timeVector'][0][-1]
        if 'factors' in lfads_h5file.keys():
            nbins = lfads_h5file['factors'].shape[1]
        elif 'controller_outputs' in lfads_h5file.keys():
            nbins = lfads_h5file['controller_outputs'].shape[1]
            
        dt_ms = np.round(trial_len_ms/nbins)
        dt = dt_ms/1000

    return dt

def get_trial_len(lfads_h5file, input_info):
    '''Gets trial length'''
    if input_info.get('autolfads'):
        return input_info['trial_len'][0][0]
    else:        
        dt = get_dt(lfads_h5file, input_info)
        trial_len_ms = input_info['seq_timeVector'][-1][-1]
        trial_len = trial_len_ms/1000
        trial_len = np.floor(trial_len/dt)*dt

    return trial_len

def polar_hist(data, N, density=True, ax=None):
    '''
    Plots polar histogram
    
    Parameters:
    data: data to plot in histogram, must be in radians
    N: number of bins in histogram

    Returns:
    ax: axis with polar histogram
    '''
    data = data%(2*np.pi)
    bins = np.linspace(0, 2*np.pi, N+1)
    counts,_ = np.histogram(data, bins, density=density)
    bin_centers = (bins[:-1] + bins[1:])/2
    width = 2*np.pi/N
    if ax is None:
        ax = plt.subplot(111, polar=True)
    
    ax.bar(bin_centers, counts, width=width)
    return ax

def spoke_plot(x, y, labels=['x','y'], ax=None, color=[0,.4,.4,.6]):

    if ax is None:
        ax = plt.subplot(111, polar=True)

    ax.set_ylim([0, 2])
    ax.set_yticks([1, 2])
    ax.set_yticklabels(labels)
    ax.set_xticks([])

    for xi, yi in zip(x,y):
        ax.plot([xi, yi], [1, 2], color=color)

    return ax

def contrasting_colors(start_hue, n_colors=2, hue_change=1, reverse=False, saturation=0.6, min_lightness=0.25, max_lightness=0.75):
    hues = [(start_hue + hue_change*i/n_colors)%1 for i in range(n_colors)]
    lightness = np.linspace(0.25, 0.75, n_colors)
    if reverse:
        lightness = lightness[::-1]
    hls_colors = [[h, l, saturation] for h,l in zip(hues, lightness)]
    rgb_colors = [colorsys.hls_to_rgb(*hls) for hls in hls_colors]
    return rgb_colors

# def get_firing_rates(df, dt, spike_dt):
#     nneurons = sum('neural' in c for c in df.columns)
#     std = cfg['target_decoding_smoothed_control_std']
#     win = int(dt/spike_dt)
#     midpoint_idx = int((win-1)/2)
#     all_smoothed = np.zeros((len(used_inds), int(trial_len/dt), nneurons)) #holds firing rates for whole experiment (to be used for dimensionality reduction)
#     for i in range(len(used_inds)):
#         smoothed = df.loc[used_inds[i]].neural.rolling(window=std*4, min_periods=1, win_type='gaussian', center=True).mean(std=std)
#         smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
#         smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
#         all_smoothed[i,:,:] = smoothed

#     return all_smoothed

def get_speed(x_vel, y_vel):
    speed = np.sqrt(x_vel**2 + y_vel**2)
    speed = savgol_filter(speed, cfg['speed_filter_win'], cfg['speed_filter_order'])

    return speed