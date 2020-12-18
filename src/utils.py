import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt

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

    trial_len_ms = input_info['seq_timeVector'][0][-1]
    nbins = lfads_h5file['controller_outputs'].shape[1]
    dt_ms = np.round(trial_len_ms/nbins)
    dt = dt_ms/1000

    return dt

def get_trial_len(lfads_h5file, input_info):
    '''Gets trial length'''
    dt = get_dt(lfads_h5file, input_info)
    trial_len_ms = input_info['seq_timeVector'][-1][-1]
    trial_len = trial_len_ms/1000
    trial_len = np.floor(trial_len/dt)*dt

    return trial_len

def polar_hist(data, N):
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
    counts,_ = np.histogram(data, bins)
    bin_centers = (bins[:-1] + bins[1:])/2
    width = 2*np.pi/N
    ax = plt.subplot(111, polar=True)
    ax.bar(bin_centers, counts, width=width)
    
    return ax