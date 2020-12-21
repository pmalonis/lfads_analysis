import numpy as np
import pandas as pd
import os
import h5py
from scipy import io

def gini(c):
    '''Computes Gini coefficient of vector c'''
    c = np.sort(np.abs(c-np.mean(c)))
    N = len(c)
    G = 1 - 2*(c/np.sum(c)).dot((N - np.arange(1,N+1) + 1/2)/N)

    return G

def subsample_trials(mat_data, trials):
    subsampled = mat_data.copy()
    subsampled['cpl_st_trial_rew'] = subsampled['cpl_st_trial_rew'][trials]

    return subsampled

if __name__=='__main__':
    raw_path = '../data/raw/rockstar.mat'
    np.random.seed(3610)
    subsamples = np.array([200,300,400,500,600])
    data = io.loadmat(raw_path)
    ntrials = len(data['cpl_st_trial_rew'])
    trial_list = np.random.permutation(np.arange(ntrials))
    for subsample in subsamples:
        sub_data = subsample_trials(data, trial_list[:subsample])
        io.savemat('../data/raw/%d_trial_rockstar.mat'%subsample, sub_data)