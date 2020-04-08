#!/usr/bin/env python
# coding: utf-8

# %%
import h5py
from scipy import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_filename = "../data/intermediate/rockstar.p"
inputInfo_filename = "../data/model_output/rockstar_inputInfo.mat"
valid_filename = "../data/model_output/rockstar_valid.h5"

df = pd.read_pickle(data_filename)

input_info = io.loadmat(inputInfo_filename)
trial_len = input_info['seq_timeVector'][-1][-1]/1000
dt = np.round(trial_len/386 * 1000)/1000
t = np.arange(0, np.floor(trial_len/dt)) * dt

inds = input_info['trainInds'][0] - 1
n = df.neural.shape[1]
with h5py.File(valid_filename) as h5file:
    for trial in range(len(inds)):
        trial_df = df.loc[trial]
        for neuron_idx in range(n):
            plt.figure()
            plt.plot(t, h5file['output_dist_params'][inds[trial],:,neuron_idx])
            neuron = trial_df.neural.columns[neuron_idx]
            plt.vlines(trial_df.neural.loc[:trial_len].query(neuron)[neuron].index.values, 0, 0.1)
            plt.ylim([0, 0.1])
#df = pd.read_pickle(data_filename)
