import numpy as np
import pandas as pd
import h5py
from pylds.models import DefaultPoissonLDS, DefaultLDS
from pickle import dump
from sklearn.model_selection import train_test_split

min_segment = 0.5
lfads_dt = 0.01
bin_size = 50 #time step for lds, in ms
thresh = 0.2 #threshold of summed absolute value of lfads controller outputs
D_latent = 10 #
test_size = 50 #number of holdout segment
random_state = 5
default_std = 100 #default smoothing std, miliseconds
pad_interval = True

def split_segments(counts):
    print("spit function entered")
    train_counts, test_counts = train_test_split(counts, test_size=test_size, random_state=random_state)

    return train_counts, test_counts

def get_forward_prediction(model, test_counts):
    
    predicted_counts = []
    for segment in test_counts:
        p = np.zeros_like(segment)
        obs_init = segment[0,:]
        state_t, _, _, _ = np.linalg.lstsq(model.C, obs_init)
        p[0,:] = model.C.dot(state_t)
        for t in range(1, p.shape[0]):
            state_t = model.A.dot(state_t)
            p[t,:] = state_t.dot(model.C.T)

        predicted_counts.append(p)
    
    return predicted_counts

def train_model(df, co, n_steps, filename=None, bin_size=bin_size):
    
    counts = segment_spike_counts(df, co, bin_size)
    train_counts, test_counts = split_segments(counts)

    n_neurons = df.neural.shape[1]
    model = DefaultPoissonLDS(n_neurons, D_latent)
    for segment in train_counts:
        model.add_data(segment)

    ll = np.zeros(n_steps)
    for i in range(n_steps):
        model.EM_step()
        ll[i] = model.log_likelihood()

    try:
        if filename is not None:
            dump(model, open(filename, 'wb'))
    except:
        pass

    return model, ll

def train_model_gaussian(df, co, n_steps, filename=None, bin_size=bin_size):
    
    counts = segment_spike_counts(df, co, bin_size)
    train_counts, test_counts = split_segments(counts)

    n_neurons = df.neural.shape[1]
    model = DefaultPoissonLDS(n_neurons, D_latent)
    for segment in train_counts:
        model.add_data(segment)

    ll = np.zeros(n_steps)
    for i in range(n_steps):
        model.EM_step()
        ll[i] = model.log_likelihood()

    try:
        if filename is not None:
            dump(model, open(filename, 'wb'))
    except:
        pass
    
    return model, ll

def bin_dataframe(df, bin_size, smooth=False, std=default_std):
    '''if smooth, then downsample according to bin_size'''
    if smooth:
        n_stds = 3
        winsize = n_stds * std * 2 #multiple of 2 for n_stds on each side of the window
        if pad_interval:
            min_periods = None
        else:
            min_periods = 1
        
        return df.rolling(winsize, win_type='gaussian', min_periods=min_periods).sum(std=std).dropna(axis=0).values[::bin_size]
    else:
        return df.rolling(bin_size).sum().values[bin_size-1::bin_size]

def get_trial_segments(signal, thresh, min_segment_idx=None):
    '''Returns two arrays representing the indices of the endpoints of segments where
    signal is less than thresh. If min_segment is given, then only the segments with length
    greater than min_segment are returned'''

    starts = np.where(np.logical_and(signal[:-1] > thresh, signal[1:] <= thresh))[0] + 1
    if signal[0] <= thresh:
        starts = np.insert(starts, 0, 0)

    stops = np.where(np.logical_and(signal[:-1] <= thresh, signal[1:] > thresh))[0] + 1

    if signal[-1] <= thresh:
        stops = np.append(stops, len(signal))

    if min_segment_idx is not None:
        include_idx = stops - starts >= min_segment_idx
        starts = starts[include_idx]
        stops = stops[include_idx]

    return starts, stops
    
def segment_spike_counts(df, co, bin_size=bin_size, smooth=False, std=default_std):
    '''Gets spike counts for segments with minimum length autonomous'''

    assert df.index[-1][0]+1==co.shape[0]
    abs_co = np.abs(co).sum(axis=2)
    min_segment_idx = int(min_segment/lfads_dt)
    counts = []
    for i in range(co.shape[0]):
        starts, stops = get_trial_segments(abs_co[i,:], thresh, min_segment_idx=min_segment_idx)
        for start, stop in zip(starts, stops):
            if start*lfads_dt <= 6*std/1000:
                continue
            if pad_interval:
                padded_start_t = start*lfads_dt - 6*std/1000
                padded_stop_t = stop*lfads_dt
                if padded_start_t < 0:
                    import pdb;pdb.set_trace()
                segment_spikes = df.loc[i].neural.loc[padded_start_t:padded_stop_t]
                binned = bin_dataframe(segment_spikes, bin_size, smooth, std=std)
                counts.append(binned)
            else:
                segment_spikes = df.loc[i].neural.loc[start*lfads_dt:stop*lfads_dt]
                binned = bin_dataframe(segment_spikes, bin_size, smooth, std=std)
                counts.append(binned)

        print('Trial %d of %d processed'%(i+1, co.shape[0]))

    return counts

def random_intervals(df, n_intervals, interval_len, trial_len, bin_size=bin_size, smooth=False, std=default_std):

    '''
    Parameters
    df: data frame containing experiment data
    n_intervals: number of intervals
    interval_len: length of intervals to sample (in seconds)
    trial_len: length of trial given to lfads
    bin_size: bin_size in number of samples (ms)
    '''
    #making sure interval_len is integer multiple of bin_size
    assert (interval_len/(bin_size/1000))%1 == 0

    #cutting out spikes after end of trial
    df=df.groupby('trial').apply(lambda _df: bin_dataframe(_df.loc[_df.index[0][0]].loc[:trial_len].neural, bin_size, smooth, std=std))
    spikes = np.concatenate(df.values)
    k = 0
    intervals = np.zeros((n_intervals, int(interval_len/(bin_size/1000)), spikes.shape[1]))
    bin_size_s = bin_size/1000 #bin_size in seconds
    while k < n_intervals:
        start_idx = np.random.random_integers(0, spikes.shape[0])
        start_t = start_idx * bin_size_s
        stop_t = start_t + interval_len
        # don't use interval if trial boundary in interval
        if np.floor((stop_t-bin_size_s)/(trial_len-bin_size_s)) > np.floor((start_t-bin_size_s)/(trial_len-bin_size_s)):
            continue

        stop_idx = int(np.round(stop_t/(bin_size/1000)))
        interval = spikes[start_idx:stop_idx,:]
        # don't use interval if it overlaps with already sampled interval
        if np.any(np.isnan(interval)):
            continue

        intervals[k, :, :] = interval

        spikes[start_idx:stop_idx,:] = np.nan
        k += 1

    return intervals

if __name__=='__main__':
    trial_type = 'all'
    data_filename = '../data/intermediate/rockstar.p'
    lfads_filename = "/home/pmalonis/226_figs/rockstar_8QTVEk_%s.h5"%trial_type

    with h5py.File(lfads_filename, 'r') as h5file:
        co = np.array(h5file['controller_outputs'])

    df = pd.read_pickle(data_filename)