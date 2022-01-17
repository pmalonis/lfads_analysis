import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    with h5py.File('temp_train.h5', 'w') as trainfile, \
    h5py.File('temp_valid.h5', 'w') as validfile, \
    h5py.File('temp_input.h5', 'w') as input_file:

        trial_size = 380
        sin_ang_freq = 2*np.pi/100 #angular frequency of sine wave used for test data
        n_co = 2
        n_factor = 40
        n_outputs = 100
        n_trials = 500
        chunk_size = 60
        overlap = 20

        def create_data(n_dim, n_trials, trial_size=trial_size, sin_ang_freq=sin_ang_freq, 
                        chunk_size=chunk_size, overlap=overlap):
            
            data = np.array([np.array([np.sin(np.random.rand(1)*2*np.pi + 
                                        sin_ang_freq*np.arange(trial_size)) for j in
                                        range(n_dim)]).T for i in range(n_trials)])
            float_size = 8 #float size in bytes
            step = chunk_size - overlap
            n_chunks = np.round((trial_size - overlap)/step).astype(int)
            chunked_data = as_strided(data, (n_chunks*n_trials, chunk_size, n_dim), (step*float_size*n_dim, float_size*n_dim, float_size))
            return chunked_data, data

        n_chunks = np.round((trial_size - overlap)/(chunk_size-overlap)).astype(int)
        inds = np.arange(n_chunks*n_trials)
        train_inds, valid_inds = train_test_split(inds, test_size=0.2)
        input_file['train_inds'] = train_inds
        input_file['valid_inds'] = valid_inds
        trials = np.concatenate([[i]*n_chunks for i in range(n_trials)])
        input_file['train_trials'] = trials[train_inds]
        input_file['valid_trials'] = trials[train_inds]
        input_file['all_trials'] = trials

        keys = ['controller_outputs', 'factors', 'output_dist_params']
        n_dims = [n_co, n_factor, n_outputs] #dimensions of data for each input above
        
        for key, n_dim in zip(keys, n_dims):
            data,td = create_data(n_dim, n_trials)
            train_data = data[train_inds,:,:]
            valid_data = data[valid_inds,:,:]
            trainfile[key] = train_data
            validfile[key] = valid_data