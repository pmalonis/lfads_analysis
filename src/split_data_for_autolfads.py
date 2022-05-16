import numpy as np
from scipy import io
import h5py
from sklearn.model_selection import train_test_split
import yaml
import os

random_state = 943
config_path = '../config.yml'
cfg = yaml.safe_load(open(config_path, 'r'))
run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
dt = 0.01
if __name__ == '__main__':
    datasets = list(cfg['datasets'].keys())
    for dataset in datasets:
        filename = os.path.basename(cfg['datasets'][dataset]['raw'])
        file_path = '../data/raw/' + filename
        matdata = io.loadmat(file_path)
        T = np.diff(matdata['cpl_st_trial_rew'].real, axis=1).flatten()
        T = np.sort(T)
        total = np.zeros(len(T))
        for i in range(len(T)):
            total[i] = (len(T)-i)*T[i] #total length of dataset if T[i]  used as cutoff

        idx = np.argmax(total)
        trial_cutoff = T[idx]
        n_trials = len(T) - idx
        print(trial_cutoff)

        neurons = [n for n in matdata.keys() if n[:4] == 'Chan']

        n_neurons = (len(neurons))
        n_bins = int(trial_cutoff//dt)

        data = np.zeros((n_trials, n_bins, n_neurons))
        trial_lengths = np.diff(matdata['cpl_st_trial_rew'].real, axis=1).flatten()
        used_inds = np.where(trial_lengths >= trial_cutoff)
        used_trials = matdata['cpl_st_trial_rew'][used_inds].real
        assert(len(used_trials)==n_trials)
        trial_len = n_bins * dt
        bins = np.arange(0, trial_len + dt, dt)
        for i in range(n_trials):
            start = used_trials[i][0]
            for j,neuron in enumerate(neurons):
                spk = matdata[neuron].real.flatten()
                trial_spk = spk[(spk >= start) & (spk < start + trial_len)] - start
                data[i,:,j],_ = np.histogram(trial_spk, bins=bins)

        train_inds, valid_inds = train_test_split(range(n_trials), test_size=0.2, 
                                                    random_state=random_state)
        
        train_inds = np.sort(train_inds)
        valid_inds = np.sort(valid_inds)
        with h5py.File('../data/raw/for_autolfads/%s.h5'%dataset,'w') as h5file:
            h5file.create_dataset('train_data', data=data[train_inds,:,:])
            h5file.create_dataset('valid_data', data=data[valid_inds,:,:])
            h5file.create_dataset('train_inds', data=train_inds.reshape((1,-1)))
            h5file.create_dataset('valid_inds', data=valid_inds.reshape((1,-1)))
            h5file.close()