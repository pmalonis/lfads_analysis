import numpy as np
from scipy import io
import h5py
from sklearn.model_selection import train_test_split
import yaml
import os

random_state = 943
round_factor = 100000 #for rounding to avoid floating point error
config_path = '../config.yml'
cfg = yaml.safe_load(open(config_path, 'r'))
run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
dt = cfg['autolfads_time_bin']
chunk_length = cfg['data_chunking']['chunk_length']
overlap = cfg['data_chunking']['overlap']
step = chunk_length - overlap

def get_n_chunks(trial_length, overlap=overlap, step=step):
    n_chunks = np.round((trial_length - overlap)/step * round_factor) / round_factor
    n_chunks = np.floor(n_chunks).astype(int)

    return n_chunks

if __name__ == '__main__':
    #datasets = list(cfg['datasets'].keys())
    datasets = [list(cfg['datasets'].keys())[0]]
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

        trial_lengths = np.diff(matdata['cpl_st_trial_rew'].real, axis=1).flatten()
        used_inds = np.where(trial_lengths >= trial_cutoff)
        matdata['cpl_st_trial_rew'] = matdata['cpl_st_trial_rew'][used_inds].real
        matdata['cpl_st_trial_rew'][:,1] = matdata['cpl_st_trial_rew'][:,0] + trial_cutoff

        neurons = [n for n in matdata.keys() if n[:4] == 'Chan']

        n_neurons = len(neurons)
        n_bins = int(np.round(chunk_length/dt * round_factor) / round_factor)
        
        all_n_chunks = [get_n_chunks(t[1]-t[0]) for t in matdata['cpl_st_trial_rew'].real]
        data = np.zeros((np.sum(all_n_chunks), n_bins, n_neurons))
        bins = np.arange(0, chunk_length + dt, dt)

        trials = np.concatenate([[i]*nchunks for i,nchunks in enumerate(all_n_chunks)])
        chunk_indices = [] #index of chunk within trial
        k = 0
        for i in range(n_trials):
            n_chunks = all_n_chunks[i]
            for idx_chunk in range(n_chunks):
                start = matdata['cpl_st_trial_rew'][i][0].real + step * idx_chunk
                chunk_indices.append(idx_chunk)
                for j,neuron in enumerate(neurons):
                    spk = matdata[neuron].real.flatten()
                    trial_spk = spk[(spk >= start) & (spk < start + chunk_length)] - start
                    data[k,:,j],_ = np.histogram(trial_spk, bins=bins)
                
                k += 1

        # trials_with_chunk_inds = np.array(list(zip(trials, chunk_indices)),
        #                                 dtype=[('trials', '<i4'), ('chunk_inds', '<i4')])

        # train_inds, valid_inds, train_trials, valid_trials = train_test_split(np.arange(data.shape[0]), 
        #                                                                     trials_with_chunk_inds, 
        #                                                                     test_size=0.2, 
        #                                                                     random_state=random_state)

        train_inds, valid_inds = train_test_split(np.arange(data.shape[0]),
                                                            test_size=0.2, 
                                                            random_state=random_state)

        order_train = np.argsort(train_inds)
        order_valid = np.argsort(valid_inds)
        train_trials = train_trials[order_train]
        valid_trials = valid_trials[order_valid]
        train_inds = train_inds[order_train]
        valid_inds = valid_inds[order_valid]
        with h5py.File('../data/raw/for_autolfads/%s/lfads_%s-truncated.h5'%(dataset,dataset),'w') as h5file:
            h5file.create_dataset('train_data', data=data[train_inds,:,:])
            h5file.create_dataset('valid_data', data=data[valid_inds,:,:])
            h5file.create_dataset('train_inds', data=train_inds)
            h5file.create_dataset('valid_inds', data=valid_inds)
            h5file.create_dataset('train_trials', data=train_trials)
            h5file.create_dataset('valid_trials', data=valid_trials)
            h5file.create_dataset('all_trials', data=trials)
            h5file.close()