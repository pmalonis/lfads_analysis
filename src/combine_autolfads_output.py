import os
import h5py
import numpy as np
from scipy.io import savemat
import yaml

def combine_datasets(train_file, valid_file, input_file, dataset):
    valid_inds = input_file['valid_inds'][:]
    train_inds = input_file['train_inds'][:]
    valid_data = valid_file[dataset][:]
    train_data = train_file[dataset][:]
    all_data = np.zeros((len(valid_inds) + len(train_inds),) 
                                    + valid_data.shape[1:])
    all_data[valid_inds] = valid_data
    all_data[train_inds] = train_data

    return all_data

def create_begin_window(chunk_length_bins, overlap_bins):
    x = np.linspace(0, 1, overlap_bins)
    begin = x**2 * np.ones(overlap_bins)
    pad = np.ones(chunk_length_bins - overlap_bins)

    return np.concatenate([begin, pad])

def create_end_window(chunk_length_bins, overlap_bins):
    x = np.linspace(0, 1, overlap_bins)
    end = (1 - x**2) * np.ones(overlap_bins)
    pad = np.ones(chunk_length_bins - overlap_bins)

    return np.concatenate([pad, end])

if __name__=='__main__':

    # train_filename = snakemake.input[0]
    # valid_filename = snakemake.input[1]
    # input_filename = snakemake.input[2]
    # output_filename = snakemake.output[0]
    
    train_filename = '../data/model_output/new_autolfads/600msChunk200msOverlap_full_data_3_epochs/lfadsOutput/model_runs_rockstar.h5_train_posterior_sample_and_average'
    valid_filename = '../data/model_output/new_autolfads/600msChunk200msOverlap_full_data_3_epochs/lfadsOutput/model_runs_rockstar.h5_valid_posterior_sample_and_average'
    input_filename = '../data/raw/for_autolfads/rockstar_full_600ms/lfads_rockstar.h5'
    output_filename = '../data/model_output/rockstar_autolfads-full-data-3-epochs_all.h5'
    matInfo_filename = '../data/model_output/long_rockstar_inputInfo.mat'

    # train_filename = '/home/macleanlab/peter/lfads_analysis/data/model_output/split_rockstar_full_600ms200ms_overlap/param_FDCWrX/single_rockstar/lfadsOutput/model_runs_rockstar.h5_train_posterior_sample_and_average'
    # valid_filename = '/home/macleanlab/peter/lfads_analysis/data/model_output/split_rockstar_full_600ms200ms_overlap/param_FDCWrX/single_rockstar/lfadsOutput/model_runs_rockstar.h5_valid_posterior_sample_and_average'
    # input_filename = '/home/macleanlab/peter/lfads_analysis/data/raw/for_autolfads/rockstar_full_600ms/lfads_rockstar.h5'
    # output_filename = '/home/macleanlab/peter/lfads_analysis/data/model_output/rockstar_lfads-full-data_all.h5'
    # matInfo_filename = '/home/macleanlab/peter/lfads_analysis/data/model_output/full_lfads_rockstar_inputInfo.mat'

    config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))
    dt = cfg['autolfads_time_bin']
    chunk_length = cfg['data_chunking']['chunk_length']
    overlap = cfg['data_chunking']['overlap']
    step = chunk_length - overlap
    chunk_length_bins = np.round(chunk_length/dt).astype(int)
    overlap_bins = np.round(overlap/dt).astype(int)
    step_bins = np.round(step/dt).astype(int)
    begin_window = create_begin_window(chunk_length_bins, overlap_bins)
    end_window = create_end_window(chunk_length_bins, overlap_bins)
    with h5py.File(input_filename) as input_file:
        valid_inds = input_file['valid_inds'][:]
        train_inds = input_file['train_inds'][:]

    n_trials = np.max(np.concatenate([valid_inds, train_inds])) + 1

    copy_datasets = ['controller_outputs', 'output_dist_params', 'factors', 'gen_states'] #datasets to copy
    with h5py.File(train_filename, 'r') as train_file, \
    h5py.File(valid_filename, 'r') as valid_file, \
    h5py.File(output_filename, 'w') as output_file,\
    h5py.File(input_filename, 'r') as input_file:
    
        trials = input_file['all_trials'][:]
        n_trials = input_file['all_trials'][:].max() + 1

        trial_chunk_counts,_ = np.histogram(trials,np.arange(len(np.unique(trials))+1))
        if np.all(trial_chunk_counts == trial_chunk_counts[0]):
            are_trials_equal = True
            chunks_per_trial = trial_chunk_counts[0]
        else:
            are_trials_equal = False

        for dataset in copy_datasets:
            # skipping 'controller_outputs' if controller dimension is set to 0
            if dataset == 'controller_outputs' and dataset not in train_file.keys():
                continue
            
            all_data_chunked = combine_datasets(train_file, valid_file, input_file, dataset)
            
            n_dims = all_data_chunked.shape[2]
            if are_trials_equal:
                trial_len_bins = chunk_length_bins * chunks_per_trial - overlap_bins*(chunks_per_trial-1)
                combined_data = np.zeros((n_trials, trial_len_bins, n_dims))
            else:
                combined_data = output_file.create_group(dataset)
            
            for trial_idx in range(n_trials):
                trial_data = all_data_chunked[trial_idx == trials, :, :].copy()
                # applying windows 
                trial_data[1:,:] = (trial_data[1:,:].swapaxes(1,2) * begin_window).swapaxes(1,2)
                trial_data[:-1,:] = (trial_data[:-1,:].swapaxes(1,2) * end_window).swapaxes(1,2)
                n_chunks = trial_data.shape[0]
                trial_len_bins = chunk_length_bins * n_chunks - overlap_bins*(n_chunks-1)
                padded_data = np.zeros((n_chunks, trial_len_bins, n_dims))
                row_not_zero = [[i] for i in range(n_chunks)]
                col_not_zero = [range(i*step_bins, i*step_bins + chunk_length_bins) for i in range(n_chunks)]
                padded_data[row_not_zero, col_not_zero,:] = trial_data
                combined_trial = padded_data.sum(0)
                if are_trials_equal:
                    combined_data[trial_idx,:] = combined_trial
                else:
                    combined_data.create_dataset('trial_%03d'%trial_idx, data=combined_trial)

            if are_trials_equal:
                output_file.create_dataset(dataset, data=combined_data)
        
        if are_trials_equal:
            input_info = {'trial_len':trial_len_bins * dt, 
                            'autolfads':True,
                            'trial_cutoff':3.86,
                            'trainInds': train_inds+1,
                            'validInds': valid_inds+1,
                            'seq_binSizeMs': np.array([[1]], dtype=np.uint8),
                            'conditionID': np.ones((all_data_chunked.shape[0],1), dtype=np.uint8),
                            'seq_timeVector': np.arange(0, all_data_chunked.shape[1]*dt, dt/1000)}

            savemat(matInfo_filename, input_info)