import h5py
import numpy as np
from scipy.io import loadmat

if __name__=='__main__':

    train_filename = snakemake.input[0]
    valid_filename = snakemake.input[1]
    inputInfo_filename = snakemake.input[2]
    output_filename = snakemake.output[0]

    input_info = loadmat(inputInfo_filename)
    valid_inds = input_info['validInds'][0] - 1
    train_inds = input_info['trainInds'][0] - 1

    n_trials = np.max(np.concatenate([valid_inds, train_inds])) + 1

    copy_datasets = ['controller_outputs', 'output_dist_params', 'factors'] #datasets to copy
    with h5py.File(train_filename, 'r') as train_file, h5py.File(valid_filename, 'r') as valid_file, h5py.File(output_filename, 'w') as output_file:
        for dataset in copy_datasets:
            # skipping 'controller_outputs' if controller dimension is set to 0
            if dataset == 'controller_outputs' and dataset not in train_file.keys():
                continue
                
            output_file.create_dataset(dataset, shape=(n_trials,)+train_file[dataset].shape[1:])
            output_file[dataset][valid_inds,:,:] = valid_file[dataset][:,:,:]
            output_file[dataset][train_inds,:,:] = train_file[dataset][:,:,:]

        