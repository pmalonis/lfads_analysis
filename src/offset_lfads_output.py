import h5py
import numpy as np
from scipy import io
import os
import sys
import yaml
import utils

if __name__=='__main__':
    config_path = '../config.yml'
    cfg = yaml.safe_load(open(config_path, 'r'))
    run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
    datasets = run_info.keys()
    for dataset in datasets:
        param = open('../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read()
        input_info = io.loadmat('../data/model_output/%s_inputInfo.mat'%dataset)
        with h5py.File('../data/model_output/%s_%s_all.h5'%(dataset,param),'r') as h5file:
            dt = utils.get_dt(h5file, input_info)

            offset_idx = np.round(cfg['neural_offset']/dt).astype(int)
            new_filename='../data/model_output/%s_%s-offset_all.h5'%(dataset,param)
            with h5py.File(new_filename) as new_h5file:
                for k,v in h5file.items():
                    new_h5file[k] = v[:,offset_idx:,:]    