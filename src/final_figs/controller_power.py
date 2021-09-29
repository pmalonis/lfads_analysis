import pandas as pd
import numpy as np
import h5py
from scipy import io
import yaml
import os
import sys
sys.path.insert(0, '..')
import utils

config_path = os.path.dirname(__file__) + '/../../config.yml'
cfg = yaml.safe_load(open(config_path, 'r'))

if __name__=='__main__':
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    datasets = run_info.keys()
    params = []
    win = .2
    for dataset in datasets:
        param = open('../../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read()
        target_df = pd.read_pickle(os.path.dirname(__file__) + '/../../data/peaks/%s_targets_all.p'%dataset)
        firstmove_df = pd.read_pickle(os.path.dirname(__file__) + '/../../data/peaks/%s_firstmove_all.p'%dataset)
        corrections_df = pd.read_pickle(os.path.dirname(__file__) + '/../../data/peaks/%s_corrections_all.p'%dataset)

        data = pd.read_pickle(os.path.dirname(__file__) + '/../../data/intermediate/%s.p'%dataset)                  
        maxima_df = pd.read_pickle(os.path.dirname(__file__) + '/../../data/peaks/%s_maxima_all.p'%dataset)
        input_info = io.loadmat(os.path.dirname(__file__) + '/../../data/model_output/%s_inputInfo.mat'%dataset)
        
        with h5py.File(os.path.dirname(__file__) + '/../../data/model_output/%s_%s_all.h5'%(dataset,param),'r') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        target_power = 0
        n_target_samples = 0
        n_correction_samples = 0
        corrections_power = 0
        ntrials = co.shape[0]
        nsamples = co.shape[1]
        assert(ntrials == target_df.index[-1][0] + 1)
        #assert(ntrials == corrections_df.index[-1][0] + 1)
        for i in range(ntrials):
            t_targets = target_df.loc[i].index.values
            t_firstmoves = firstmove_df.loc[i].index.values
            t_targets = np.array([t_targets[t_targets < t][-1] 
                                    for t in t_firstmoves])
            if i in corrections_df.index:
                t_corrections = corrections_df.loc[i].index.values
            else:
                t_corrections = np.array([])
            for j in range(nsamples):
                t = dt*j
                if np.any((t - t_targets > 0) & (t - t_targets < win)): #within win seconds after target
                    target_power += (np.abs(co[i,j,:])).sum(0) #(co[i,j,:]**2).sum(0)
                    n_target_samples += 1
                elif np.any((t_corrections - t > 0) & (t_corrections - t < win)): #within win seconds before correction
                    corrections_power += (np.abs(co[i,j,:])).sum(0)#(co[i,j,:]**2).sum(0)
                    n_correction_samples += 1
                    
        total_samples = co.size
        #total_power = np.sum(co**2)
        total_power = np.sum(np.abs(co))
        target_power_p = target_power/total_power
        correction_power_p = corrections_power/total_power
        target_time_p = n_target_samples/total_samples
        correction_time_p = n_correction_samples/total_samples
        print('%s:'%dataset)
        print('Target windows are %.3f of controller power, %.3f of time.'%(target_power_p, target_time_p))
        print('Correction windows are %.3f of controller power, %.3f of time.'%(correction_power_p, correction_time_p))
        print('\n')
        