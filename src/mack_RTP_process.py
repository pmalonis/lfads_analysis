import numpy as np
from scipy import io
from glob import glob
from parse import parse

if __name__=='__main__':
    directory = '../data/raw/mk080111M1m/'
    spike_files = glob(directory + 'spikeChan*')
    out_dict = {}
    for spike_file in spike_files:
        spike_data = io.loadmat(spike_file)
        chan_number = parse(directory + 'spikeChan-{num:d}.mat', spike_file)['num']
        out_dict['Chan%03da'%chan_number] = spike_data['spikeChan']['ts'][0,0]

    kin_dict = io.loadmat(directory + 'unitdata.mat')
    out_dict['x'] = kin_dict['x']
    out_dict['y'] = kin_dict['y']
    st_trial = kin_dict['events'][np.where(kin_dict['events']['label']=='sTrial')]['times'][0].flatten()
    e_trial = kin_dict['events'][np.where(kin_dict['events']['label']=='eTrial')]['times'][0].flatten()
    st_trial = np.array([st_trial[np.argmin(et-st_trial[st_trial<et])] for et in e_trial]) 
    out_dict['st_trial'] = st_trial
    out_dict['endmv'] = e_trial
    out_dict['cpl_st_trial_rew'] = np.array([st_trial, e_trial]).T
    out_dict['hit_target'] = kin_dict['events'][np.where(kin_dict['events']['label']=='hitTarg')]['times'][0]

    io.savemat(directory + 'mk08011M1m.mat', out_dict)
