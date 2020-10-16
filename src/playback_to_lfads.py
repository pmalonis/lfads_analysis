# This script processes reads the .mat files for the playback experiment
# and processes them into new mat files format for reading by LFADS. The 
# data from each file is saved as four new .mat files, one for each condition
# of the experiment. The data is formated the same as that for the 
# basic RTP experiment
from scipy import io
import numpy as np
import string

filename = snakemake.input[0]
data = io.loadmat(filename)

used_conditions = ['active', 'vis_pb', 'prob_pb', 'dual_pb']
condition_keys = ['succKinPrePlayback', 'succ visPLBK', 
                  'succ propPLBK', 'succ dualPLBK']

cond = {k[0]:data['conditions']['epochs'][0,i]
        for i,k in enumerate(data['conditions']['label'].flatten()) 
        if k[0] in condition_keys}

condition_dicts = [] #list of dicts that will be saved as mat files
for k in condition_keys:
    condition_dict = {'cpl_st_trial_rew':cond[k]}
    condition_dicts.append(condition_dict)

elect_unit_idx = 0 #index of unit on each electrode
neuron_names = []
for i, unit in enumerate(data['units'][0]):
    neuron_name = 'Chan%03d'%unit['chan'][0]
    if i > 0:
        if neuron_name==neuron_names[-1][:7]:
            elect_unit_idx += 1
        else:
            elect_unit_idx = 0
            
    neuron_name += string.ascii_lowercase[elect_unit_idx]
    neuron_names.append(neuron_name)

    for condition_dict in condition_dicts:
        #eliminating spikes outside range of trials    
        spikes = unit['stamps']
        min_t = np.min(condition_dict['cpl_st_trial_rew'])
        max_t = np.max(condition_dict['cpl_st_trial_rew'])
        spikes = spikes[(spikes >= min_t) & (spikes < max_t)]
        condition_dict[neuron_name] = spikes

for i, condition_dict in enumerate(condition_dicts):
    io.savemat(snakemake.output[0], condition_dict)