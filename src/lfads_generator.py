#%%
import h5py
import sys
from scipy.io import loadmat
sys.path.insert(0, '/home/pmalonis/models/research/lfads/')
from lfads import GenGRU
import tensorflow as tf
import json
import numpy as np

param_file = "/home/pmalonis/lfads_analysis/data/model_output/no_controller_params.mat"
push_mean_file = "/home/pmalonis/lfads_analysis/data/model_output/model_runs_rockstar.h5_train_posterior_push_mean"

params = loadmat(param_file)
W_hru=params['gen_gengru_h_to_ru_W']
b_hru=params['gen_gengru_h_to_ru_b']
W_rhc=params['gen_gengru_rh_to_c_W']
b_rhc=params['gen_gengru_rh_to_c_b']
g = params['prior_g0_mean'] 

with tf.Session() as sess:
    with h5py.File(push_mean_file,'r') as f:
        hps = json.load(open('/home/pmalonis/lfads_analysis/data/model_output/hyperparameters-0.txt'))
        gen = GenGRU(hps['gen_dim'],input_weight_scale=hps['gen_cell_input_weight_scale'],rec_weight_scale=hps['gen_cell_input_weight_scale'],clip_value=hps['cell_clip_value']) 
        h,_ = gen(None, tf.Variable(np.array([f['gen_ics'][0,:]], dtype=np.float32)))
        gen_states = np.zeros(f['gen_states'].shape[1:])
        for i in range(f['gen_states'].shape[1]):
            g = h.eval({'GenGRU/Gates/h_2_ru/W:0':W_hru, 'GenGRU/Gates/h_2_ru/b:0':b_hru, 'GenGRU/Candidate/rh_2_c/b:0':b_rhc, 'GenGRU/Candidate/rh_2_c/W:0':W_rhc,'Variable:0':g})
            gen_states[i,:] = g
            