import numpy as np
from scipy.io import loadmat
import h5py

param_file = "/home/pmalonis/no_controller_params.mat"
push_mean_file = "/home/pmalonis/model_runs_rockstar.h5_train_posterior_push_mean"
forget_bias = 1.0

params = loadmat(param_file)
W_hru=params['gen_gengru_h_to_ru_W']
b_hru=params['gen_gengru_h_to_ru_b']
W_rhc=params['gen_gengru_rh_to_c_W']
b_rhc=params['gen_gengru_rh_to_c_b']


sig = lambda x: 1/(1+np.exp(-x))
with h5py.File(push_mean_file) as push_mean:
    h = np.array([push_mean['gen_ics'][0,:]])
    gen_states = np.zeros_like(push_mean['gen_states'][0,:])
    #gen_states[0,:] = h
    for i in range(gen_states.shape[0]):
        weight_scale = 1#/np.sqrt(len(h))
        r=sig(weight_scale*h.dot(W_hru)[0,:200] + b_hru[0,:200])
        u=sig(weight_scale*h.dot(W_hru)[0,200:] + b_hru[0,200:] + forget_bias)
        c=np.tanh(weight_scale*(r*h).dot(W_rhc) + b_rhc)
        h=u*h + (1-u)*c
        gen_states[i,:] = h