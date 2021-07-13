import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optimize_target_prediction as opt
import utils
from importlib import reload
import os
import yaml
from scipy import io
import h5py
from matplotlib import rcParams
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
reload(opt)
reload(utils)


config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

datasets = ['rockstar']#,'raju', 'mack']
params = ['final-fixed-2OLS24']#, 'final-fixed-2OLS24', 'mack-kl-co-sweep-0Wo8i9']
nbins = 12
win_start = cfg['post_target_win_start']
win_stop = 0.25#cfg['post_target_win_stop']
win_len = win_stop - win_start
lfads_filename = '../data/model_output/' + '_'.join([datasets[0], params[0], 'all.h5'])
with h5py.File(lfads_filename, 'r+') as h5file:
    co = h5file['controller_outputs'][:]
    
n_co = co.shape[2]

if __name__=='__main__':
    fig1, axplot = plt.subplots(n_co, len(datasets))
    for i, (dataset, param) in enumerate(zip(datasets, params)):
        data_filename = '../data/intermediate/' + dataset + '.p'
        lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
        inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
        
        df = data_filename = pd.read_pickle(data_filename)
        input_info = io.loadmat(inputInfo_filename)
        with h5py.File(lfads_filename, 'r+') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        peak_df_train = pd.read_pickle('../data/peaks/%s_%s_peaks_train.p'%(dataset,param))
        peak_df_test = pd.read_pickle('../data/peaks/%s_%s_peaks_test.p'%(dataset,param))

        peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()

        X,y = opt.get_inputs_to_model(peak_df, co, trial_len, dt, 
                                    win_start=win_start, 
                                    win_stop=win_stop)
        
        theta = np.arctan2(y[:,1], y[:,0])
        theta_bins = np.arange(-np.pi, np.pi+np.pi/(nbins/2), np.pi/(nbins/2))
        y_cat = np.digitize(theta, theta_bins)
        model = RandomForestClassifier()
        model.fit(X, y_cat)
        idx = np.load('../data/intermediate/train_test_split/%s_trials_train.npy'%dataset)
        df = df.loc[idx]
        win = int(win_len/dt)

        win_entropy = np.zeros((co.shape[0], co.shape[1]-win))
        #win_proba = np.zeros((co.shape[0], co.shape[1]-win, nbins))
        for i in list(set(df.index.get_level_values('trial'))):
            for idx_start in range(co.shape[1]-win):
                t_win = np.arange(idx_start, idx_start+win)*dt
                t_target = df.loc[i].kinematic.query('hit_target').index.values
                if np.any([np.any((0 < t-t_target) & (t-t_target < win_stop)) for t in t_win]):
                    win_entropy[i, idx_start] = np.nan
                else:
                    p = model.predict_proba(co[i,idx_start:idx_start+win,:].reshape(1,-1))
                    p = np.ma.MaskedArray(p, p==0)
                    win_entropy[i,idx_start] = -np.sum(p*np.log(p))

        
                


            
        
