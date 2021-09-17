import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from utils import inputs_to_model
from optimize_target_prediction import get_inputs_to_model
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
import optimize_target_prediction as opt
import feedback_optimize_target_prediction as fotp
import utils
from importlib import reload
import os
import yaml
from scipy import io
import h5py
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from matplotlib import rcParams
from scipy.stats import spearmanr

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r')) 

nbins = 12
fb_win_start = -0.3#0.00#-0.1#cfg['post_target_win_start']
fb_win_stop = 0.0#0.3#0.1#cfg['post_target_win_stop']
win_start = -0.30
win_stop = 0.0

def print_correlation(dataset):
    param = open('../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric'])).read()

    data_filename = '../data/intermediate/' + dataset + '.p'
    lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
    inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
    
    df = data_filename = pd.read_pickle(data_filename)
    input_info = io.loadmat(inputInfo_filename)
    with h5py.File(lfads_filename, 'r+') as h5file:
        co = h5file['controller_outputs'][:]
        dt = utils.get_dt(h5file, input_info)
        trial_len = utils.get_trial_len(h5file, input_info)

    peak_df_train = pd.read_pickle('../data/peaks/%s_new-firstmove_train.p'%(dataset))
    peak_df_test = pd.read_pickle('../data/peaks/%s_new-firstmove_test.p'%(dataset))

    peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()

    fb_peak_df_train = pd.read_pickle('../data/peaks/%s_new-corrections_train.p'%(dataset))
    fb_peak_df_test = pd.read_pickle('../data/peaks/%s_new-corrections_test.p'%(dataset))

    fb_peak_df = pd.concat([fb_peak_df_train, fb_peak_df_test]).sort_index()
    n_co = co.shape[2]
    X,y = opt.get_inputs_to_model(peak_df, co, trial_len, dt, df=df, 
                                win_start=win_start, 
                                win_stop=win_stop)

    fb_X,fb_y = opt.get_inputs_to_model(fb_peak_df, co, trial_len, dt, df=df,
                                        win_start=fb_win_start, win_stop=fb_win_stop)

    win_size = int((win_stop - win_start)/dt)
    fb_win_size = int((fb_win_stop - fb_win_start)/dt)

    theta = np.arctan2(y[:,1], y[:,0])
    fb_theta = np.arctan2(fb_y[:,1], fb_y[:,0])
    assert(nbins%2==0)
    
    bin_theta = np.pi / (nbins/2)
    mean_zscores = np.zeros(n_co*nbins)
    max_zscores = np.zeros(n_co*nbins)
    fb_mean_zscores = np.zeros(n_co*nbins)
    fb_max_zscores = np.zeros(n_co*nbins)
    for j in range(n_co):
        ymin, ymax = (np.min(X[:,j*win_size:(j+1)*win_size]), np.max(X[:,j*win_size:(j+1)*win_size]))
        fb_co_mean = []
        co_mean = []
        fb_co_max = []
        co_max = []
        fb_co_rank = [] 
        co_rank = [] 
        for i in range(-nbins//2, nbins//2):
            min_theta = i * bin_theta
            max_theta = (i+1) * bin_theta
            co_av = X[:,j*win_size:(j+1)*win_size][(theta > min_theta) & (theta <= max_theta)].mean(0)
            fb_co_av = fb_X[:,j*fb_win_size:(j+1)*fb_win_size][(fb_theta > min_theta) & (fb_theta <= max_theta)].mean(0) 
            co_mean.append(co_av.mean())
            fb_co_mean.append(fb_co_av.mean())
            co_max.append(co_av.max())
            fb_co_max.append(fb_co_av.max())

        co_rank += list(np.argsort(co_max))
        fb_co_rank += list(np.argsort(fb_co_max))

        mean_zscores[j*nbins:(j+1)*nbins] = zscore(co_mean)
        max_zscores[j*nbins:(j+1)*nbins] = zscore(co_max)
        fb_mean_zscores[j*nbins:(j+1)*nbins] = zscore(fb_co_mean)
        fb_max_zscores[j*nbins:(j+1)*nbins] = zscore(fb_co_max)

    r_value, p_value = pearsonr(mean_zscores, fb_mean_zscores)
    
    if 0.01 <= p_value < 0.05:
        p_str = 'p = %0.2f'%p_value
    elif 0.001 <= p_value < 0.01:
        p_str = 'p < 0.01'
    elif p_value < 0.001:
        p_str = 'p < 0.001'
    else:
        p_str = 'p > 0.05'

    dset_str = 'raju' if 'raju' in dataset else dataset
    print('%s correlation, r = %0.2f, %s'%(dset_str, r_value, p_str))

for dataset in ['rockstar','mack','raju-M1-no-bad-trials']:
    X,y=get_inputs_to_model(*inputs_to_model(dataset,'corrections'),win_start=-0.25,win_stop=0,hand_time=0.0)
    X_new,y_new=get_inputs_to_model(*inputs_to_model(dataset,'new-corrections'),win_start=-0.25,win_stop=0,hand_time=0.0)

    model = MultiOutputRegressor(SVR())

    score = np.mean(cross_val_score(model, X, y[:,:2]))
    new_score = np.mean(cross_val_score(model, X_new, y_new[:,:2]))

    print_correlation(dataset)
    print('%s old corrections decoding performance: %f'%(dataset,score))
    print('%s new corrections decoding performance: %f'%(dataset,new_score))
    print('\n')