import numpy as np
import h5py
from scipy import io
from glob import glob
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GroupKFold
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import subprocess as sp
from utils import get_indices
import utils
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score
from chunking_split_data_for_autolfads import get_n_chunks

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

y_min = 0
y_max = 1

offset = 0.15 #offset between kinematics and neual activity
dt = 0.010
kin_dt = 0.001
win = int(dt/kin_dt)

def get_rs(X, Y, groups, n_splits, dt, kinematic_vars=['x', 'y', 'x_vel', 'y_vel'], 
           use_reg=False, regularizer='ridge', alpha=1, random_state=None):

    rs = {k:np.zeros(n_splits) for k in kinematic_vars} 
    kf = GroupKFold(n_splits=n_splits)
    variance_weighted_rs = np.zeros(n_splits)
    if use_reg == True:
        if regularizer == 'ridge':
                model = Ridge(alpha=alpha, normalize=True)
        elif regularizer == 'lasso':
                model = Lasso(alpha=alpha, normalize=True)
        else:
                raise ValueError("Regularizer must be ridge or lasso")
    else:
        model = LinearRegression()

    for k, (train_idx, test_idx) in enumerate(kf.split(X, Y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        for i, variable in enumerate(kinematic_vars):
            rs[variable][k] = r2_score(Y_test[:,i], Y_pred[:,i])

        variance_weighted_rs[k] = r2_score(Y_test, Y_pred, multioutput='variance_weighted')

    return rs, variance_weighted_rs

def get_smoothed_rates(data, trial_len, dt, kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], used_inds=None):
    if used_inds == None:
        used_inds = np.arange(data.index[-1][0]+1)

    n_trials = data.index[-1][0] + 1
    n_samples = int(trial_len/dt) * n_trials
    n_neurons = data.loc[0].neural.shape[1]
    X = np.zeros((n_samples, n_neurons))
    sigma = cfg['rate_sigma_kin_decoding']
    kin_dt = 0.001 #dt of kinematics
    win = int(dt/kin_dt)
    midpoint_idx = win//2 - 1 #midpoint of lfads time step to take for downsampling kinematics
    n_trials = data.index[-1][0] + 1
    n_neurons = data.neural.shape[1]
    k = 0
    for i in used_inds:
        smoothed = data.loc[i].neural.rolling(window=sigma*6, min_periods=1, win_type='gaussian', center=True).mean(std=sigma)
        smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
        smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
        X[k:k+smoothed.shape[0], :] = smoothed
        k += smoothed.shape[0]

    return X

def get_kinematics(data, dt, kinematic_vars=['x', 'y', 'x_vel', 'y_vel'], used_inds=None):
    if used_inds == None:
        used_inds = np.arange(data.index[-1][0]+1)

    kin_dt = 0.001 #dt of kinematics
    win = int(dt/kin_dt)
    midpoint_idx = win//2 - 1  #midpoint of lfads time step to take for downsampling kinematics
    def downsample_offset(_df):
        trial_idx = _df.index[0][0]
        trial_df = _df.loc[trial_idx]

        trial_len = trial_df.index[-1] - cfg['preprocess']['post_trial_pad']
        n_chunks = get_n_chunks(trial_len)
        chunked_trial_len = n_chunks * cfg['data_chunking']['chunk_length'] - (n_chunks-1)*cfg['data_chunking']['overlap']
        offset_trial = trial_df.loc[offset:chunked_trial_len+offset]
        downsampled = offset_trial.kinematic[kinematic_vars].iloc[midpoint_idx::win]
        return downsampled

    downsampled_kinematics = data.groupby('trial').apply(downsample_offset)

    Y = downsampled_kinematics.loc[used_inds].values

    return Y


def get_lfads_predictor(predictor):
    n_trials = len(predictor.keys())
    trial_data = [predictor['trial_%03d'%i][:] for i in range(n_trials)]
    groups = [[i]*d.shape[0] for i,d in enumerate(trial_data)]
    trial_data = np.concatenate(trial_data)
    groups = np.concatenate(groups)
    
    return trial_data, groups

if __name__=='__main__':

    # experiment_file = snakemake.input[0]
    # lfads_file = snakemake.input[1]
    # inputInfo_file = snakemake.input[2]
    # output_files = snakemake.output
    # trial_type = snakemake.wildcards.trial_type

    experiment_file = '../data/intermediate/rockstar_full.p'
    #lfads_file = '../data/model_output/rockstar_autolfads-1000msChunk200msOverlap-keep-ratio-low-range_all.h5'#'../data/model_output/rockstar_autolfads-split-trunc-01_all.h5'
    #inputInfo_file = '../data/model_output/rockstar_inputInfo.mat'
    #lfads_file = '../data/model_output/rockstar_split-rockstar-1000ms200ms-overlap-FDCWrX_all.h5'
    
    #inputInfo_file = '../data/model_output/splitlfads_rockstar_inputInfo.mat'
    #lfads_file = '../data/model_output/rockstar_kl-co-dim-search-FDCWrX_all.h5'
    #inputInfo_file = '../data/model_output/bu_rockstar_inputInfo.mat'
    
    #lfads_file = '../data/model_output/rockstar_lfads-full-data_all.h5'

    lfads_file = '../data/model_output/rockstar_autolfads-full-data-3-epochs_all.h5'
    #lfads_file = '/home/macleanlab/peter/lfads_analysis/data/model_output/rockstar_autolfads-trunc-1000msChunk200msOverlap_all.h5'
    inputInfo_file = '../data/model_output/long_rockstar_inputInfo.mat'
    #lfads_file = '../data/model_output/rockstar_autolfads-split-trunc-02_all.h5'#'../data/model_output/rockstar_autolfads-split-trunc-01_all.h5'
    #inputInfo_file = '../data/model_output/autolfads_rockstar_inputInfo.mat'
    
    output_files = ['../figures/rockstar_autolfads-full_rate-decode.pdf',
                    '../figures/rockstar_autolfads-full_factor-decode.pdf']
    trial_type = 'all'

    # experiment_file = '../data/intermediate/rockstar.p'
    # lfads_file = '../data/model_output/rockstar_autolfads-split-trunc-02_all.h5'#'../data/model_output/rockstar_autolfads-split-trunc-01_all.h5'
    # inputInfo_file = '../data/model_output/rockstar_inputInfo.mat'
    # output_files = ['../figures/rockstar_autolfads-split-trunc-02_rate-decode.pdf',
    #                 '../figures/rockstar_autolfads-split-trunc-02_factor-decode.pdf']
    # trial_type = 'all'

    random_state = 1243

    input_info = io.loadmat(inputInfo_file)
    df = pd.read_pickle(experiment_file)

    kinematic_vars = ['x_vel', 'y_vel']
    n_splits = cfg['kin_decoding_cv_splits']
    with h5py.File(lfads_file,'r') as h5_file:
        dt = utils.get_dt(h5_file, input_info)
        Y = get_kinematics(df, dt)
        #X_smoothed = get_smoothed_rates(df, trial_len, dt)
        for fig_idx, predictor in enumerate(['output_dist_params']): #enumerate(['output_dist_params', 'factors']):
            X_lfads, groups = get_lfads_predictor(h5_file[predictor])
            #smoothed_rs,_ = get_rs(X_smoothed, Y, n_splits, trial_len, dt, n_trials)
            lfads_rs,_ = get_rs(X_lfads, Y, groups, n_splits, dt, kinematic_vars=kinematic_vars)

            # ## Plotting
            # commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
            # with PdfPages(output_files[fig_idx], metadata={'commit':commit}) as pdf:
            #     for plot_idx, k in enumerate(kinematic_vars):
            #         # this just re-arranges data into dataframe for seaborn pointplot 
            #         rs = []
            #         data_type = []
            #         idx = []
            #         for i in range(n_splits*2):
            #             if i < n_splits:
            #                 rs.append(lfads_rs[k][i])
            #                 data_type.append('LFADS')
            #                 idx.append(i)
            #             else:
            #                 rs.append(smoothed_rs[k][i - n_splits])
            #                 data_type.append('Smoothed Spikes')
            #                 idx.append(i - n_splits)

            #         r_df = pd.DataFrame(zip(*[rs, data_type, idx]),
            #                         columns = ['Performance (r^2)', 'Predictor', 'Train Test Split Index'])
            #         fig = plt.figure()
            #         sns.pointplot(x='Predictor', y='Performance (r^2)', hue='Train Test Split Index', data=r_df)
            #         plt.ylim([0, 1])
            #         plt.title(k)
            #         pdf.savefig(fig)
            #         plt.close()
