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

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

y_min = 0
y_max = 1

offset = 0.15 #offset between kinematics and neual activity
dt = 0.010
kin_dt = 0.001
win = int(dt/kin_dt)

def get_rs(X, Y, n_splits, trial_len, dt, n_trials, kinematic_vars=['x', 'y', 'x_vel', 'y_vel'], 
           use_reg=False, regularizer='ridge', alpha=1, random_state=None):
    
    T_trial = np.round(trial_len/dt).astype(int)
    groups = np.repeat(np.arange(n_trials), T_trial)

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

def get_kinematics(data, trial_len, dt, kinematic_vars=['x', 'y', 'x_vel', 'y_vel'], used_inds=None):
    if used_inds == None:
        used_inds = np.arange(data.index[-1][0]+1)

    kin_dt = 0.001 #dt of kinematics
    win = int(dt/kin_dt)
    midpoint_idx = win//2 - 1  #midpoint of lfads time step to take for downsampling kinematics
    downsampled_kinematics = data.groupby('trial').apply(lambda _df: _df.loc[_df.index[0][0]].loc[offset:trial_len+offset].kinematic[kinematic_vars].iloc[midpoint_idx::win])
    Y = downsampled_kinematics.loc[used_inds].values

    return Y

def get_lfads_predictor(predictor):
    return predictor.reshape((predictor.shape[0]*predictor.shape[1], -1))

if __name__=='__main__':

    experiment_file = snakemake.input[0]
    lfads_file = snakemake.input[1]
    inputInfo_file = snakemake.input[2]

    # experiment_file = '../data/intermediate/rockstar.p'
    # lfads_file = '../data/model_output/rockstar_xsgZ0x_valid.h5'
    # inputInfo_file = '../data/model_output/rockstar_xsgZ0x_inputInfo.mat'
    random_state = 1243

    input_info = io.loadmat(inputInfo_file)

    used_inds = get_indices(input_info, snakemake.wildcards.trial_type)
    df = pd.read_pickle(experiment_file)
    kinematic_vars = ['x', 'y', 'x_vel', 'y_vel']
    n_splits = cfg['kin_decoding_cv_splits']
    with h5py.File(lfads_file,'r') as h5_file:
        trial_len = utils.get_trial_len(h5_file, input_info)
        dt = utils.get_dt(h5_file, input_info)
        Y = get_kinematics(df, trial_len, dt)
        X_smoothed = get_smoothed_rates(df, trial_len, dt)
        for fig_idx, predictor in enumerate(['output_dist_params', 'factors']):
            X_lfads = get_lfads_predictor(h5_file[predictor][:])
            smoothed_rs,_ = get_rs(X_smoothed, Y, n_splits)
            lfads_rs,_ = get_rs(X_lfads, Y, n_splits)

            ## Plotting
            commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
            with PdfPages(snakemake.output[fig_idx], metadata={'commit':commit}) as pdf:
                for plot_idx, k in enumerate(kinematic_vars):
                    # this just re-arranges data into dataframe for seaborn pointplot 
                    rs = []
                    data_type = []
                    idx = []
                    for i in range(n_splits*2):
                        if i < n_splits:
                            rs.append(lfads_rs[k][i])
                            data_type.append('LFADS')
                            idx.append(i)
                        else:
                            rs.append(smoothed_rs[k][i - n_splits])
                            data_type.append('Smoothed Spikes')
                            idx.append(i - n_splits)

                    r_df = pd.DataFrame(zip(*[rs, data_type, idx]),
                                    columns = ['Performance (r^2)', 'Predictor', 'Train Test Split Index'])
                    fig = plt.figure()
                    sns.pointplot(x='Predictor', y='Performance (r^2)', hue='Train Test Split Index', data=r_df)
                    plt.ylim([0, 1])
                    plt.title(k)
                    pdf.savefig(fig)
                    plt.close()
