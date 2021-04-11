import numpy as np
import h5py
from scipy import io
from glob import glob
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import subprocess as sp
from utils import get_indices
from sklearn.linear_model import Ridge, Lasso

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

y_min = 0
y_max = 1

offset = 0.150 #offset between kinematics and neual activity
dt = 0.010
kin_dt = 0.001
win = int(dt/kin_dt)


# def get_rs(X, Y, n_splits, kinematic_vars=['x', 'y', 'x_vel', 'y_vel']):
#     rs = {k:np.zeros(n_splits) for k in kinematic_vars} 
#     kf = KFold(n_splits=n_splits)
#     for k, (train_idx, test_idx) in enumerate(kf.split(X)):
#         X_train, X_test = X[train_idx], X[test_idx]
#         Y_train, Y_test = Y[train_idx], Y[test_idx]
#         Y_est = X_test.dot(np.linalg.lstsq(X_train, Y_train)[0])
#         for i, variable in enumerate(kinematic_vars):
#             rs[variable][k] = np.corrcoef(Y_est[:,i],Y_test[:,i])[1,0]**2

#     return rs

def get_rs(X, Y, n_splits, kinematic_vars=['x', 'y', 'x_vel', 'y_vel'],alpha=1):
    X = X[:,:-1]
    rs = {k:np.zeros(n_splits) for k in kinematic_vars} 
    kf = KFold(n_splits=n_splits)
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        model = Ridge(alpha=alpha)
        model.fit(X_train, Y_train)
        Y_est = model.predict(X_test)
        for i, variable in enumerate(kinematic_vars):
            rs[variable][k] = np.corrcoef(Y_est[:,i],Y_test[:,i])[1,0]**2

    return rs

# experiment_file = '../data/intermediate/raju.p'
# lfads_file = '../data/model_output/raju_8QTVEk_all.h5'
# inputInfo_file = '../data/model_output/raju_inputInfo.mat'
# input_info = io.loadmat(inputInfo_file)
# used_inds = get_indices(input_info, 'all')

def get_xy(lfads_file):
    data = pd.read_pickle(experiment_file)
    kinematic_vars = ['x', 'y', 'x_vel', 'y_vel']

    with h5py.File(lfads_file,'r') as h5_file:
        X = []
        trial_len = h5_file['factors'].shape[1] * dt #trial length cutoff used 
        midpoint_idx = 4 #midpoint of lfads time step to take for downsampling kinematics
        for i in used_inds:
            smoothed = data.loc[i].neural.rolling(window=300, min_periods=1, win_type='gaussian', center=True).mean(std=50)
            smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
            smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
            X.append(smoothed)

            X_smoothed = np.vstack(X)
            X_smoothed = np.hstack((X_smoothed, np.ones((X_smoothed.shape[0],1))))  

    downsampled_kinematics = data.groupby('trial').apply(lambda _df: _df.loc[_df.index[0][0]].loc[offset:trial_len+offset].kinematic[kinematic_vars].iloc[midpoint_idx::win])
    Y = downsampled_kinematics.loc[used_inds].values
    
    return X_smoothed, Y

def fit_smooth(lfads_file, n_splits = 5):
    data = pd.read_pickle(experiment_file)
    kinematic_vars = ['x', 'y', 'x_vel', 'y_vel']

    with h5py.File(lfads_file,'r') as h5_file:
        X = []
        trial_len = h5_file['factors'].shape[1] * dt #trial length cutoff used 
        midpoint_idx = 4 #midpoint of lfads time step to take for downsampling kinematics
        for i in used_inds:
            smoothed = data.loc[i].neural.rolling(window=300, min_periods=1, win_type='gaussian', center=True).mean(std=50)
            smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
            smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
            X.append(smoothed)

            X_smoothed = np.vstack(X)
            X_smoothed = np.hstack((X_smoothed, np.ones((X_smoothed.shape[0],1))))  

    downsampled_kinematics = data.groupby('trial').apply(lambda _df: _df.loc[_df.index[0][0]].loc[offset:trial_len+offset].kinematic[kinematic_vars].iloc[midpoint_idx::win])
    Y = downsampled_kinematics.loc[used_inds].values

    smoothed_rs = get_rs(X_smoothed, Y, n_splits)

    return smoothed_rs

if __name__=='__main__':

    experiment_file = snakemake.input[0]
    lfads_file = snakemake.input[1]
    inputInfo_file = snakemake.input[2]

    # experiment_file = '../data/intermediate/rockstar.p'
    # lfads_file = '../data/model_output/rockstar_xsgZ0x_valid.h5'
    # inputInfo_file = '../data/model_output/rockstar_xsgZ0x_inputInfo.mat'

    input_info = io.loadmat(inputInfo_file)

    used_inds = get_indices(input_info, snakemake.wildcards.trial_type)
    data = pd.read_pickle(experiment_file)
    kinematic_vars = ['x', 'y', 'x_vel', 'y_vel']

    for fig_idx, predictor in enumerate(['output_dist_params', 'factors']):
        with h5py.File(lfads_file,'r') as h5_file:
            X = []
            trial_len = h5_file[predictor].shape[1] * dt #trial length cutoff used 
            midpoint_idx = 4 #midpoint of lfads time step to take for downsampling kinematics
            for i in used_inds:
                smoothed = data.loc[i].neural.rolling(window=300, min_periods=1, win_type='gaussian', center=True).mean(std=50)
                smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
                smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
                X.append(smoothed)

            X_smoothed = np.vstack(X)
            X_smoothed = np.hstack((X_smoothed, np.ones((X_smoothed.shape[0],1))))  

            factors = h5_file[predictor][:,:,:]
            X = factors.reshape((factors.shape[0]*factors.shape[1], -1))
            X = np.hstack((X, np.ones((X.shape[0],1))))
            Y = np.zeros(h5_file[predictor].shape[:2] + (len(kinematic_vars),))

            X_lfads = np.copy(X)
            downsampled_kinematics = data.groupby('trial').apply(lambda _df: _df.loc[_df.index[0][0]].loc[offset:trial_len+offset].kinematic[kinematic_vars].iloc[midpoint_idx::win])
            Y = downsampled_kinematics.loc[used_inds].values.astype(float)

        ## Fitting
        n_splits = 5
        smoothed_rs = get_rs(X_smoothed, Y, n_splits)
        lfads_rs = get_rs(X_lfads, Y, n_splits)

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
