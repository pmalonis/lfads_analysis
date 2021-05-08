import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.stats import entropy
from scipy.special import softmax
from scipy.signal import savgol_filter
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
from scipy.signal import find_peaks
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
#plt.rcParams['font.family'] = 'FreeSarif'
reload(opt)
reload(utils)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def xy_at_peak(_df, peak_list, dt):
    trial_idx = _df.index[0][0]
    trial_peaks = peak_list[trial_idx]
    t_peaks = trial_peaks * dt
    try:
        trial_targets = _df.loc[trial_idx].kinematic.query('hit_target')
    except:
        import pdb;pdb.set_trace()
    target_idx = [trial_targets.index.get_loc(tp, method='bfill') for tp in t_peaks]
    target_pos = trial_targets.iloc[target_idx][['x','y']]

    peak_idx = [_df.loc[trial_idx].index.get_loc(tp, method='nearest') for tp in t_peaks]
    hand_pos = _df.loc[trial_idx].iloc[peak_idx].kinematic[['x','y']]
    return target_pos.values - hand_pos

#datasets = ['rockstar', 'raju', 'mack']
datasets = ['rockstar','mack']
#params = ['final-fixed-2OLS24', 'final-fixed-2OLS24', 'mack-kl-co-sweep-0Wo8i9']
params = ['fixed-rockstar-2OLS24', 'mack-kl-co-sweep-0Wo8i9']
nbins = 12
win_start = 0.05#cfg['post_target_win_start']
win_stop = 0.2#cfg['post_target_win_stop']
entropy_threshold = 0.02
savgol_params = {'window_length':11, 'polyorder':2}

lfads_filename = '../data/model_output/' + '_'.join([datasets[0], params[0], 'all.h5'])
with h5py.File(lfads_filename, 'r+') as h5file:
    co = h5file['controller_outputs'][:]
    
n_co = co.shape[2]

if __name__=='__main__':
    fig1, axplot = plt.subplots(n_co, len(datasets))

    pred = {}
    dirs = {}
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

        # filtering co
        co = savgol_filter(co, axis=1, **savgol_params)

        win_size = int((win_stop-win_start)/dt)
        bytesize = co.dtype.itemsize
        n_wins = co.shape[1] - win_size + 1 #number of windows per trial
        sliding_window_co = as_strided(co, 
                                       (co.shape[0],
                                        co.shape[1]-win_size+1,
                                        n_co,win_size),
                                       (bytesize*co.shape[1],
                                        bytesize*co.shape[2],
                                        bytesize, 
                                        bytesize*co.shape[2]))
        co = co.copy()
        # getting equal number of targets in each row
        targets = df.kinematic.query('hit_target')
        min_n_targets = targets.groupby('trial').count().min().min()
        targets = targets.groupby('trial').apply(lambda _df: _df.loc[_df.index[0][0]].iloc[:min_n_targets])

        t_targets = targets.index.get_level_values('time').values.reshape((-1, min_n_targets))

        t_win_start = np.arange(n_wins) * dt
        t_from_target = np.subtract.outer(t_targets, t_win_start)
        win_mask = np.any((t_from_target > -win_stop) & (t_from_target < win_size*dt), axis=1) # windows that overlap with post target win

        peak_df_train = pd.read_pickle('../data/peaks/%s_%s_peaks_train.p'%(dataset,param))
        peak_df_test = pd.read_pickle('../data/peaks/%s_%s_peaks_test.p'%(dataset,param))

        peak_df = pd.concat([peak_df_train, peak_df_test]).sort_index()

        X,y = opt.get_inputs_to_model(peak_df, df, co, trial_len, dt, 
                                    win_start=win_start, 
                                    win_stop=win_stop)

        theta = np.arctan2(y[:,1], y[:,0])
        assert(nbins%2==0)
        
        bin_theta = np.pi / (nbins/2)
        colors = sns.color_palette('hls', nbins)
        t_ms = np.arange(win_start, win_stop, dt) * 1000
        co_av = np.zeros((n_co * win_size, nbins))
        bin_centers = np.zeros(nbins)
        for j in range(n_co):
            fig, axplot = plt.subplots()
            axplot.set_title('Controller %s'%(j+1))
            ymin, ymax = (np.min(X[:,j::n_co]), np.max(X[:,j::n_co]))
            
            #setting dimensions for inset
            left = .7#win_start + .8 * (win_stop-win_start)
            bottom = .7#ymin + .8 * (ymax-ymin)
            width = .15 #* (win_stop - win_start)
            height = .15 #* (ymax - ymin)
            inset = fig.add_axes([left, bottom, width, height], polar=True)
            co_min = np.inf
            co_max = -np.inf
            for i in range(-nbins//2, nbins//2):
                min_theta = i * bin_theta
                max_theta = (i+1) * bin_theta
                bin_centers[i+nbins//2] = (max_theta - min_theta)/2
                #normalizing by each co
                bin_av = X[:,j*win_size:(j+1)*win_size][(theta > min_theta) & (theta <= max_theta)].mean(0)
                #bin_av -= np.mean(bin_av)
                # bin_av /= np.linalg.norm(bin_av)
                co_av[j*win_size:(j+1)*win_size, i + nbins//2] = bin_av #X[:,j*win_size:(j+1)*win_size][(theta > min_theta) & (theta <= max_theta)].mean(0)

        # normalizing co and average
        win_means = np.mean(sliding_window_co, axis=3)
        # sliding_window_co -= as_strided(win_means,
        #                                 sliding_window_co.shape,
        #                                 win_means.strides + (0,))
        # win_norms = np.linalg.norm(sliding_window_co, axis=3)
        # sliding_window_co /= as_strided(win_norms,
        #                                 sliding_window_co.shape,
        #                                 win_norms.strides + (0,))
        sliding_window_co = sliding_window_co.reshape(sliding_window_co.shape[:2] + (-1,)) #concatenating different controllers in last dimension
        #co_av -= co_av.mean(0)
        #co_av /= np.linalg.norm(co_av,axis=0)

        #corrs = sliding_window_co.dot(co_av)
        #corrs[corrs<0] = 0
        tiled_av = np.tile(co_av, sliding_window_co.shape[:2] + (1,1))
        tiled_av = np.moveaxis(tiled_av, 3, 0)
        corrs = 1/np.sqrt(np.sum((sliding_window_co - tiled_av) ** 2, axis=3))
        corrs = np.moveaxis(corrs,0,3)
        win_entropy = entropy(softmax(corrs, axis=2),axis=2)
        masked_entropy = np.ma.MaskedArray(win_entropy, mask=win_mask)
        masked_entropy = (masked_entropy.T-masked_entropy.mean(1)).T  
        thetas_at_peak = []
        predict_at_peak = []
        peak_list = []
        for trial_idx in range(masked_entropy.shape[0]):
            p,_ = find_peaks(-masked_entropy[trial_idx, :], distance=win_size, height=entropy_threshold)
            peak_list.append(p)
            predict_at_peak.append(np.argmax(corrs[trial_idx,p,:], axis=1))
            # t_peak = p*dt
            # target_idx = targets.loc[trial_idx].index.get_loc(t_peak, method='bfill')
            # target_x, target_y = targets.loc[trial_idx].iloc[target_idx][['x','y']].values
            # peak_idx = df.loc[trial_idx]
        
        pos = df.groupby('trial').apply(lambda _df:xy_at_peak(_df,peak_list,dt)) 
        dirs[dataset] = np.arctan2(pos.y,pos.x).values
        pred[dataset] = np.concatenate(predict_at_peak)