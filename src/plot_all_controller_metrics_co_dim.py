import numpy as np
import pandas as pd
import h5py
from scipy import io
import utils
import yaml
import os
import peak_trainvalidtest_split as ps
import matplotlib.pyplot as plt
import subsample_analysis as sa
from scipy.signal import savgol_filter, periodogram
from scipy.stats import kurtosis
import pickle
import timing_analysis as ta
import decode_lfads as dl
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from snr import get_event_co, get_background_co
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 18

absolute_min_heights = 0.3
relative_min_heights = 3
win_start = 0.0
win_stop = 0.3
peak_with_threshold = 0.9
figsize = (12,5)
n_splits = 5

run_info_path = os.path.join(os.path.dirname(__file__), '../lfads_file_locations.yml')
run_info = yaml.safe_load(open(run_info_path, 'r'))

class Dataset_Info():
    def __init__(self, dataset):
        self.kl_weight = {'gaussian':[], 'laplace':[]}
        self.measure = {'gaussian':[], 'laplace':[]}
        self.name = run_info[dataset]['name']

    def plot(self, ax):
        try:
            for k in ['gaussian', 'laplace']:
                assert(len(self.kl_weight[k]) == len(self.measure[k]))
        except AssertionError:
            raise ValueError("kl_weights and measure must have the same length")

        for prior in self.kl_weight.keys():
            idx = np.argsort(self.kl_weight[prior])
            self.kl_weight[prior] = np.array(self.kl_weight[prior])[idx]
            self.measure[prior] = np.array(self.measure[prior])[idx]
        
        ax.plot(self.kl_weight['gaussian'], self.measure['gaussian'])
        #ax.plot(self.kl_weight['laplace'], self.measure['laplace'])
        ax.set_xlabel('Input Dimensionality')
        #ax.set_title("Monkey " + self.name)
        #ax.'legend'(['Dense Prior', 'Sparse Prior'])

class Run_Data():
    def __init__(self, dataset, param, df, co, trial_len, dt):
        self.dataset = dataset
        self.param = param
        self.df = df
        self.co = co
        self.trial_len = trial_len
        self.dt = dt

class Measure():
    def __init__(self, title='', filename=None):
        '''measure class inherits from run data'''
        self.datasets = {}
        self.title = title
        self.filename = filename
        self.fig = None

    def add_dataset(self, dataset):
        self.datasets[dataset] = Dataset_Info(dataset)

    def add_run(self, prior, kl_weight, dataset, param, df, co, trial_len, dt):
        if dataset not in self.datasets.keys():
            self.add_dataset(dataset)

        self.datasets[dataset].kl_weight[prior].append(kl_weight)
        run = Run_Data(dataset, param, df, co, trial_len, dt)
        m = self.compute_measure(run)
        self.datasets[dataset].measure[prior].append(m)

    def compute_measure(self, run):
        pass

    def plot(self):
        # fig = plt.figure(figsize=figsize)
        # axes = fig.subplots(1, len(self.datasets.keys()))
        fig, ax = plt.subplots(figsize=figsize)
        for i,k in enumerate(self.datasets.keys()):
            # self.datasets[k].plot(axes[i])
            # axes[i].set_ylabel(self.ylabel)
            self.datasets[k].plot(ax)
            ax.set_ylabel(self.ylabel)
        
        fig.suptitle(self.title)
        ax.legend(['Monkey %s'%d.name for d in self.datasets.values()])
        plt.locator_params(num_ticks=4)
        self.fig = fig

    def savefig(self):
        if self.filename is not None and self.fig is not None:
            self.fig.savefig(self.filename)
        else:
            print("fig is None, nothing was saved")

class Absolute_Target_Peak(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='Proportion of targets with peaks'

    def compute_measure(self, run):
        peak_path = '../data/peaks/%s_%s_peaks_all_thresh=%0.1f.p'%(run.dataset, run.param, absolute_min_heights)
    
        if os.path.exists(peak_path):
            peak_df = pd.read_pickle(peak_path)
        else:
            peak_df = ta.get_peak_df(df, co, trial_len, absolute_min_heights, dt, relative=False, win_start=win_start, win_stop=win_stop)
            peak_df.to_pickle(peak_path)

        npeaks = (peak_df['latency_0'].notnull() | peak_df['latency_1'].notnull()).sum()

        return npeaks/peak_df.shape[0]

class Relative_Target_Peak(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='Proportion of targets with peaks'

    def compute_measure(self, run):    
        peak_path = '../data/peaks/%s_%s_peaks_relative_%dsds.p'%(run.dataset, run.param, relative_min_heights)
        if os.path.exists(peak_path):
            peak_df = pd.read_pickle(peak_path)
        else:            
            peak_df = ta.get_peak_df(df, co, trial_len, relative_min_heights, dt, relative=True, win_start=win_start, win_stop=win_stop)
            peak_df.to_pickle(peak_path)

        npeaks = (peak_df['latency_0'].notnull() | peak_df['latency_1'].notnull()).sum()

        return npeaks/peak_df.shape[0]

class Firstmove_AUC(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='ROC AUC'

    def compute_measure(self, run):    
        firstmove_path = '../data/peaks/%s_targets_all.p'%run.dataset
        corrections_path = '../data/peaks/%s_corrections_all.p'%run.dataset
        firstmove_df = pd.read_pickle(firstmove_path)
        corr_df = pd.read_pickle(corrections_path)
        background = firstmove_df.groupby('trial').apply(lambda _df: get_background_co(_df, corr_df, run.co, run.dt, run.trial_len))
        firstmove = firstmove_df.groupby('trial').apply(lambda _df: get_event_co(_df, run.co, run.dt, run.trial_len))
        noise = np.concatenate(background.values)
        signal = np.concatenate(firstmove.values)
        y_score = np.concatenate([signal, noise])
        y_true = np.zeros(len(y_score))
        y_true[:len(signal)] = 1
        score = roc_auc_score(y_true, y_score)

        return score

class Firstmove_Precision(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='Average Precision'

    def compute_measure(self, run):    
        firstmove_path = '../data/peaks/%s_firstmove_all.p'%run.dataset
        corrections_path = '../data/peaks/%s_corrections_all.p'%run.dataset
        firstmove_df = pd.read_pickle(firstmove_path)
        corr_df = pd.read_pickle(corrections_path)
        background = firstmove_df.groupby('trial').apply(lambda _df: get_background_co(_df, corr_df, run.co, run.dt, run.trial_len))
        firstmove = firstmove_df.groupby('trial').apply(lambda _df: get_event_co(_df, run.co, run.dt, run.trial_len))
        noise = np.concatenate(background.values)
        signal = np.concatenate(firstmove.values)
        y_score = np.concatenate([signal, noise])
        y_true = np.zeros(len(y_score))
        y_true[:len(signal)] = 1
        weights = np.ones(len(y_score))
        weights[len(signal):] = len(signal)/len(noise)
        #precision,_,_ = precision_recall_curve(y_true, y_score, sample_weight=weights)
        #score = np.max(precision[precision<1])
        score = average_precision_score(y_true, y_score, sample_weight=weights)

        return score


class Decode(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kinematics = {}
        self.firing_rates = {}
        self.ylabel='Decoding Accuracy ($\mathregular{r^2}$)'

    def add_dataset(self, dataset, firing_rates):
        self.datasets[dataset] = Dataset_Info(dataset) 
        self.firing_rates[dataset] = firing_rates

    def add_run(self, prior, kl_weight, dataset, param, df, co, trial_len, dt):
        if dataset not in self.datasets.keys():
            #X_smoothed = dl.get_smoothed_rates(df, trial_len, dt)
            X_smoothed = None
            self.add_dataset(dataset, X_smoothed)

        self.datasets[dataset].kl_weight[prior].append(kl_weight)
        run = Run_Data(dataset, param, df, co, trial_len, dt)
        m = self.compute_measure(run)
        self.datasets[dataset].measure[prior].append(m)

    def compute_measure(self, run):
        lfads_filename = os.path.dirname(__file__) + '../data/model_output/' + '_'.join([run.dataset, run.param, 'all.h5'])
        with h5py.File(lfads_filename, 'r') as h5file:
            X = dl.get_lfads_predictor(h5file['factors'][:])

        Y = dl.get_kinematics(df, trial_len, dt)
        n_trials = run.df.index[-1][0] + 1
        return self.get_decoding_performance(X, Y, run.trial_len, 
                                            run.dt, n_trials)

    def get_decoding_performance(self, X, Y, trial_len, dt, n_trials, n_splits=n_splits):
        rs,_ = dl.get_rs(X, Y, n_splits, trial_len, dt, n_trials)
        #rs = {k:v for k,v in rs.items() if 'vel' in k}
        decode_perf = [np.mean([rs[k][i] for k in rs.keys()]) 
                        for i in range(n_splits)] #mean over kinematic variables for each split
        decode_mean = np.mean(decode_perf)
        decode_std = np.std(decode_perf)
        return decode_mean, decode_std

    def plot(self):
        fig, ax = plt.subplots(figsize=figsize)
        for i,k in enumerate(self.datasets.keys()):
            decode_mean, decode_std = zip(*self.datasets[k].measure['gaussian'])
            idx = np.argsort(self.datasets[k].kl_weight['gaussian'])
            decode_mean = np.array(decode_mean)[idx]   
            decode_std = np.array(decode_std)[idx]  
            kl_weight = np.array(self.datasets[k].kl_weight['gaussian'])[idx]
            plt.plot(kl_weight, decode_mean)
            plt.fill_between(kl_weight, 
                            decode_mean - decode_std,
                            decode_mean + decode_std, alpha=0.2)
            ax.legend(['Monkey %s'%d.name for d in self.datasets.values()])
            plt.locator_params(num_ticks=4)
            ax.set_ylabel(self.ylabel)
            ax.set_xlabel('Controller Dimensionality')
        
        fig.suptitle(self.title)
        ax.legend(['Monkey %s'%d.name for d in self.datasets.values()])
        plt.locator_params(num_ticks=4)
        self.fig = fig

    # def plot(self, fig=None):
    #     if fig is None:
    #         fig = plt.figure(figsize=figsize)
    #     axes = fig.subplots(1, len(self.datasets.keys()))
    #     if len(self.datasets.keys()) == 1:
    #         axes = [axes]
            
    #     for i,k in enumerate(self.datasets.keys()):
    #         self.datasets[k].plot(axes[i])
    #         axes[i].set_ylabel(self.ylabel)

    #         #adding firing rate decoding performancce
    #         Y = self.kinematics[k] 
    #         X_smoothed = self.firing_rates[k]
    #         rate_decode = self.get_decoding_performance(X_smoothed, Y)
    #         n_points = len(self.datasets[k].kl_weight['gaussian'])
    #         axes[i].plot(self.datasets[k].kl_weight['gaussian'], 
    #                         np.ones(n_points) * rate_decode)
            
            #adding autolfads performance for rockstar
            # if k=='rockstar':
            #     with h5py.File('../data/model_output/rockstar_autolfads-laplace-prior_all.h5', 'r') as h5file:
            #         X_autolfads,_ = dl.get_lfads_predictor(h5file['factors'][:])
                
            #     autolfads_decode = self.get_decoding_performance(X_autolfads, Y)
            #     n_points = len(self.datasets[k].kl_weight['gaussian'])
            #     axes[i].plot(self.datasets[k].kl_weight['gaussian'], 
            #                 np.ones(n_points) * autolfads_decode, 'r')
            #     axes[i].legend(['Dense Prior', 'Sparse Prior', 'Firing Rate Decode', 
            #                     'Autolfads Decode'])
            # else:
            #     axes[i].legend(['Dense Prior', 'Sparse Prior', 'Firing Rate Decode'])

        # lns = axes[0].get_lines()
        # lns.pop(1)
        # plt.legend(handles=lns, labels=['Decoding from LFADS Factors', 'Decoding from Firing Rates'],
        #             bbox_to_anchor=(0.5, 0.8, 0.5, 0.5), loc='lower center', ncol=1)
        # fig.suptitle(self.title)
        # self.fig = fig
    
# class Total_Peaks(Measure):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.ylabel='Reconstruction Loss, Validation Set'
    
#     def compute_measure(self, run):

#         all_peaks = ta.get_peaks(co, dt, min_heights)
#         total_peaks = 0
#         for i in range(all_peaks.shape[0]):
#             for j in range(all_peaks[i,0].shape[0]):
#                 if np.any(np.abs(all_peaks[i,0][j] - all_peaks[i,1]) < 0.1):
#                     continue
#                 else:
#                     total_peaks += 1

#             total_peaks += len(all_peaks[i,1])

#         n_targets = peak_df.shape[0]
#         #peak_path = '../data/peaks/%s_%s_peaks_all_thresh=%0.1f.p'%(run.dataset, run.param, min_heights)
#         peak_path = '../data/peaks/%s_%s_peaks_relative_3sds.p'%(run.dataset, run.param)
#         if os.path.exists(peak_path):
#             peak_df = pd.read_pickle(peak_path)
#         else:
#             #peak_df,_ = ps.split_peak_df(df, co, trial_len, dt, dataset, param)
#             peak_df = ta.get_peak_df(df, co, trial_len, min_heights, dt, win_start=win_start, win_stop=win_stop)
#             peak_df.to_pickle(peak_path)

#         npeaks = (peak_df['latency_0'].notnull() | peak_df['latency_1'].notnull()).sum()
#         peak_counts[prior].append(total_peaks/n_targets)

#         return run_info[run.dataset]['params'][run.param]['fit']['recon_train']

class Gini(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='Controller Gini Coefficient'

    def compute_measure(self, run):
        co_ginis = np.zeros(run.co.shape[2])
        co_power_weight = (run.co**2).sum((0,1))
        co_power_weight /= np.sum(co_power_weight)
        for co_idx in range(co.shape[2]):
            co_ginis[co_idx] = sa.gini(run.co[:,:,co_idx].flatten())
            
        return np.sum(co_ginis * co_power_weight)

class Spectral_Centroid(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='Spectral Mean'

    def compute_measure(self, run):
        # f,p = periodogram(run.co.sum(2), fs=100, axis=1)
        # p = p.mean(0)
        # p /= p.sum(0)
        
        # return np.sum(f*p)

        f,p = periodogram(run.co, axis=1)
        p = p.mean(0)
        p /= p.sum(0)
        c = np.zeros(p.shape[1])
        for i in range(p.shape[1]):
            c[i] = np.sum(f*p[:,i])

        return -np.mean(c)

class Power(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='Controller Gini Coefficient'

    def compute_measure(self, run):
        return np.mean([sa.gini(run.co[:,:,i].flatten()) for i in range(run.co.shape[2])])

class Kurtosis(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='Controller Kurtosis'

    def compute_measure(self, run):
        #return np.mean([kurtosis(run.co[:,:,i]) for i in range(run.co.shape[2])])
        return kurtosis(np.abs(savgol_filter(run.co,11,2,axis=1)).sum(2).flatten())

class Fit(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ylabel='Reconstruction Loss, Validation Set'
    
    def compute_measure(self, run):
        return run_info[run.dataset]['params'][run.param]['fit']['recon_train']

if __name__=='__main__':

    metric_dict = {#'gini': Gini,
               #'fit': Fit,
               #'spectral': Spectral_Centroid,
               #'kurtosis': Kurtosis,
               #'power': Power,
               'decode': Decode,
               #'absolute_target_peak': Absolute_Target_Peak,
               #'relative_target_peak': Relative_Target_Peak,
               #'firstmove_auc': Firstmove_AUC,
               #'firstmove_precision': Firstmove_Precision
                }

    for kl_weight in [2.0]:
        # measures = [m(filename='../figures/final_figures/%s_co_dim.svg'%k) for k,m in 
        #             metric_dict.items()]    
        measures = [m(filename='../figures/final_figures/numbered/2e.pdf') for k,m in 
                    metric_dict.items()]    
        for dataset in run_info.keys():
            df = pd.read_pickle('../data/intermediate/%s.p'%dataset)
            for param in run_info[dataset]['params'].keys():
                lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
                if 'raju' in dataset and not os.path.exists(lfads_filename):
                    continue
                
                if run_info[dataset]['params'][param]['param_values'].get('kl_co_weight') != kl_weight:
                    continue

                inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
                input_info = io.loadmat(inputInfo_filename)
                with h5py.File(lfads_filename, 'r') as h5file:
                    dt = utils.get_dt(h5file, input_info)
                    trial_len = utils.get_trial_len(h5file, input_info)
                    if 'controller_outputs' in h5file.keys():
                        co = h5file['controller_outputs'][:]
                    else:
                        co = np.zeros((df.index[-1][0], int(trial_len/dt), 2))

                prior = run_info[dataset]['params'][param]['param_values'].get('ar_prior_dist')
                if prior is None:
                    prior = 'gaussian'

                co_dim = run_info[dataset]['params'][param]['param_values']['co_dim']
                if co_dim > 4:
                    continue 
                for measure in measures:
                    measure.add_run(prior, co_dim, dataset, param, df, co, trial_len, dt)
                    
        for measure in measures:
            measure.plot()
            plt.xticks([0,1,2,3,4])
            plt.tight_layout()
            measure.savefig()