from ast import literal_eval
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
import pandas as pd
from scipy import io
import utils
import matplotlib.pyplot as plt
import os
import yaml

def get_background_co(_df, corr_df, co, dt, trial_len, win_start=-0.2, win_stop=0.0):
    #_df is trial dataframe of firstmove_df
    i = _df.index[0][0]
    t_firstmove = _df.loc[i].loc[:trial_len-win_stop-dt].index.values
    if i in corr_df.index.get_level_values('trial'):
        t_corr = corr_df.loc[i].loc[:trial_len-win_stop-dt].index.values
    else:
        t_corr = []

    t_all = np.sort(np.concatenate([t_firstmove, t_corr]))
    t_start = t_all + win_start
    t_start = np.append(t_start, trial_len)
    t_start = np.round(t_start/dt).astype(int)
    t_stop = t_all + win_stop
    t_stop = np.insert(t_stop, 0, 0)
    t_stop = np.round(t_stop/dt).astype(int)
    nwin = np.round((win_stop-win_start)/dt).astype(int)

    #average controller magnitude in sliding windows
    try:
        background_co = [np.abs(co[i,i:i+nwin,:]).sum() 
                                for a,b in zip(t_stop, t_start)
                                for i in range(a,b-nwin)] #t_stop is first to exclude peri-event windows
    except:
        import pdb;pdb.set_trace()

    return background_co

def get_event_co(_df, co, dt, trial_len, win_start=-0.2, win_stop=0.0):
    #_df is trial dataframe of firstmove_df
    i = _df.index[0][0]
    # if i in _df.loc[i].index.get_level_values('trial'):
    #     t_event = _df.loc[i].loc[:trial_len].index.values
    # else:
    #     return []
    t_event = _df.loc[i].loc[:trial_len].index.values
    if len(t_event)==0:
        return []

    t_start = t_event + win_start
    t_start = np.round(t_start/dt).astype(int)
    t_stop = t_event + win_stop
    t_stop = np.round(t_stop/dt).astype(int)
    event_co = [np.abs(co[i,max(a,0):b,:]).sum() for a,b in zip(t_start, t_stop)]
    
    return event_co

def rocauc(background, event):
    noise = background
    signal = event
    y_score = np.concatenate([signal, noise])
    y_true = np.zeros(len(y_score))
    y_true[:len(signal)] = 1
    weights = np.ones(len(y_score))
    weights[len(signal):] = len(signal)/len(noise)
    try:
        score = roc_auc_score(y_true, y_score, sample_weight=weights)
    except:
        import pdb;pdb.set_trace()

    return score

def all_event_data(data, firstmove_df, corr_df, maxima_df, co, dt, trial_len, win_start, win_stop):
    background = np.concatenate(firstmove_df.groupby('trial').apply(lambda _df: get_background_co(_df, corr_df, co, dt, trial_len, win_start, win_stop)).values)
    firstmove = np.concatenate(firstmove_df.groupby('trial').apply(lambda _df: get_event_co(_df, co, dt, trial_len, win_start, win_stop)).values)
    corrections = np.concatenate(corr_df.groupby('trial').apply(lambda _df: get_event_co(_df, co, dt, trial_len, win_start, win_stop)).values)
    maxima = np.concatenate(maxima_df.groupby('trial').apply(lambda _df: get_event_co(_df, co, dt, trial_len, win_start, win_stop)).values)

    return background, firstmove, corrections, maxima

def all_rocauc(background, firstmove, corrections, maxima):
    firstmove_score = rocauc(background, firstmove)
    corrections_score = rocauc(background, corrections)
    maxima_score = rocauc(background, maxima)

    return firstmove_score, corrections_score, maxima_score

def co_power_ecdf(co, win):
    co_power = [np.abs(co[trial, i:i+win, :]).sum() 
        for i in range(0, co.shape[1]-win) 
        for trial in range(co.shape[0])]
    
    return ECDF(co_power)


def combined_dataset_score(backgrounds, events, controllers, win):
    p_background = [] #list of all percentile scores for each dataset background
    p_event = [] #list of all percentile scores for each dataset background
    for background, event, co in zip(backgrounds, events, controllers):
        ecdf = co_power_ecdf(co, win)
        try:
            p_background.append(ecdf(background))
        except:
            import pdb;pdb.set_trace()
        p_event.append(ecdf(event))

    return rocauc(np.concatenate(p_background), np.concatenate(p_event))

if __name__=='__main__':
    config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))

    run_info = yaml.safe_load(open(os.path.dirname(__file__) + '/../lfads_file_locations.yml', 'r'))
    win_lims = [literal_eval(w) for w in cfg['target_preprocessing_search']['win_lim']]
    win_centers = [start + (stop-start)/2 for start,stop in win_lims]
    all_dataset_scores = {}
    all_background = {}
    all_firstmove = {}
    all_corrections = {}
    all_maxima = {}
    all_controllers = []
    for dataset in run_info.keys():
        param = open(os.path.dirname(__file__) + '/../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read().strip()
    
        data = pd.read_pickle('../data/intermediate/%s.p'%dataset)                  
        firstmove_df = pd.read_pickle('../data/peaks/%s_firstmove_all.p'%dataset)
        corr_df = pd.read_pickle('../data/peaks/%s_corrections_all.p'%dataset)
        maxima_df = pd.read_pickle('../data/peaks/%s_maxima_all.p'%dataset)
        input_info = io.loadmat('../data/model_output/%s_inputInfo.mat'%dataset)
        
        with h5py.File('../data/model_output/%s_%s_all.h5'%(dataset,param),'r') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        nneurons = sum('neural' in c for c in data.columns)
        std = cfg['target_decoding_smoothed_control_std']
        dt_idx = int(dt/0.001)
        midpoint_idx = int((dt_idx-1)/2)
        all_smoothed = np.zeros((co.shape[0], int(trial_len/dt), nneurons)) #holds firing rates for whole experiment (to be used for dimensionality reduction)
        for i in range(co.shape[0]):
            smoothed = data.loc[i].neural.rolling(window=std*4, min_periods=1, win_type='gaussian', center=True).mean(std=std)
            smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::dt_idx]
            smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
            all_smoothed[i,:,:] = smoothed

        co = all_smoothed
        # background = np.concatenate(firstmove_df.groupby('trial').apply(lambda _df: get_background_co(_df, corr_df, co, dt, trial_len)).values)
        # firstmove = np.concatenate(firstmove_df.groupby('trial').apply(lambda _df: get_event_co(_df, co, dt, trial_len)).values)
        # corrections = np.concatenate(corr_df.groupby('trial').apply(lambda _df: get_event_co(_df, co, dt, trial_len)).values)
        # maxima = np.concatenate(maxima_df.groupby('trial').apply(lambda _df: get_event_co(_df, co, dt, trial_len)).values)
        all_background[run_info[dataset]['name']] = []
        all_firstmove[run_info[dataset]['name']] = []
        all_corrections[run_info[dataset]['name']] = []
        all_maxima[run_info[dataset]['name']] = []
        dataset_scores = []
        for win_start, win_stop in win_lims:
            background, firstmove, corrections, maxima = all_event_data(data, firstmove_df, corr_df, maxima_df, co, dt, trial_len, win_start, win_stop)
            win_score = all_rocauc(background, firstmove, corrections, maxima)
            dataset_scores.append(win_score)
            all_background[run_info[dataset]['name']].append(background)
            all_firstmove[run_info[dataset]['name']].append(firstmove)
            all_corrections[run_info[dataset]['name']].append(corrections)
            all_maxima[run_info[dataset]['name']].append(maxima)

        all_dataset_scores[run_info[dataset]['name']] = np.array(dataset_scores)
        all_controllers.append(co)

    plot_idx = 0
    titles = ['First\n Movement', 'Correction']
    fig = plt.figure(figsize=(12,9))
    for event_idx, event in enumerate(titles):
        dset_names = [d['name'] for d in run_info.values()]
        for i, dset_name in enumerate(dset_names):
            plt.subplot(2, 3, plot_idx+1)
            plt.plot(win_centers, all_dataset_scores[dset_name][:,event_idx])
            #plt.ylabel("ROC AUC")
            plt.plot(win_centers, np.ones_like(win_centers)*0.5, 'k')
            plt.ylim([0.3, 0.8])
            if plot_idx < 3:
                plt.title("Monkey " + dset_name)

            plot_idx +=1

        fig.text(0.05, 0.7 - 0.4*event_idx, event, ha='center')
        #plt.suptitle(event)

    fig.text(0.5, 0.05, "Window Center Relative to Reference Event (ms)", ha='center')
    fig.text(0.08, 0.5, "ROC AUC", ha='center', 
             rotation='vertical')
    
    plot_idx = 0
    fig2 = plt.figure(figsize=(12,5))
    dset_names = [d['name'] for d in run_info.values()]
    for i, dset_name in enumerate(dset_names):
        plt.subplot(1, 3, plot_idx+1)
        plt.plot(win_centers, all_dataset_scores[dset_name][:,2])
        #plt.ylabel("ROC AUC")j
        plt.plot(win_centers, np.ones_like(win_centers)*0.5, 'k')
        plt.ylim([0.3, 0.8])
        if plot_idx < 3:
            plt.title("Monkey " + dset_name)

        plot_idx +=1

    fig2.text(0.5, 0.02, "Window Center Relative to Speed Maxima (ms)", 
            ha='center')
    fig2.text(0.08, 0.5, "ROC AUC", ha='center', 
             rotation='vertical')
    
    fig.savefig("../figures/final_figures/rates_firstmove_corrections_auc.svg")
    fig2.savefig("../figures/final_figures/rates_maxima_auc.svg")

    # plt.savefig(snakemake.output.all_datasets)

    # plt.figure()
    # plt.plot(win_centers, combined_scores)
    # plt.xlabel("Window Center Relative to Movement (ms)")
    # plt.ylabel("ROC AUC")
    # plt.title('First Movement')
    # #plt.savefig(snakemake.output.combined)

    # combined_scores = []
    # for win_idx,(win_start, win_stop) in enumerate(win_lims):
    #     win = np.round((win_stop-win_start)/dt).astype(int)
    #     combined_scores.append(combined_dataset_score([all_background[k][win_idx] for k in [v['name'] for v in run_info.values()]],
    #                                                   [all_corrections[k][win_idx] for k in [v['name'] for v in run_info.values()]],
    #                                                   all_controllers,
    #                                                   win))
 
    # # plt.savefig(snakemake.output.all_datasets)

    # plt.figure()
    # plt.plot(win_centers, combined_scores)
    # plt.xlabel("Window Center Relative to Movement (ms)")
    # plt.ylabel("ROC AUC")
    # plt.title('Corrections')
    # #plt.savefig(snakemake.output.combined)
    
    
    # combined_scores = []
    # for win_idx,(win_start, win_stop) in enumerate(win_lims):
    #     win = np.round((win_stop-win_start)/dt).astype(int)
    #     combined_scores.append(combined_dataset_score([all_background[k][win_idx] for k in [v['name'] for v in run_info.values()]],
    #                                                   [all_maxima[k][win_idx] for k in [v['name'] for v in run_info.values()]],
    #                                                   all_controllers,
    #                                                   win))
 
    # # plt.savefig(snakemake.output.all_datasets)

    # plt.figure()
    # plt.plot(win_centers, combined_scores)
    # plt.xlabel("Window Center Relative to Movement (ms)")
    # plt.ylabel("ROC AUC")
    # plt.title('Maxima')
    # #plt.savefig(snakemake.output.combined)