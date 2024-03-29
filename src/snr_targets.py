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
from sklearn.utils import resample
plt.rcParams['font.size'] = 18

roc_bootstrap_repeats: 200
background_subsample: 0.05 

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def bootstrap_ci(background, event, n=cfg['roc_bootstrap_repeats'], 
                                    background_p=cfg['background_subsample'],
                                    alpha=cfg['roc_ci_alpha']):
    '''
    background: background controller window magnitudes
    event: event controller window magnitudes
    n: number of repeated resamplings to perform
    alpha: confidence interval level
    background_p: proportion of background in resampled set sizze 

    returns
    confidence interval of auc roc estimate
    '''
    dist = np.zeros(n)
    n_samples_bg= int(len(background)*background_p)
    for i in range(n):
        #resample(background)
        #resample(event)
        background_sample = resample(background, n_samples=n_samples_bg)
        event_sample = resample(event)
        dist[i] = rocauc(background_sample, event_sample)
                        
    dist = np.sort(dist)
    idx_lower = np.round(n*alpha/2).astype(int) - 1
    idx_upper = np.round(n*(1-alpha/2)).astype(int)
    
    return dist[idx_lower], dist[idx_upper]

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
        background_co = [np.abs(co[i,i:i+nwin,:]).sum() for a,b in zip(t_stop, t_start) for i in range(a,b-nwin)] #t_stop is first to exclude peri-event windows
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

def all_ci(background, firstmove, corrections, maxima):
    firstmove_ci = bootstrap_ci(background, firstmove)
    corrections_ci = bootstrap_ci(background, corrections)
    maxima_ci = bootstrap_ci(background, maxima)

    return firstmove_ci, corrections_ci, maxima_ci

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

    run_info = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../lfads_file_locations.yml'), 'r'))
    datasets = list(run_info.keys())
    win_lims = [literal_eval(w) for w in cfg['target_auc_win_lims']]
    win_centers = [1000*(start + (stop-start)/2) for start,stop in win_lims]
    all_dataset_scores = {}
    all_dataset_ci = {}
    all_background = {}
    all_firstmove = {}
    all_corrections = {}
    all_maxima = {}
    all_controllers = []
    for dataset in datasets:
        param = open(os.path.join(os.path.dirname(__file__), '../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric']))).read().strip()
        data = pd.read_pickle(os.path.join(os.path.dirname(__file__), '../data/intermediate/%s.p'%dataset))
        firstmove_df = pd.read_pickle(os.path.join(os.path.dirname(__file__), '../data/peaks/%s_targets-not-one_all.p'%dataset))
        corr_df = pd.read_pickle(os.path.join(os.path.dirname(__file__), '../data/peaks/%s_corrections_all.p'%dataset))
        maxima_df = pd.read_pickle(os.path.join(os.path.dirname(__file__), '../data/peaks/%s_maxima_all.p'%dataset))
        input_info = io.loadmat(os.path.join(os.path.dirname(__file__), '../data/model_output/%s_inputInfo.mat'%dataset))
        
        with h5py.File(os.path.join(os.path.dirname(__file__), '../data/model_output/%s_%s_all.h5'%(dataset,param)),'r') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        all_background[run_info[dataset]['name']] = []
        all_firstmove[run_info[dataset]['name']] = []
        all_corrections[run_info[dataset]['name']] = []
        all_maxima[run_info[dataset]['name']] = []
        dataset_scores = []
        dataset_ci = []
        for win_start, win_stop in win_lims:
            background, firstmove, corrections, maxima = all_event_data(data, firstmove_df, corr_df, maxima_df, co, dt, trial_len, win_start, win_stop)
            win_score = all_rocauc(background, firstmove, corrections, maxima)
            win_ci = all_ci(background, firstmove, corrections, maxima)
            dataset_scores.append(win_score)
            dataset_ci.append(win_ci)
            all_background[run_info[dataset]['name']].append(background)
            all_firstmove[run_info[dataset]['name']].append(firstmove)
            all_corrections[run_info[dataset]['name']].append(corrections)
            all_maxima[run_info[dataset]['name']].append(maxima)

        all_dataset_scores[run_info[dataset]['name']] = np.array(dataset_scores)
        all_dataset_ci[run_info[dataset]['name']] = np.array(dataset_ci)
        all_controllers.append(co)

    plot_idx = 0
    titles = ['']#['First\n Movement', 'Correction']
    fig = plt.figure(figsize=(15,6.5))
    for event_idx, event in enumerate(titles):
        dset_names = [d['name'] for d in run_info.values()]
        for i, dset_name in enumerate(dset_names):
            plt.subplot(1, 3, plot_idx+1)
            win_rocs = np.array(all_dataset_scores[dset_name][:,event_idx])
            win_cis = np.array(all_dataset_ci[dset_name][:,event_idx])
            plt.plot(win_centers, all_dataset_scores[dset_name][:,event_idx])
            plt.fill_between(win_centers, win_cis[:,0], win_cis[:,1], alpha=0.2)
            print('%s peak ROC: %f'%(dset_name, np.max(all_dataset_scores[dset_name][:,event_idx])))
            if i == 0:
                plt.ylabel("ROC AUC")
            else:
                plt.gca().set_yticklabels([])
            plt.plot(win_centers, np.ones_like(win_centers)*0.5, 'k')
            plt.ylim([0.2, 0.9])
            if plot_idx < 3:
                plt.title("Monkey " + dset_name)
            plt.yticks(fontsize=14)
            plt.xticks([-200, -100, 0, 100, 200], fontsize=14)
            plot_idx +=1

        #fig.text(0.05, 0.7 - 0.4*event_idx, event, ha='center')
        #plt.suptitle(event)

    fig.text(0.5, 0.01, "Window Center Relative to Target Appearance (ms)", ha='center')

    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), '../figures/final_figures/auc_targets.svg'))
    fig.savefig(os.path.join(os.path.dirname(__file__), '../figures/final_figures/numbered/4d.pdf'))

    # combined_scores = []
    # for win_idx,(win_start, win_stop) in enumerate(win_lims):
    #     win = np.round((win_stop-win_start)/dt).astype(int)
    #     combined_scores.append(combined_dataset_score([all_background[k][win_idx] for k in [v['name'] for v in run_info.values()]],
    #                                                   [all_firstmove[k][win_idx] for k in [v['name'] for v in run_info.values()]],
    #                                                   all_controllers,
    #                                                   win))
 
    # # plt.savefig(snakemake.output.all_datasets)

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