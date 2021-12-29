import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from ast import literal_eval
import os
import sys
import yaml
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
import model_evaluation as me
import optimize_target_prediction as otp
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 20
#plt.rcParams['axes.labelweight'] = 'bold'
#plt.rcParams['font.weight'] = 'bold'

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

run_info = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../../lfads_file_locations.yml'), 'r'))
# def plot_window_perf(output, use_rates, event_label, color):
#     #output['total_test_score'] = output[['mean_test_x_score','mean_test_y_score']].mean(1)
#     #output['std_total_test_score'] = output[['std_test_x_score','std_test_y_score']].mean(1)
#     if use_rates:
#         output = output.query('use_rates & fit_direction & reference=="hand"')
#     else:
#         output = output.query('~use_rates & fit_direction & reference =="hand"')

#     score_name = 'cosine_score'

#     win_lims = set(output['win_lim'])
#     win_centers = []
#     win_performances = []
#     win_stds = []
#     for win_lim in win_lims:
#         win_start, win_stop = literal_eval(win_lim)
#         # if win_stop - win_start != cfg['window_compare_length']:
#         #     continue
    
#         win_performance = output.query('win_lim==@win_lim')[['dataset', 'win_lim', 'mean_test_%s'%score_name]].groupby(['dataset']).max()['mean_test_%s'%score_name].mean() #average over datasets
#         win_performances.append(win_performance)

#         #indices of maximum score for each dataset
#         idxmax = output.query('win_lim==@win_lim').groupby('dataset')['mean_test_%s'%score_name].idxmax()
#         #computing mean std across monkeys
#         #win_std = output.loc[idxmax]['std_test_var_weighted_score'].values.mean()win_std = output.loc[idxmax]['std_test_var_weighted_score'].values.mean()
#         win_std = output.loc[idxmax]['std_test_%s'%score_name].values.mean()
#         win_stds.append(win_std)
#         win_centers.append(1000 * (win_start + (win_stop-win_start)/2))

#     idx = np.argsort(win_centers)
#     win_centers = np.array(win_centers)[idx]
#     win_stds = np.array(win_stds)[idx]
#     win_performances = np.array(win_performances)[idx]
#     plt.plot(win_centers, win_performances, color=color)
#     plt.fill_between(win_centers, win_performances-win_stds, win_performances+win_stds, color=color, alpha=0.2)
#     plt.ylabel('Decoder Performance ($\mathregular{r^2}$)')
#     #plt.xlabel('Window Relative to %s (ms)'%event_label)
#     #plt.ylim([0, .8])
    
    
def plot_window_perf(output, use_rates, event_label, color):
    #output['total_test_score'] = output[['mean_test_x_score','mean_test_y_score']].mean(1)
    #output['std_total_test_score'] = output[['std_test_x_score','std_test_y_score']].mean(1)
    if use_rates:
        output = output.query('use_rates & ~fit_direction & reference=="hand"')
    else:
        output = output.query('~use_rates & ~fit_direction & reference =="hand"')

    score_name = 'var_weighted_score'

    test_peak_dfs = []
    dset_names = []
    dfs = []
    cos = []
    trial_lens = []
    dts = []
    datasets = list(run_info.keys())
    for dataset in datasets:
        lfads_params = open(os.path.join(os.path.dirname(__file__), '../../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric']))).read().strip()
        data_filename = os.path.join(os.path.dirname(__file__), '../../data/intermediate/' + dataset + '.p')
        lfads_filename = os.path.join(os.path.dirname(__file__), '../../data/model_output/' + '_'.join([dataset, lfads_params, 'all.h5']))
        inputInfo_filename = os.path.join(os.path.dirname(__file__), '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat']))
        peak_filename = os.path.join(os.path.dirname(__file__), '../../data/peaks/' + '_'.join([dataset, 'targets-not-one_test.p']))
        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)
        test_peak_dfs.append(pd.read_pickle(os.path.join(os.path.dirname(__file__), '../../data/peaks/%s_targets-not-one_test.p'%dataset)))
        dfs.append(df)
        cos.append(co)
        trial_lens.append(trial_len)
        dts.append(dt)
        dset_names.append(run_info[dataset]['name'])

    win_lims = list(set(output['win_lim']))
    win_starts = []
    win_performances = []
    win_stds = []
    nruns = len(win_lims) * len(datasets)
    k=0
    for win_lim in win_lims:
        win_start, win_stop = literal_eval(win_lim)
        # if win_stop - win_start != cfg['window_compare_length']:
        #     continue
        dset_win_performances = []
        dset_win_vars = []
        for dataset, dset_name, test_peak_df, df, co, trial_len, dt in zip(datasets, dset_names, test_peak_dfs, dfs, cos, trial_lens, dts):
            #indices of maximum score for each dataset
            idxmax = output.query('win_lim==@win_lim & dataset==@dset_names')['mean_test_%s'%score_name].idxmax()
            model_row = output.loc[idxmax]
            preprocess_dict, model = me.get_row_params(model_row)

            X, y = otp.get_inputs_to_model(test_peak_df, co, trial_len, 
                                                dt, df, **preprocess_dict)
            scorer = make_scorer(otp.var_weighted_score_func)
            cv_results = cross_val_score(model, X, y, 
                                            scoring=scorer, 
                                            cv=cfg['target_prediction_cv_splits'])
            dset_win_performances.append(np.mean(cv_results))
            dset_win_vars.append(np.var(cv_results))
            k+=1
            print('run %d of %d completed'%(k,nruns))
        #computing mean std across monkeys
        win_performances.append(np.mean(dset_win_performances))
        win_stds.append(np.sqrt(np.sum(dset_win_vars)))
        win_starts.append(1000 * win_start)

    idx = np.argsort(win_starts)
    win_starts = np.array(win_starts)[idx]
    win_stds = np.array(win_stds)[idx]
    win_performances = np.array(win_performances)[idx]
    plt.plot(win_starts, win_performances, color=color)
    plt.fill_between(win_starts, win_performances-win_stds, win_performances+win_stds, color=color, alpha=0.2)
    plt.ylabel('Decoder Performance ($\mathregular{r^2}$)')
    plt.xticks(np.array(win_starts).astype(int))
    #plt.xlabel('Window Relative to %s (ms)'%event_label)
    #plt.ylim([0, .8])
    
if __name__=='__main__':
    initial_filename = os.path.join(os.path.dirname(__file__), "../../data/peaks/window_comparison_targets-not-one.csv")
    initial_output = pd.read_csv(initial_filename)
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))

    colors = utils.contrasting_colors(**cfg['colors']['window_performance'])

    fig = plt.figure(figsize=(10,8))
    plot_window_perf(initial_output, use_rates=False, 
                    event_label='Target Appearance', color=colors[0])

    plot_window_perf(initial_output, use_rates=True, 
                    event_label='Target Appearance', color=colors[1])

    plt.xlabel("Start of Window Relative to Target Appearance (ms)", ha='center')
    fig.text(0.6, .45, 'Inferred Inputs', color=colors[0])
    fig.text(0.7, .83, 'Firing Rates', color=colors[1])

    plt.savefig(os.path.join(os.path.dirname(__file__), '../../figures/final_figures/numbered/5b.pdf'))