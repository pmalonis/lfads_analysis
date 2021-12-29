import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from ast import literal_eval
import os
import sys
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 20
#plt.rcParams['axes.labelweight'] = 'bold'
#plt.rcParams['font.weight'] = 'bold'

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def plot_window_perf(output, use_rates, event_label, color):
    output['total_test_score'] = output[['mean_test_x_score','mean_test_y_score']].mean(1)
    output['std_total_test_score'] = output[['std_test_x_score','std_test_y_score']].mean(1)
    if use_rates:
        output = output.query('use_rates')
    else:
        output = output.query('~use_rates')

    win_lims = set(output['win_lim'])
    win_centers = []
    win_performances = []
    win_stds = []
    for win_lim in win_lims:
        win_start, win_stop = literal_eval(win_lim)
        if win_stop - win_start != cfg['window_compare_length']:
            continue
    
        win_performance = output.query('win_lim==@win_lim')[['dataset', 'win_lim', 'total_test_score']].groupby(['dataset']).max()['total_test_score'].mean() #average over datasets
        win_performances.append(win_performance)

        #indices of maximum score for each dataset
        idxmax = output.query('win_lim==@win_lim').groupby('dataset')['total_test_score'].idxmax()
        #computing mean std across monkeys
        win_std = output.loc[idxmax]['std_total_test_score'].values.mean()
        win_stds.append(win_std)
        win_centers.append(win_start + (win_stop-win_start)/2)

    idx = np.argsort(win_centers)
    win_centers = np.array(win_centers)[idx]
    win_stds = np.array(win_stds)[idx]
    win_performances = np.array(win_performances)[idx]
    plt.plot(win_centers, win_performances, color=color)
    plt.fill_between(win_centers, win_performances-win_stds, win_performances+win_stds, color=color, alpha=0.2)
    plt.ylabel('Decoder Performance (r^2)')
    #plt.xlabel('Window Relative to %s (ms)'%event_label)
    plt.ylim([0, .8])
    
if __name__=='__main__':
    initial_filename = "../../data/peaks/params_search_targets-not-one.csv"#snakemake.input[0]
    correction_filename = "../../data/peaks/params_search_corrections.csv"#snakemake.input[1]
    initial_output = pd.read_csv(initial_filename)
    correction_output = pd.read_csv(correction_filename)

    config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))

    colors = utils.contrasting_colors(**cfg['colors']['window_performance'])

    fig = plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plot_window_perf(initial_output, use_rates=False, 
                    event_label='Target Appearance', color=colors[0])
    plt.title('Initial Movement')
    plt.subplot(1,2,2)
    plt.title('Corrective Movement')
    plot_window_perf(correction_output, use_rates=False, 
                    event_label='Target Appearance', color=colors[0])
    plt.subplot(1,2,1)
    plot_window_perf(initial_output, use_rates=True, 
                    event_label='Target Appearance', color=colors[1])
    plt.title('Initial Movement')
    plt.subplot(1,2,2)
    plt.title('Corrective Movement')
    plot_window_perf(correction_output, use_rates=True, 
                    event_label='Target Appearance', color=colors[1])
    plt.legend(['Controller', 'Rates'])
    fig.text(0.5, 0.02, "Window Center Relative to Target Appearance (ms)", ha='center')
    # fig.text(0.02, .75, 'Controller')
    # fig.text(0.02, .25, 'Rates')

    plt.savefig('../../figures/final_figures/window_performance.png')
    #plt.savefig(snakemake.output[0])