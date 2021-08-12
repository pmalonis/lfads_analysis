import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from ast import literal_eval
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'
#plt.rcParams['font.weight'] = 'bold'

def plot_window_perf(output, use_rates):
    output['total_test_score'] = output[['mean_test_x_score','mean_test_y_score']].mean(1)
    if use_rates:
        output = output.query('use_rates')
    else:
        output = output.query('~use_rates')

    win_lims = set(output['win_lim'])
    win_centers = []
    win_performances = []
    for win_lim in win_lims:
        win_start, win_stop = literal_eval(win_lim)
        if win_stop-win_start !=0.2:
            continue
    
        win_performance = output.query('win_lim==@win_lim')[['dataset', 'win_lim', 'total_test_score']].groupby(['dataset']).max().groupby(['win_lim']).mean()['total_test_score']
        if win_performance.shape[0]==0:
            continue
        
        win_centers.append(win_start + (win_stop-win_start)/2)
        win_performances.append(win_performance)

    idx = np.argsort(win_centers)
    plt.plot(np.array(win_centers)[idx], np.array(win_performances)[idx])
    plt.ylabel('Decoder Performance (r^2)')
    #plt.xlabel('Window Relative to Movement (ms)')
    plt.ylim([0, .8])
    
if __name__=='__main__':
    initial_filename = "../../data/peaks/params_search_firstmove.csv"#snakemake.input[0]
    correction_filename = "../../data/peaks/params_search_corrections.csv"#snakemake.input[1]
    initial_output = pd.read_csv(initial_filename)
    correction_output = pd.read_csv(correction_filename)

    fig = plt.figure(figsize=(12,10))
    plt.subplot(1,2,1)
    plot_window_perf(initial_output, use_rates=False)
    plt.title('Initial Movement')
    plt.subplot(1,2,2)
    plt.title('Corrective Movement')
    plot_window_perf(correction_output, use_rates=False)
    plt.subplot(1,2,1)
    plot_window_perf(initial_output, use_rates=True)
    plt.title('Initial Movement')
    plt.subplot(1,2,2)
    plt.title('Corrective Movement')
    plot_window_perf(correction_output, use_rates=True)
    plt.legend(['Controller', 'Rates'])
    fig.text(0.5, 0.02, "Window Center Relative to Movement (ms)", ha='center')
    # fig.text(0.02, .75, 'Controller')
    # fig.text(0.02, .25, 'Rates')

    #plt.savefig(snakemake.output[0])