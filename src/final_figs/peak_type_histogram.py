from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import io
import h5py
import pandas as pd
import matplotlib.pyplot as plt

if __name__=='__main__':
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    datasets = run_info.keys()
    params = []
    for dataset in datasets:
        params.append(open('../data/peaks/param_%s_thresh=0.3.txt'%(dataset)).read())

    dataset_names = [run_info[dataset]['name'] for dataset in datasets]
    dataset_names = datasets
    for dataset, param in zip(datasets, params):
        data_filename = '../data/intermediate/' + dataset + '.p'
        lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
        inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])

    df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)
    thresholds = [0.1,0.2,0.3,0.4]
    peak_dfs = []
    for threshold in thresholds:
        peak_path = '../data/peaks/%s_%s_peaks_all_thresh=%0.1f.p'%(dataset, param, threshold)
        if False:#os.path.exists(peak_path):
            peak_dfs.append(pd.read_pickle(peak_path))
        else:
            peak_dfs.append(ta.get_peak_df(df, co, trial_len, threshold, dt, win_start=win_start, win_stop=win_stop))

    bardf=pd.DataFrame({'input 1':[(peak_dfs[i].latency_0.notnull() & peak_dfs[i].latency_1.isnull()).sum() for i in range(len(peak_dfs))],
    'input 2':[(peak_dfs[i].latency_1.notnull() & peak_dfs[i].latency_0.isnull()).sum() for i in range(len(peak_dfs))],
    'both':[(peak_dfs[i].latency_1.notnull() & peak_dfs[i].latency_0.notnull()).sum() for i in range(len(peak_dfs))],
    'neither':[(peak_dfs[i].latency_1.isnull() & peak_dfs[i].latency_0.isnull()).sum() for i in range(len(peak_dfs))]},
    index=peak_thresholds)

    cmap = matplotlib.cm.get_cmap('viridis')
    rgba = cmap(np.linspace(0,1,len(peak_thresholds)))
    ax = bardf.plot.barh(stacked=True, color=rgba)
    fig = ax.get_figure()
    plt.savefig('../../figures/final_figures/peak_type_histogram_%s.png'%dataset)