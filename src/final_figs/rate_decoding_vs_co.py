import pandas as pd
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt

run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))

if __name__=='__main__':
    initial_filename = '../../data/peaks/gaussian_gini_firstmove_window_search_rates.csv'
    correction_filename = '../../data/peaks/gaussian_gini_correction_window_search_rates.csv'

    initial_search_filename = '../../data/peaks/gaussian_gini_firstmove_window_search.csv'
    correction_search_filename = '../../data/peaks/gaussian_gini_correction_window_search.csv'

    initial_output = pd.read_csv(initial_filename)
    corr_output = pd.read_csv(correction_filename)

    search_initial = pd.read_csv(initial_search_filename)
    search_correction = pd.read_csv(correction_search_filename)

    initial_output['total_test_score'] = initial_output[['mean_test_x_score','mean_test_y_score']].mean(1)
    corr_output['total_test_score'] = corr_output[['mean_test_x_score','mean_test_y_score']].mean(1)

    search_initial['total_test_score'] = search_initial[['mean_test_x_score','mean_test_y_score']].mean(1)
    search_correction['total_test_score'] = search_correction[['mean_test_x_score','mean_test_y_score']].mean(1)    

    datasets = [d['name'] for d in run_info.values()]

    initial_rate_pcs = np.sort(list(set(initial_output['rate_pcs'].values)))
    corr_rate_pcs = np.sort(list(set(initial_output['rate_pcs'].values)))
    assert(np.all(corr_rate_pcs==initial_rate_pcs))
    rate_pcs = initial_rate_pcs
    corr_performance = {d:np.zeros(len(rate_pcs)) for d in datasets}
    initial_performance = {d:np.zeros(len(rate_pcs)) for d in datasets}
    for i,pc in enumerate(rate_pcs):
        for d in datasets:
            initial_performance[d][i] = initial_output.query('rate_pcs == %d & use_rates'%pc).groupby('dataset').max()['total_test_score'].loc[d]
            corr_performance[d][i] = corr_output.query('rate_pcs == %d & use_rates'%pc).groupby('dataset').max()['total_test_score'].loc[d]

    plt.figure()
    for j,d in enumerate(datasets):
        plt.subplot(1, len(datasets), j+1)
        plt.plot(rate_pcs, initial_performance[d])
        initial_co_performance = search_initial[['dataset','total_test_score']].groupby('dataset').max()['total_test_score'].loc[d]
        plt.plot(rate_pcs, np.ones_like(rate_pcs)*initial_co_performance)
        plt.ylim([0,0.7])
        plt.ylabel('Decoder Performance (r^2)')
        if j==1:
            plt.xlabel('Number of PCs Used for Rate Decoding')
        plt.title(d)

    plt.legend(['Rate Decoding', 'Controller Decoding'])
    plt.suptitle('Initial Movement')
    plt.savefig('../../figures/final_figures/rate_decoding_initial.svg')
    plt.figure()
    for j,d in enumerate(datasets):
        plt.subplot(1, len(datasets), j+1)
        plt.plot(rate_pcs, corr_performance[d])
        correction_co_performance = search_correction[['dataset','total_test_score']].groupby('dataset').max()['total_test_score'].loc[d]
        plt.plot(rate_pcs, np.ones_like(rate_pcs)*correction_co_performance)
        plt.legend(['Rate Decoding', 'Controller Decoding'])
        plt.ylim([0,0.5])
        plt.ylabel('Decoder Performance (r^2)')
        if j==1:
            plt.xlabel('Number of PCs Used for Rate Decoding')
        plt.title(d)

    plt.suptitle('Corrective Movement')
    plt.legend(['Rate Decoding', 'Controller Decoding'])
    plt.savefig('../../figures/final_figures/rate_decoding_corretive.svg')