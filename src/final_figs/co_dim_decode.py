import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
import os
from scipy import io
sys.path.insert(0, '..')
import decode_lfads as dl
import utils

config_path = os.path.dirname(__file__) + '/../../config.yml'
cfg = yaml.safe_load(open(config_path, 'r'))

if __name__=='__main__':
    n_splits = 5
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    datasets = [list(run_info.keys())[0]]
    datasets = run_info.keys()
    params = []
    params_no_co = []
    for dataset in datasets:
        params.append(open('../../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read())
        selected_kl_weight = 2.0#run_info[dataset]['params'][params[-1]]['param_values']['kl_co_weight']
        for dset_param_hash in run_info[dataset]['params'].keys():
            dset_param_dict = run_info[dataset]['params'][dset_param_hash]['param_values']
            if dset_param_dict['co_dim'] == 0 and dset_param_dict['kl_co_weight'] == selected_kl_weight:
                params_no_co.append(dset_param_hash)

    plt.figure(figsize=(15,5))
    plot_dfs = []
    for i, (dataset, param, param_no_co) in enumerate(zip(datasets, params, params_no_co)):
        data_filename = '../../data/intermediate/' + dataset + '.p'
        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
        inputInfo_filename = '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)

        with h5py.File(lfads_filename) as h5file:
            X_lfads = dl.get_lfads_predictor(h5file['factors'][:])

        X_smoothed = dl.get_smoothed_rates(df, trial_len, dt, 
                                        kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                                        used_inds=None)
        Y = dl.get_kinematics(df, trial_len, dt,
                            kinematic_vars=['x', 'y', 'x_vel', 'y_vel'], 
                            used_inds=None)
        rs, _ = dl.get_rs(X_lfads, Y, n_splits, kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                        use_reg=False, regularizer='ridge', alpha=1, random_state=None)

        mean_rs = [np.mean([rs[k][i] for k in rs.keys()]) for i in range(n_splits)]

        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param_no_co, 'all.h5'])
        with h5py.File(lfads_filename) as h5file:
            X_lfads_no_co = dl.get_lfads_predictor(h5file['factors'][:])

        rs_no_co, _ = dl.get_rs(X_lfads_no_co, Y, n_splits, kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                    use_reg=False, regularizer='ridge', alpha=1, random_state=None)

        mean_rs += [np.mean([rs_no_co[k][i] for k in rs_no_co.keys()]) 
                    for i in range(n_splits)]

        rs_smoothed, _ = dl.get_rs(X_smoothed, Y, n_splits, kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                    use_reg=False, regularizer='ridge', alpha=1, random_state=None)

        mean_rs += [np.mean([rs_smoothed[k][i] for k in rs_smoothed.keys()]) 
                    for i in range(n_splits)]

        predictor = ['LFADS Factors\nwith Controller']*n_splits + ['LFADS Factors\nNo Controller']*n_splits + ['Firing Rates']*n_splits

        plot_df = pd.DataFrame.from_dict({'Predictor':predictor,'Variance Explained':mean_rs})
        plot_dfs.append(plot_df)

    for i, (dataset,plot_df) in enumerate(zip(datasets, plot_dfs)):
        ax = plt.subplot(1, len(datasets), i+1)
        sns.pointplot(x='Predictor', y='Variance Explained', 
                        data=plot_dfs[i], markers='.')
        plt.title("Monkey " + run_info[dataset]['name'])
        plt.ylim([0.3, 1.0])
        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.xlabel('')
        ax.set_xticklabels(['LFADS\nwith\nController', 
                            'LFADS\nNo\nController', 
                            'Firing\nRates'])
    
    plt.savefig("../../figures/final_figures/kinematic_decoding.png")
