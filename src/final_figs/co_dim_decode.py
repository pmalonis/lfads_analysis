import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
import os
from scipy import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import decode_lfads as dl
import utils
plt.rcParams['font.size'] = 20
plt.rcParams['pdf.fonttype'] = 42

config_path = '../../config.yml'
cfg = yaml.safe_load(open(config_path, 'r'))

if __name__=='__main__':
    n_splits = 5
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    datasets = run_info.keys()
    params = []
    params_no_co = []
    for dataset in datasets:
        params.append(open('../../data/peaks/%s_selected_param_%s.txt'%(dataset,cfg['selection_metric'])).read())
        selected_kl_weight = run_info[dataset]['params'][params[-1]]['param_values']['kl_co_weight']
        for dset_param_hash in run_info[dataset]['params'].keys():
            dset_param_dict = run_info[dataset]['params'][dset_param_hash]['param_values']
            if dset_param_dict['co_dim'] == 0 and dset_param_dict['kl_co_weight'] == selected_kl_weight:
                params_no_co.append(dset_param_hash)

    plt.figure(figsize=(8, 8))
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
        
        n_trials = df.index[-1][0] + 1
        rs, _ = dl.get_rs(X_lfads, Y, n_splits, trial_len, dt, n_trials, kinematic_vars=['x', 'y', 'x_vel', 'y_vel'], 
                        use_reg=False, regularizer='ridge', alpha=1, random_state=None)

        mean_rs = [np.mean([rs[k][i] for k in rs.keys()])  for i in range(n_splits)]

        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param_no_co, 'all.h5'])
        with h5py.File(lfads_filename) as h5file:
            X_lfads_no_co = dl.get_lfads_predictor(h5file['factors'][:])

        rs_no_co, _ = dl.get_rs(X_lfads_no_co, Y, n_splits, trial_len, dt, n_trials, kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                    use_reg=False, regularizer='ridge', alpha=1, random_state=None)

        mean_rs += [np.mean([rs_no_co[k][i] for k in rs_no_co.keys()]) 
                    for i in range(n_splits)]

        rs_smoothed, _ = dl.get_rs(X_smoothed, Y, n_splits, trial_len, dt, n_trials,  kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                    use_reg=False, regularizer='ridge', alpha=1, random_state=None)

        mean_rs += [np.mean([rs_smoothed[k][i] for k in rs_smoothed.keys()]) for i in range(n_splits)]
        predictor = ['LFADS Factors\nwith Controller']*n_splits + ['LFADS Factors\nNo Controller']*n_splits + ['Firing Rates']*n_splits
        monkey = len(predictor) * [run_info[dataset]['name']]
        plot_df = pd.DataFrame.from_dict({'Predictor':predictor,'Variance Explained':mean_rs, 'Monkey':monkey})
        plot_dfs.append(plot_df)

    all_plot_df = pd.concat(plot_dfs)
    
    for predictor in set(all_plot_df['Predictor']):
        print(predictor.replace('\n',' ') + ':')
        monkey_means = all_plot_df.query('Predictor==@predictor').groupby('Monkey').mean()
        monkey_std = all_plot_df.query('Predictor==@predictor').groupby('Monkey').std()
        monkey_means = pd.concat([monkey_means, monkey_std], axis=1)
        monkey_means.columns = ['Mean', 'Standard Deviation']
        print(monkey_means)
        print('\t All mean: %f'%monkey_means['Mean'].values.mean())
        print('\t All standard deviation: %f'%monkey_means['Mean'].values.std())
        print('\t All standard error: %f'%(monkey_means['Mean'].values.std()/monkey_means['Mean'].shape[0]))

            
    #ax = plt.subplot(1, len(datasets), i+1)
    # g = sns.pointplot(x='Predictor', y='Variance Explained', 
    #                     data=plot_dfs[i], markers='')
    g = sns.pointplot(x='Predictor', y='Variance Explained', 
                        data=all_plot_df, hue='Monkey')
    plt.setp(g.axes.lines, linewidth=1.5)
    plt.locator_params(nbins=5)
   #plt.title("Monkey " + run_info[dataset]['name'])
    #plt.ylim([0.35, 0.85])
    #plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel('')
    plt.ylabel('Decoding Performance ($\mathregular{r^2}$)')
    ax=plt.gca()
    ax.set_xticklabels(['LFADS\nwith\nController',
                        'LFADS\nNo\nController', 
                        'Firing\nRates'])
    
    plt.gcf().tight_layout()
    plt.savefig("../../figures/final_figures/kinematic_decoding.svg")
    plt.savefig("../../figures/final_figures/numbered/2b.pdf")
    plt.savefig("../../figures/final_figures/kinematic_decoding.png")
